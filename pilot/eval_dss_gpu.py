"""GPU version of eval_dss.py: the same four policies and the same explanation check,
with all test batches (and all 16 ground-truth branches per decision) simulated together.

Policies: recipe, fixed rule (sugar & oil x1.2), best fixed action (train_dss.py), DSS.
With the level controller (v3) no policy can overfill the vessel, so the DSS simply
recommends the candidate with the highest predicted 24 h gain (the volume model is
still used to drop candidates predicted to exceed the limit, as in v2).
Explanations M1 (TreeSHAP), M2 (delta-TreeSHAP), M3 (contrastive Shapley) are compared
with the Shapley game evaluated by the simulator.
Outputs: runs/<exp>/results/decisions.csv, runs/<exp>/results/batches.csv
"""
import argparse
import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from dss_common import (ACTION_MVS, DECISION_TIMES, LEVEL_CONTROL, OUT_DIR, RECIPE_ACTION, state_features_batch,
                        window_end)
from dss_policy import frame, recommend
from eval_dss import FIX_SO12, exact_shapley, mix
from jaxsim import BatchSet
from train_dss import FEATURES, MODELS_DIR, RESULTS_DIR

SUBSETS = [frozenset(c) for r in range(5) for c in itertools.combinations(ACTION_MVS, r)]


USAGE_KEYS = ["vmax", "sugar_L", "oil_L", "discharged_L", "air"]


def as_row(a):
    return [a[mv] for mv in ACTION_MVS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--first-seed", type=int, default=1)
    args = ap.parse_args()
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    n = len(seeds)
    ymodel = lgb.Booster(model_file=f"{MODELS_DIR}/surrogate_yield.txt")
    vmodel = lgb.Booster(model_file=f"{MODELS_DIR}/surrogate_vmax.txt")
    static_action = json.load(open(f"{MODELS_DIR}/static_action.json"))
    explainer = shap.TreeExplainer(ymodel)
    print(f"[{OUT_DIR}] {n} test batches, level control = {LEVEL_CONTROL}", flush=True)
    t0 = time.time()

    # --- recipe and the two fixed policies, simulated together ---
    policies = [("recipe", RECIPE_ACTION), ("fix_so12", FIX_SO12), ("fix_static", static_action)]
    fixed = BatchSet(seeds * len(policies), LEVEL_CONTROL)
    fixed.run_until(DECISION_TIMES[0])
    fixed.run_until(230, np.array([as_row(a) for _, a in policies for _ in seeds]))
    use = fixed.usage_since(0)
    out = pd.DataFrame({"seed": seeds})
    for p, (name, _) in enumerate(policies):
        sl = slice(p * n, (p + 1) * n)
        out[f"{name}_yield"] = fixed.total_yield[sl]
        for key in USAGE_KEYS:
            out[f"{name}_{key}"] = use[key][sl]
    print(f"  fixed policies done ({time.time() - t0:.0f}s)", flush=True)

    # --- DSS ---
    main_bs, mult, rows = BatchSet(seeds, LEVEL_CONTROL), None, []
    for t_dec in DECISION_TIMES:
        main_bs.run_until(t_dec, mult)
        feats = state_features_batch(main_bs).to_dict("records")
        recs = recommend(feats, ymodel, vmodel)
        chosen, plan, dec_rows = [], [], []
        for i, (s, rec) in enumerate(zip(feats, recs)):
            action, changed = rec["action"], rec["changed"]
            row = {"seed": seeds[i], "t_dec": t_dec, "changed": changed, "n_feasible": rec["n_feasible"],
                   **{f"a_{mv}": action[mv] for mv in ACTION_MVS}, "pred_gain": rec["pred_gain"],
                   "pred_vmax": float(vmodel.predict(frame(s, [action]))[0])}
            if changed:
                sub_pred = dict(zip(SUBSETS, ymodel.predict(frame(s, [mix(action, S) for S in SUBSETS]))))
                m3 = exact_shapley(lambda S: sub_pred[S] - sub_pred[frozenset()])
                sv = explainer.shap_values(frame(s, [action, RECIPE_ACTION]))
                m1 = {mv: float(sv[0][FEATURES.index(f"a_{mv}")]) for mv in ACTION_MVS}
                m2 = {mv: m1[mv] - float(sv[1][FEATURES.index(f"a_{mv}")]) for mv in ACTION_MVS}
                for name, vec in [("m1", m1), ("m2", m2), ("m3", m3)]:
                    row.update({f"{name}_{mv}": vec[mv] for mv in ACTION_MVS})
                plan += [(i, len(dec_rows), S, mix(action, S)) for S in SUBSETS]
            dec_rows.append(row)
            chosen.append(action)

        # ground truth: all 16 subsets of every changed recommendation, simulated together
        if plan:
            # pad to a fixed n*16 branches so JAX compiles the branch simulation only once
            padded = plan + [plan[0]] * (n * len(SUBSETS) - len(plan))
            br = main_bs.fork([i for i, _, _, _ in padded])
            y0, k0 = br.total_yield, br.k
            br.run_until(window_end(t_dec), np.array([as_row(a) for _, _, _, a in padded]))
            gain = (br.total_yield - y0)[:len(plan)]
            vmax = br.usage_since(k0)["vmax"][:len(plan)]
            by_row = {}
            for j, (_, r, S, _) in enumerate(plan):
                by_row.setdefault(r, {})[S] = (gain[j], vmax[j])
            for r, sims in by_row.items():
                sim_val = {S: v[0] for S, v in sims.items()}
                gt = exact_shapley(lambda S: sim_val[S] - sim_val[frozenset()])
                dec_rows[r]["true_gain"] = sim_val[frozenset(ACTION_MVS)] - sim_val[frozenset()]
                dec_rows[r]["true_vmax"] = sims[frozenset(ACTION_MVS)][1]
                dec_rows[r].update({f"gt_{mv}": gt[mv] for mv in ACTION_MVS})
        rows += dec_rows
        mult = np.array([as_row(a) for a in chosen])
        print(f"  decision {t_dec:>3} h: {sum(r['changed'] for r in dec_rows)}/{n} recommendations changed, "
              f"{len(plan)} ground-truth branches ({time.time() - t0:.0f}s)", flush=True)
    main_bs.run_until(230, mult)
    use = main_bs.usage_since(0)
    out["dss_yield"] = main_bs.total_yield
    for key in USAGE_KEYS:
        out[f"dss_{key}"] = use[key]

    pd.DataFrame(rows).sort_values(["seed", "t_dec"]).to_csv(f"{RESULTS_DIR}/decisions.csv", index=False)
    out.to_csv(f"{RESULTS_DIR}/batches.csv", index=False)
    print(out[["seed"] + [f"{p}_yield" for p, _ in policies] + ["dss_yield"]].round(0).to_string(index=False))
    print(f"saved {RESULTS_DIR}/decisions.csv and batches.csv in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
