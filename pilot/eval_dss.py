"""Closed-loop evaluation of the explainable DSS on unseen test batches.

For each test seed, four policies are simulated over the whole batch:
  recipe      - the IndPenSim recipe, no changes
  fix_so12    - fixed rule: sugar and oil feed x1.2 from 96 h onwards
  fix_static  - best fixed action chosen by the yield model on training data (train_dss.py)
  dss         - the DSS: at each decision point it scores all candidate actions, drops
                those predicted to overfill the vessel, and recommends the best one
                (or keeps the recipe if no predicted gain).
Each DSS recommendation is explained three ways (attribution of the 4 action changes):
  M1 TreeSHAP      - standard SHAP of the prediction f(s, a*)        [common practice]
  M2 dTreeSHAP     - SHAP(s, a*) - SHAP(s, recipe)                   [naive contrast]
  M3 Contrastive   - exact Shapley of the predicted gain over recipe,
                     players = the 4 action changes                  [proposed]
Ground truth: the same Shapley game evaluated with the SIMULATOR (fork the batch,
simulate all 16 subsets of action changes for 24 h).
Outputs: runs/<exp>/results/decisions.csv, runs/<exp>/results/batches.csv
"""
import argparse
import itertools
import json
import math
import os
import time
import warnings
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

from dss_common import (ACTION_MVS, CANDIDATES, DECISION_TIMES, OUT_DIR, RECIPE_ACTION, V_LIMIT_L,
                        action_features, action_policy, state_features, usage_since, window_end)
from sim import Batch
from train_dss import FEATURES, MODELS_DIR, RESULTS_DIR

MIN_PRED_GAIN_KG = 0.0   # keep the recipe unless the model predicts a gain
FIX_SO12 = {"Fs": 1.2, "Foil": 1.2, "Fg": 1.0, "Fpaa": 1.0}
PARTS_DIR = f"{RESULTS_DIR}/parts"


def mix(action, subset):
    """Recipe action with only the MVs in `subset` changed to `action`."""
    return {mv: (action[mv] if mv in subset else 1.0) for mv in ACTION_MVS}


def exact_shapley(value):
    """Exact Shapley values for the 4 action changes given value(frozenset) -> float."""
    n = len(ACTION_MVS)
    phi = {}
    for mv in ACTION_MVS:
        others = [o for o in ACTION_MVS if o != mv]
        total = 0.0
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(others, r):
                S = frozenset(S)
                total += w * (value(S | {mv}) - value(S))
        phi[mv] = total
    return phi


def simulate_window(batch, action, t_dec):
    """(penicillin gained, max volume) in [t_dec, window end] under `action`, from a fork."""
    b = batch.fork()
    y0, k0 = b.total_yield, b.k
    t1 = window_end(t_dec)
    b.run_until(t1, action_policy(action, t_dec, t1))
    return b.total_yield - y0, usage_since(b, k0)["vmax"]


def run_fixed(seed, action):
    """Whole batch under a fixed action from the first decision point (None = recipe)."""
    b = Batch(seed)
    b.run_until(230, None if action is None else action_policy(action, DECISION_TIMES[0], 230))
    return b.total_yield, usage_since(b, 0)


def one_seed(seed):
    warnings.filterwarnings("ignore")
    import shap
    ymodel = lgb.Booster(model_file=f"{MODELS_DIR}/surrogate_yield.txt")
    vmodel = lgb.Booster(model_file=f"{MODELS_DIR}/surrogate_vmax.txt")
    static_action = json.load(open(f"{MODELS_DIR}/static_action.json"))
    explainer = shap.TreeExplainer(ymodel)

    def frame(state, actions):
        return pd.DataFrame([{**state, **action_features(a)} for a in actions])[FEATURES]

    def tree_shap(state, action):
        sv = explainer.shap_values(frame(state, [action]))[0]
        return {mv: float(sv[FEATURES.index(f"a_{mv}")]) for mv in ACTION_MVS}

    t0 = time.time()
    out = {"seed": seed}
    for name, action in [("recipe", None), ("fix_so12", FIX_SO12), ("fix_static", static_action)]:
        y, u = run_fixed(seed, action)
        out.update({f"{name}_yield": y, f"{name}_vmax": u["vmax"],
                    f"{name}_sugar_L": u["sugar_L"], f"{name}_oil_L": u["oil_L"]})

    batch, policy, rows = Batch(seed), None, []
    for t_dec in DECISION_TIMES:
        batch.run_until(t_dec, policy)
        s = state_features(batch)
        X = frame(s, CANDIDATES)
        preds = ymodel.predict(X, num_threads=1)
        vpred = vmodel.predict(X, num_threads=1)
        pred_recipe = float(ymodel.predict(frame(s, [RECIPE_ACTION]), num_threads=1)[0])
        feasible = vpred <= V_LIMIT_L
        if feasible.any():
            i_best = int(np.argmax(np.where(feasible, preds, -np.inf)))
            pred_gain = float(preds[i_best]) - pred_recipe
            action = CANDIDATES[i_best] if pred_gain > MIN_PRED_GAIN_KG else RECIPE_ACTION
        else:
            pred_gain, action = 0.0, RECIPE_ACTION
        changed = action != RECIPE_ACTION
        row = {"seed": seed, "t_dec": t_dec, "changed": changed, "n_feasible": int(feasible.sum()),
               **{f"a_{mv}": action[mv] for mv in ACTION_MVS},
               "pred_gain": pred_gain if changed else 0.0,
               "pred_vmax": float(vmodel.predict(frame(s, [action]), num_threads=1)[0])}
        if changed:
            # M3: contrastive Shapley on the surrogate (16 model calls)
            subsets = [frozenset(c) for r in range(5) for c in itertools.combinations(ACTION_MVS, r)]
            sub_pred = dict(zip(subsets, ymodel.predict(frame(s, [mix(action, S) for S in subsets]),
                                                        num_threads=1)))
            m3 = exact_shapley(lambda S: sub_pred[S] - sub_pred[frozenset()])
            # M1 / M2: standard TreeSHAP on the action features
            m1 = tree_shap(s, action)
            m1_recipe = tree_shap(s, RECIPE_ACTION)
            m2 = {mv: m1[mv] - m1_recipe[mv] for mv in ACTION_MVS}
            # Ground truth: same game, simulator as value function (16 forks x 24 h)
            sims = {S: simulate_window(batch, mix(action, S), t_dec) for S in subsets}
            sim_val = {S: v[0] for S, v in sims.items()}
            gt = exact_shapley(lambda S: sim_val[S] - sim_val[frozenset()])
            row["true_gain"] = sim_val[frozenset(ACTION_MVS)] - sim_val[frozenset()]
            row["true_vmax"] = sims[frozenset(ACTION_MVS)][1]
            for name, vec in [("m1", m1), ("m2", m2), ("m3", m3), ("gt", gt)]:
                row.update({f"{name}_{mv}": vec[mv] for mv in ACTION_MVS})
        rows.append(row)
        policy = action_policy(action, t_dec, window_end(t_dec))
    batch.run_until(230, policy)
    u = usage_since(batch, 0)
    out.update({"dss_yield": batch.total_yield, "dss_vmax": u["vmax"],
                "dss_sugar_L": u["sugar_L"], "dss_oil_L": u["oil_L"]})
    print(f"seed {seed}: recipe {out['recipe_yield']:.0f} | x1.2 rule {out['fix_so12_yield']:.0f} | "
          f"static {out['fix_static_yield']:.0f} | DSS {out['dss_yield']:.0f} kg "
          f"({time.time() - t0:.0f}s)", flush=True)
    return rows, out


def part_paths(seed):
    return f"{PARTS_DIR}/decisions_seed{seed}.csv", f"{PARTS_DIR}/batch_seed{seed}.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    # One pair of files per finished seed, so an interrupted run resumes where it stopped.
    os.makedirs(PARTS_DIR, exist_ok=True)
    todo = [s for s in seeds if not os.path.exists(part_paths(s)[1])]
    print(f"[{OUT_DIR}] {len(seeds) - len(todo)} seeds already done, {len(todo)} to run "
          f"on {args.workers} workers", flush=True)
    if todo:
        with Pool(args.workers) as pool:
            for r, b in pool.imap_unordered(one_seed, todo):
                dec_path, batch_path = part_paths(b["seed"])
                pd.DataFrame(r).to_csv(dec_path, index=False)
                pd.DataFrame([b]).to_csv(batch_path + ".tmp", index=False)
                os.replace(batch_path + ".tmp", batch_path)   # written last = seed complete
    dec = pd.concat([pd.read_csv(part_paths(s)[0]) for s in seeds])
    bat = pd.concat([pd.read_csv(part_paths(s)[1]) for s in seeds])
    dec.sort_values(["seed", "t_dec"]).to_csv(f"{RESULTS_DIR}/decisions.csv", index=False)
    bat.sort_values("seed").to_csv(f"{RESULTS_DIR}/batches.csv", index=False)
    print(f"saved {RESULTS_DIR}/decisions.csv and batches.csv ({len(seeds)} seeds)")


if __name__ == "__main__":
    main()
