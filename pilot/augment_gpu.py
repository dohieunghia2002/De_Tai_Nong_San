"""Targeted data collection around the DSS's own recommendations (experiment v4).

The surrogate is first trained on random actions, which rarely visit the region the DSS
actually uses (high feed and aeration, 120-144 h). Here the current DSS is run on the
TRAINING batches; at every decision point the 16 subsets of its recommended changes are
simulated for 24 h from the same state - one-change-at-a-time and all combinations -
and added to the training data. Test batches are never used.

Rows get kind "aug<round>". Run train_dss.py afterwards to refit the models.
"""
import argparse
import itertools
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from dss_common import ACTION_MVS, DECISION_TIMES, LEVEL_CONTROL, OUT_DIR, state_features_batch, window_end
from dss_policy import recommend
from eval_dss import mix
from jaxsim import BatchSet
from train_dss import MODELS_DIR

SUBSETS = [frozenset(c) for r in range(5) for c in itertools.combinations(ACTION_MVS, r)]


def as_row(a):
    return [a[mv] for mv in ACTION_MVS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=240)
    ap.add_argument("--first-seed", type=int, default=1000)
    args = ap.parse_args()
    kind = f"aug{args.round}"
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    n = len(seeds)
    path = f"{OUT_DIR}/data/train.csv"
    data = pd.read_csv(path)
    data = data[data.kind != kind]                  # re-running a round replaces its rows
    ymodel = lgb.Booster(model_file=f"{MODELS_DIR}/surrogate_yield.txt")
    vmodel = lgb.Booster(model_file=f"{MODELS_DIR}/surrogate_vmax.txt")
    print(f"[{OUT_DIR}] round {args.round}: DSS on {n} training batches, 16 branches per decision", flush=True)
    t0 = time.time()

    main_bs, mult, rows = BatchSet(seeds, LEVEL_CONTROL), None, []
    for t_dec in DECISION_TIMES:
        main_bs.run_until(t_dec, mult)
        feats = state_features_batch(main_bs).to_dict("records")
        recs = recommend(feats, ymodel, vmodel)
        plan = [(i, mix(r["action"], S)) for i, r in enumerate(recs) if r["changed"] for S in SUBSETS]
        if plan:
            padded = plan + [plan[0]] * (n * len(SUBSETS) - len(plan))   # fixed shape: compile once
            br = main_bs.fork([i for i, _ in padded])
            y0, k0 = br.total_yield, br.k
            br.run_until(window_end(t_dec), np.array([as_row(a) for _, a in padded]))
            dy = (br.total_yield - y0)[:len(plan)]
            use = {key: v[:len(plan)] for key, v in br.usage_since(k0).items()}
            for j, (i, a) in enumerate(plan):
                rows.append({"seed": seeds[i], "t_dec": t_dec, "kind": kind, **feats[i],
                             **{f"a_{mv}": a[mv] for mv in ACTION_MVS}, "dyield": dy[j],
                             "vmax_w": use["vmax"][j], "sugar_w": use["sugar_L"][j], "oil_w": use["oil_L"][j],
                             "discharged_w": use["discharged_L"][j]})
        mult = np.array([as_row(r["action"]) for r in recs])
        print(f"  decision {t_dec:>3} h: {sum(r['changed'] for r in recs)}/{n} changed, {len(plan)} branches "
              f"({time.time() - t0:.0f}s)", flush=True)

    new = pd.DataFrame(rows)
    out = pd.concat([data, new], ignore_index=True).sort_values(["seed", "t_dec", "kind"], kind="stable")
    out.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"added {len(new)} rows ({kind}); training data now {len(out)} rows ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
