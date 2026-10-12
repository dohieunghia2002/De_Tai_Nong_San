"""GPU version of gen_data.py: same procedure, same random actions, all seeds at once.

For every training seed: recipe until the first decision point; at each decision
point branch into the recipe action + main action + K random actions (24 h each);
the main trajectory continues with its main action.
Output: runs/<exp>/data/train.csv (same columns as the CPU version).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from dss_common import (ACTION_MVS, DECISION_TIMES, LEVEL_CONTROL, OUT_DIR, RECIPE_ACTION, TRAIN_RANGE,
                        state_features_batch, window_end)
from jaxsim import BatchSet

K_RANDOM = 4


def random_action(rng):
    return {mv: float(rng.uniform(*TRAIN_RANGE)) for mv in ACTION_MVS}


def as_matrix(actions):
    return np.array([[a[mv] for mv in ACTION_MVS] for a in actions])


def generate(seeds, level_control, verbose=True):
    """Rows identical in layout (and, without level control, in values) to gen_data.one_seed."""
    seeds = list(seeds)
    rngs = [np.random.default_rng(10_000 + s) for s in seeds]
    main = BatchSet(seeds, level_control)
    main_mult = None
    rows = []
    t0 = time.time()
    for t_dec in DECISION_TIMES:
        main.run_until(t_dec, main_mult)
        feats = state_features_batch(main).to_dict("records")
        plan = []                                   # (batch index, kind, action), same draw order as CPU
        for i, rng in enumerate(rngs):
            main_a = random_action(rng)
            acts = [("recipe", RECIPE_ACTION), ("main", main_a)] + \
                   [("random", random_action(rng)) for _ in range(K_RANDOM)]
            plan += [(i, kind, a) for kind, a in acts]
        branches = main.fork([i for i, _, _ in plan])
        y0, k0 = branches.total_yield, branches.k
        branches.run_until(window_end(t_dec), as_matrix([a for _, _, a in plan]))
        dy = branches.total_yield - y0
        use = branches.usage_since(k0)
        for j, (i, kind, a) in enumerate(plan):
            rows.append({"seed": seeds[i], "t_dec": t_dec, "kind": kind, **feats[i],
                         **{f"a_{mv}": a[mv] for mv in ACTION_MVS},
                         "dyield": dy[j], "vmax_w": use["vmax"][j], "sugar_w": use["sugar_L"][j],
                         "oil_w": use["oil_L"][j], "discharged_w": use["discharged_L"][j]})
        main_mult = as_matrix([a for _, kind, a in plan if kind == "main"])
        if verbose:
            print(f"  decision {t_dec:>3} h: {len(plan)} branches simulated, {time.time() - t0:.0f}s elapsed",
                  flush=True)
    main.run_until(230, main_mult)
    return rows, main.total_yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=120)
    ap.add_argument("--first-seed", type=int, default=1000)
    args = ap.parse_args()
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    print(f"[{OUT_DIR}] {len(seeds)} training batches, level control = {LEVEL_CONTROL}", flush=True)
    t0 = time.time()
    rows, _ = generate(seeds, LEVEL_CONTROL)
    os.makedirs(f"{OUT_DIR}/data", exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["seed", "t_dec", "kind"], kind="stable")
    df.to_csv(f"{OUT_DIR}/data/train.csv", index=False)
    print(f"saved {len(df)} rows to {OUT_DIR}/data/train.csv in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
