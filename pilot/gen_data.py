"""Generate training data for the DSS surrogate models.

For each training seed: run the recipe until the first decision point. At every
decision point, branch the batch and simulate the next 24 h under the recipe
action plus K random actions (same state, different actions, so the models can
learn action effects). The main trajectory then continues with one random action
so that later decision points see diverse states.

Each row records the penicillin gained in the window (dyield), the maximum broth
volume reached (vmax_w) and the sugar/oil fed (sugar_w, oil_w).
Output: runs/<exp>/data/train.csv, one row per (seed, decision time, action).
"""
import argparse
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from dss_common import (ACTION_MVS, DECISION_TIMES, OUT_DIR, RECIPE_ACTION, TRAIN_RANGE, action_features,
                        action_policy, state_features, usage_since, window_end)
from sim import Batch

K_RANDOM = 4
DATA_DIR = f"{OUT_DIR}/data"
PARTS_DIR = f"{DATA_DIR}/parts"


def part_path(seed):
    return f"{PARTS_DIR}/train_seed{seed}.csv"


def random_action(rng):
    return {mv: float(rng.uniform(*TRAIN_RANGE)) for mv in ACTION_MVS}


def simulate_window(batch, action, t_dec):
    """Outcome of the window [t_dec, window end] under `action`, from a fork of `batch`."""
    b = batch.fork()
    y0, k0 = b.total_yield, b.k
    t1 = window_end(t_dec)
    b.run_until(t1, action_policy(action, t_dec, t1))
    u = usage_since(b, k0)
    return {"dyield": b.total_yield - y0, "vmax_w": u["vmax"], "sugar_w": u["sugar_L"], "oil_w": u["oil_L"]}


def one_seed(seed):
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(10_000 + seed)
    batch = Batch(seed)
    rows = []
    for t_dec in DECISION_TIMES:
        batch.run_until(t_dec, getattr(batch, "_policy", None))
        s = state_features(batch)
        main_action = random_action(rng)
        actions = [("recipe", RECIPE_ACTION), ("main", main_action)] + \
                  [("random", random_action(rng)) for _ in range(K_RANDOM)]
        for kind, a in actions:
            rows.append({"seed": seed, "t_dec": t_dec, "kind": kind, **s, **action_features(a),
                         **simulate_window(batch, a, t_dec)})
        batch._policy = action_policy(main_action, t_dec, window_end(t_dec))
    batch.run_until(230, batch._policy)
    return rows, batch.total_yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=120)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    # One file per finished seed, so an interrupted run resumes where it stopped.
    os.makedirs(PARTS_DIR, exist_ok=True)
    todo = [s for s in seeds if not os.path.exists(part_path(s))]
    print(f"[{OUT_DIR}] {len(seeds) - len(todo)} seeds already done, {len(todo)} to run "
          f"on {args.workers} workers", flush=True)
    t0 = time.time()
    if todo:
        with Pool(args.workers) as pool:
            for i, (r, final_yield) in enumerate(pool.imap_unordered(one_seed, todo), 1):
                seed = r[0]["seed"]
                pd.DataFrame(r).to_csv(part_path(seed) + ".tmp", index=False)
                os.replace(part_path(seed) + ".tmp", part_path(seed))
                print(f"[{i}/{len(todo)}] seed {seed} done, final yield (random walk) "
                      f"{final_yield:.0f} kg, {time.time() - t0:.0f}s elapsed", flush=True)
    df = pd.concat([pd.read_csv(part_path(s)) for s in seeds]).sort_values(["seed", "t_dec", "kind"])
    df.to_csv(f"{DATA_DIR}/train.csv", index=False)
    print(f"saved {len(df)} rows ({len(seeds)} seeds) to {DATA_DIR}/train.csv")


if __name__ == "__main__":
    main()
