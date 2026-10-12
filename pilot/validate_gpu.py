"""Check the GPU simulator (jaxsim.py) against the CPU results of experiment v2.

1. Recipe batches, seeds 1..24: final yield, max volume, sugar and oil fed
   vs runs/v2/results/batches.csv (CPU, scipy RK45).
2. Training-data generation, seeds 1000..1009: every row (state features and
   24 h outcomes of each branch) vs runs/v2/data/train.csv (CPU).
Both use the v2 setting (no level controller).
"""
import argparse
import os
import time

import jax
import numpy as np
import pandas as pd

from dss_common import ACTION_MVS
from gen_data_gpu import generate
from jaxsim import BatchSet
from train_dss import STATE_COLS

TOL_REL = 1e-3      # "match" if relative differences stay below 0.1 %


def rel_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", required=True, help="folder with the CPU v2 results (…/pilot_v2/runs/v2)")
    ap.add_argument("--gen-seeds", type=int, default=10)
    args = ap.parse_args()
    print("JAX devices:", jax.devices(), flush=True)
    ok_all = True

    # 1. recipe batches
    cpu = pd.read_csv(os.path.join(args.v2_dir, "results", "batches.csv")).sort_values("seed")
    t0 = time.time()
    bs = BatchSet(cpu.seed.tolist(), level_control=False)
    bs.run_until(230)
    use = bs.usage_since(0)
    print(f"\n[1] {len(bs)} recipe batches simulated together in {time.time() - t0:.1f}s (incl. compilation)")
    table = pd.DataFrame({"seed": cpu.seed.values, "yield_cpu": cpu.recipe_yield.values, "yield_gpu": bs.total_yield,
                          "vmax_cpu": cpu.recipe_vmax.values, "vmax_gpu": use["vmax"]})
    table["yield_rel_diff_%"] = rel_diff(table.yield_gpu, table.yield_cpu) * 100
    print(table.round(3).to_string(index=False))
    checks = {"yield": rel_diff(bs.total_yield, cpu.recipe_yield), "vmax": rel_diff(use["vmax"], cpu.recipe_vmax),
              "sugar": rel_diff(use["sugar_L"], cpu.recipe_sugar_L), "oil": rel_diff(use["oil_L"], cpu.recipe_oil_L)}
    for name, d in checks.items():
        good = d.max() < TOL_REL
        ok_all &= good
        print(f"    {name:6s} max rel. diff {d.max() * 100:.4f}%  mean {d.mean() * 100:.4f}%  -> "
              f"{'KHỚP' if good else 'LỆCH'}")

    # 2. training-data rows
    seeds = list(range(1000, 1000 + args.gen_seeds))
    t0 = time.time()
    rows, _ = generate(seeds, level_control=False, verbose=False)
    print(f"\n[2] training data for {len(seeds)} seeds generated in {time.time() - t0:.1f}s")
    gpu = pd.DataFrame(rows)
    cpu_tr = pd.read_csv(os.path.join(args.v2_dir, "data", "train.csv"))
    cpu_tr = cpu_tr[cpu_tr.seed.isin(seeds)]
    key = ["seed", "t_dec", "kind", "a_key"]
    for d in (gpu, cpu_tr):
        d["a_key"] = d[[f"a_{mv}" for mv in ACTION_MVS]].round(9).astype(str).agg("|".join, axis=1)
    m = gpu.merge(cpu_tr, on=key, suffixes=("_gpu", "_cpu"))
    print(f"    rows: GPU {len(gpu)}, CPU {len(cpu_tr)}, matched on (seed, time, action) {len(m)}")
    ok_all &= len(m) == len(cpu_tr) == len(gpu)
    for col in ["dyield", "vmax_w", "sugar_w", "oil_w"] + STATE_COLS:
        a, b = m[f"{col}_gpu"].values, m[f"{col}_cpu"].values
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, rel_diff(a, b))
        good = np.nanmax(d) < TOL_REL
        ok_all &= bool(good)
        if col in ["dyield", "vmax_w", "sugar_w", "oil_w"] or not good:
            print(f"    {col:10s} max rel. diff {np.nanmax(d) * 100:.4f}%  -> {'KHỚP' if good else 'LỆCH'}")
    print("\nKẾT LUẬN:", "Bộ mô phỏng GPU khớp với kết quả CPU, có thể chạy thí nghiệm." if ok_all else
          "CÓ SAI LỆCH - gửi toàn bộ kết quả ô này cho Claude trước khi chạy thí nghiệm.")


if __name__ == "__main__":
    main()
