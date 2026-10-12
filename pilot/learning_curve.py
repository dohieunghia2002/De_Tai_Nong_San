"""Learning curve for the yield surrogate: does more random data help, or does it
need data in the right place?

Configurations: random-action data from the first 60 / 120 / 180 / 240 training
batches, and 240 batches + the targeted rows from augment_gpu.py. Each is scored by
5-fold cross-validation grouped by batch, on held-out batches only, separately on
  - random rows: contrast vs the recipe in random-walk states,
  - DSS-neighbourhood rows (aug*): contrast vs the recipe around the DSS's own
    recommendations - the quantity the explanations depend on.
Output: runs/<exp>/results/learning_curve.csv
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from dss_common import OUT_DIR
from train_dss import BASE_KINDS, FEATURES, PARAMS, RESULTS_DIR, add_state_id, contrast, is_reference


def score(df, train_mask, seeds_used):
    """Grouped CV over `seeds_used`; returns contrast metrics on held-out batches."""
    pred = np.full(len(df), np.nan)
    seeds_used = np.array(sorted(seeds_used))
    for tr_s, va_s in GroupKFold(n_splits=5).split(seeds_used, groups=seeds_used):
        tr = train_mask & df.seed.isin(seeds_used[tr_s]).values
        va = df.seed.isin(seeds_used[va_s]).values
        m = lgb.LGBMRegressor(**PARAMS).fit(df.loc[tr, FEATURES], df.loc[tr, "dyield"])
        pred[va] = m.predict(df.loc[va, FEATURES])
    held = df.seed.isin(seeds_used).values
    df = df.assign(pred=pred)
    c_true, c_pred = contrast(df, "dyield"), contrast(df, "pred")
    out = {}
    nonref = ~is_reference(df)
    base = df.kind.isin(BASE_KINDS).values
    for name, mask in [("random_rows", held & nonref & base), ("dss_rows", held & nonref & ~base)]:
        ok = mask & ~np.isnan(c_pred)
        if ok.sum() >= 2:
            out[f"{name}_r2_contrast"] = r2_score(c_true[ok], c_pred[ok])
            out[f"{name}_sign_agreement"] = float(np.mean(np.sign(c_true[ok]) == np.sign(c_pred[ok])))
            out[f"{name}_n"] = int(ok.sum())
    return out


def main():
    df = add_state_id(pd.read_csv(f"{OUT_DIR}/data/train.csv")).reset_index(drop=True)
    all_seeds = sorted(df.seed.unique())
    base = df.kind.isin(BASE_KINDS).values
    has_aug = (~base).any()
    rows = []
    for n_seeds in [60, 120, 180, 240]:
        if n_seeds > len(all_seeds):
            continue
        used = all_seeds[:n_seeds]
        rows.append({"training_data": f"random actions, {n_seeds} batches", "n_batches": n_seeds,
                     **score(df, base, used)})
        print(rows[-1], flush=True)
    if has_aug:
        used = all_seeds[:240]
        rows.append({"training_data": "random actions, 240 batches + DSS-neighbourhood rows", "n_batches": 240,
                     **score(df, np.ones(len(df), bool), used)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/learning_curve.csv", index=False)
    print(f"saved {RESULTS_DIR}/learning_curve.csv")


if __name__ == "__main__":
    main()
