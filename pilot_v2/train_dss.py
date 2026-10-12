"""Train the DSS surrogates from (observable state, action multipliers):
  - yield model: penicillin gained in the next 24 h (kg)
  - volume model: maximum broth volume reached in the next 24 h (L)
Validation is grouped by batch (seed), so no batch is in both training and validation.

Also picks the best STATIC action (one fixed multiplier set used from 96 h to the
end of every batch) with the same yield model, as a fair non-adaptive baseline.
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from dss_common import ACTION_MVS, CANDIDATES, OUT_DIR, V_LIMIT_L

STATE_COLS = ["t_h", "V", "Wt", "T", "pH", "DO2", "O2_offgas", "CO2_offgas", "OUR", "CER",
              "OUR_12h", "CER_12h", "DO2_12h", "P_off", "X_off", "PAA_off", "NH3_off", "Visc_off"]
ACTION_COLS = [f"a_{mv}" for mv in ACTION_MVS]
FEATURES = STATE_COLS + ACTION_COLS
PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)
MODELS_DIR = f"{OUT_DIR}/models"
RESULTS_DIR = f"{OUT_DIR}/results"
STATIC_MIN_FEASIBLE = 0.95   # static action must keep predicted volume under the limit in 95% of states


def contrast(df, col):
    """Value minus the recipe-action value at the same (seed, decision time)."""
    ref = df[df.kind == "recipe"].set_index(["seed", "t_dec"])[col]
    return df[col].values - ref.loc[list(zip(df.seed, df.t_dec))].values


def grouped_oof(df, target):
    oof = np.zeros(len(df))
    for tr, va in GroupKFold(n_splits=5).split(df, groups=df.seed):
        m = lgb.LGBMRegressor(**PARAMS).fit(df.iloc[tr][FEATURES], df.iloc[tr][target])
        oof[va] = m.predict(df.iloc[va][FEATURES])
    return oof


def best_static_action(yield_model, vol_model, states):
    """Candidate with the highest mean predicted 24 h gain over all training states,
    among those predicted to stay under the volume limit in most states."""
    best, best_gain, table = None, -np.inf, []
    for a in CANDIDATES:
        X = states.copy()
        for mv in ACTION_MVS:
            X[f"a_{mv}"] = a[mv]
        gain = float(yield_model.predict(X[FEATURES]).mean())
        feasible = float((vol_model.predict(X[FEATURES]) <= V_LIMIT_L).mean())
        table.append({**a, "mean_pred_gain": gain, "feasible_share": feasible})
        if feasible >= STATIC_MIN_FEASIBLE and gain > best_gain:
            best, best_gain = a, gain
    table = pd.DataFrame(table)
    if best is None:   # nothing feasible often enough: take the most feasible, then the best gain
        top = table.sort_values(["feasible_share", "mean_pred_gain"], ascending=False).iloc[0]
        best, best_gain = {mv: float(top[mv]) for mv in ACTION_MVS}, float(top.mean_pred_gain)
    return best, best_gain, table


def main():
    df = pd.read_csv(f"{OUT_DIR}/data/train.csv")
    print(f"{len(df)} rows, {df.seed.nunique()} batches")
    df["pred"] = grouped_oof(df, "dyield")
    df["pred_vmax"] = grouped_oof(df, "vmax_w")
    nonrecipe = df.kind != "recipe"
    d_true, d_pred = contrast(df, "dyield")[nonrecipe], contrast(df, "pred")[nonrecipe]
    metrics = {
        "n_rows": int(len(df)), "n_batches": int(df.seed.nunique()),
        "cv_r2_dyield": r2_score(df.dyield, df.pred), "cv_mae_dyield_kg": mean_absolute_error(df.dyield, df.pred),
        "cv_r2_contrast_vs_recipe": r2_score(d_true, d_pred),
        "cv_mae_contrast_kg": mean_absolute_error(d_true, d_pred),
        "cv_sign_agreement_contrast": float(np.mean(np.sign(d_true) == np.sign(d_pred))),
        "cv_r2_vmax": r2_score(df.vmax_w, df.pred_vmax),
        "cv_mae_vmax_L": mean_absolute_error(df.vmax_w, df.pred_vmax),
        "share_windows_over_volume_limit": float((df.vmax_w > V_LIMIT_L).mean()),
    }
    yield_model = lgb.LGBMRegressor(**PARAMS).fit(df[FEATURES], df["dyield"])
    vol_model = lgb.LGBMRegressor(**PARAMS).fit(df[FEATURES], df["vmax_w"])

    states = df[df.kind == "recipe"][STATE_COLS].reset_index(drop=True)
    static, static_gain, table = best_static_action(yield_model, vol_model, states)
    metrics["static_action"] = static
    metrics["static_action_mean_pred_gain_kg"] = static_gain
    for k, v in metrics.items():
        print(f"{k:34s} {v:.3f}" if isinstance(v, float) else f"{k:34s} {v}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    yield_model.booster_.save_model(f"{MODELS_DIR}/surrogate_yield.txt")
    vol_model.booster_.save_model(f"{MODELS_DIR}/surrogate_vmax.txt")
    json.dump(static, open(f"{MODELS_DIR}/static_action.json", "w"), indent=2)
    json.dump(metrics, open(f"{RESULTS_DIR}/surrogate_cv.json", "w"), indent=2)
    table.sort_values("mean_pred_gain", ascending=False).to_csv(f"{RESULTS_DIR}/static_candidates.csv", index=False)
    df.to_csv(f"{RESULTS_DIR}/surrogate_oof.csv", index=False)


if __name__ == "__main__":
    main()
