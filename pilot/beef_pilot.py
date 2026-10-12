"""Companion pilot on real agri-food data: beef e-nose (Wijaya, Sarno & Zulaika,
Data in Brief 2018; Mendeley Data 10.17632/mwmhh766fc.3, CC BY 4.0).

Decision task: at every minute, decide RELEASE (excellent/good/acceptable) or
REJECT (spoiled) from the last 30 minutes of sensor readings (causal features only).

Checks:
  1. Leakage: random minute-level split vs leave-one-series-out (LOSO).
  2. XAI audit: does the model rely on gas sensors or on ambient temperature/humidity
     (which drift with time, the same way the time-indexed labels do)?
  3. Real-time explanation stability: rank agreement of SHAP vectors between
     consecutive minutes.
Output: results/beef_summary.json
"""
import json
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
GAS = ["MQ135", "MQ136", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ8", "MQ9"]
AMBIENT = ["Humidity", "Temperature"]
WIN = 30  # minutes
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, n_jobs=2, verbose=-1)


def load():
    frames = []
    for i in range(1, 6):
        d = pd.read_csv(f"data/beef/TS{i}.csv")
        d["series"] = i
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def window_features(d, sensors):
    out = []
    for _, g in d.groupby("series"):
        g = g.sort_values("minute")
        f = pd.DataFrame(index=g.index)
        for s in sensors:
            r = g[s].rolling(WIN, min_periods=WIN)
            f[f"{s}_mean"] = r.mean()
            f[f"{s}_slope"] = g[s].diff(WIN - 1) / (WIN - 1)
        out.append(f)
    return pd.concat(out)


def loso(X, y, groups):
    pred = np.zeros(len(y), dtype=int)
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        m = lgb.LGBMClassifier(**PARAMS).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def metrics(y, p):
    spoiled = y == 1
    return {"accuracy": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro")),
            "false_release_rate": float(np.mean(p[spoiled] == 0))}   # spoiled meat released


def shap_audit(X, y, groups):
    """Share of mean |SHAP| carried by gas vs ambient features, and temporal stability."""
    share, stab = [], []
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        m = lgb.LGBMClassifier(**PARAMS).fit(X[tr], y[tr])
        sv = shap.TreeExplainer(m).shap_values(X[te])
        sv = sv[1] if isinstance(sv, list) else sv
        imp = np.abs(sv).mean(0)
        amb = [i for i, c in enumerate(X.columns) if c.split("_")[0] in AMBIENT]
        share.append(imp[amb].sum() / imp.sum())
        rhos = [spearmanr(sv[i], sv[i + 1]).statistic for i in range(len(sv) - 1)]
        stab.append(float(np.nanmean(rhos)))
    return float(np.mean(share)), float(np.mean(stab))


def main():
    d = load()
    results = {}
    for name, sensors in [("gas+ambient", GAS + AMBIENT), ("gas_only", GAS)]:
        F = window_features(d, sensors)
        ok = F.notna().all(axis=1)
        X, y, g = F[ok].reset_index(drop=True), (d["class"][ok] == "spoiled").astype(int).values, d["series"][ok].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        leaky = metrics(yte, lgb.LGBMClassifier(**PARAMS).fit(Xtr, ytr).predict(Xte))
        honest = metrics(y, loso(X, y, g))
        amb_share, stability = shap_audit(X, y, g)
        results[name] = {"random_split_leaky": leaky, "leave_one_series_out": honest,
                         "shap_share_ambient": amb_share, "shap_temporal_stability_spearman": stability,
                         "n_samples": int(len(y))}
    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/beef_summary.json", "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
