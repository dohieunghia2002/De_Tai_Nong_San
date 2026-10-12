"""Summarise pilot results into runs/<exp>/results/summary.json (numbers quoted in the report)."""
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

from dss_common import ACTION_MVS, DECISION_TIMES, V_LIMIT_L
from train_dss import RESULTS_DIR

METHODS = {"m1": "TreeSHAP (standard)", "m2": "Delta-TreeSHAP (naive contrast)",
           "m3": "Contrastive Shapley (proposed)"}
POLICIES = {"recipe": "Recipe", "fix_so12": "Fixed rule: sugar & oil x1.2",
            "fix_static": "Best fixed action (model-chosen)", "dss": "DSS (adaptive)"}
MIN_EFFECT_KG = 5.0       # ground-truth components smaller than this are treated as ~0
N_BOOT = 2000


def boot_ci(values, groups, stat=np.mean, n=N_BOOT, seed=0):
    """95% CI by resampling whole batches (seeds), since decisions within a batch correlate."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"v": values, "g": groups})
    ids = df.g.unique()
    by = {g: df.v[df.g == g].values for g in ids}
    stats = [stat(np.concatenate([by[g] for g in rng.choice(ids, len(ids))])) for _ in range(n)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def paired(b, a, seeds):
    """Compare policy b against policy a on the same batches."""
    d = (b - a).values
    return {"mean_diff_kg": float(d.mean()), "ci": boot_ci(d, seeds), "mean_diff_pct": float((d / a.values).mean() * 100),
            "n_better": int((d > 0).sum()), "wilcoxon_p": float(wilcoxon(b, a).pvalue) if np.any(d != 0) else 1.0}


def fidelity(dec, m):
    gt = dec[[f"gt_{mv}" for mv in ACTION_MVS]].values
    ex = dec[[f"{m}_{mv}" for mv in ACTION_MVS]].values
    top1 = (np.abs(gt).argmax(1) == np.abs(ex).argmax(1)).astype(float)
    rho = np.array([spearmanr(g, e).statistic if np.std(e) > 0 and np.std(g) > 0 else 0.0
                    for g, e in zip(gt, ex)])
    cos = np.einsum("ij,ij->i", gt, ex) / (np.linalg.norm(gt, axis=1) * np.linalg.norm(ex, axis=1) + 1e-12)
    mae = np.abs(gt - ex).mean(1)
    mask = np.abs(gt) >= MIN_EFFECT_KG
    sign = np.array([np.mean(np.sign(g[k]) == np.sign(e[k])) if k.any() else np.nan
                     for g, e, k in zip(gt, ex, mask)])
    g = dec.seed.values
    ok = ~np.isnan(sign)
    return {
        "top1_agreement": float(top1.mean()), "top1_ci": boot_ci(top1, g),
        "spearman_mean": float(rho.mean()), "spearman_ci": boot_ci(rho, g),
        "cosine_mean": float(cos.mean()), "cosine_ci": boot_ci(cos, g),
        "mae_per_component_kg": float(mae.mean()), "mae_ci": boot_ci(mae, g),
        "mean_sum_kg": float(ex.sum(1).mean()),
        "sign_agreement": float(np.nanmean(sign)) if ok.any() else None,
        "top_driver_counts": {mv: int(c) for mv, c in zip(ACTION_MVS, np.bincount(np.abs(ex).argmax(1), minlength=4))},
        "mean_attribution_kg": {mv: float(v) for mv, v in zip(ACTION_MVS, ex.mean(0))},
    }


def main():
    bat = pd.read_csv(f"{RESULTS_DIR}/batches.csv")
    dec = pd.read_csv(f"{RESULTS_DIR}/decisions.csv")
    cv = json.load(open(f"{RESULTS_DIR}/surrogate_cv.json"))
    seeds = bat.seed.values

    policies = {}
    for p in POLICIES:
        policies[p] = {"mean_yield_kg": float(bat[f"{p}_yield"].mean()), "sd_yield_kg": float(bat[f"{p}_yield"].std()),
                       "batches_over_volume_limit": int((bat[f"{p}_vmax"] > V_LIMIT_L).sum()),
                       "mean_vmax_L": float(bat[f"{p}_vmax"].mean()),
                       "mean_sugar_L": float(bat[f"{p}_sugar_L"].mean()), "mean_oil_L": float(bat[f"{p}_oil_L"].mean()),
                       "yield_per_1000L_sugar": float((bat[f"{p}_yield"] / bat[f"{p}_sugar_L"] * 1000).mean())}
        if f"{p}_discharged_L" in bat:      # broth bled off (scheduled discharges + level controller), v3+
            policies[p]["mean_discharged_L"] = float(bat[f"{p}_discharged_L"].mean())
        if f"{p}_air" in bat:               # aeration used (Fg x h, simulator units), v4+
            policies[p]["mean_air"] = float(bat[f"{p}_air"].mean())
            policies[p]["air_vs_recipe_pct"] = float((bat[f"{p}_air"] / bat.recipe_air - 1).mean() * 100)
            policies[p]["sugar_vs_recipe_pct"] = float((bat[f"{p}_sugar_L"] / bat.recipe_sugar_L - 1).mean() * 100)
        if p != "recipe":
            policies[p]["vs_recipe"] = paired(bat[f"{p}_yield"], bat.recipe_yield, seeds)
    best_fixed = max(["fix_so12", "fix_static"], key=lambda p: bat[f"{p}_yield"].mean())
    comparisons = {"dss_vs_best_fixed": {"best_fixed": best_fixed,
                                         **paired(bat.dss_yield, bat[f"{best_fixed}_yield"], seeds)}}

    ch = dec[dec.changed].copy()
    recs = {"n_decisions": int(len(dec)), "n_changed": int(len(ch)),
            "distinct_actions": int(ch[[f"a_{mv}" for mv in ACTION_MVS]].drop_duplicates().shape[0])}
    if len(ch) >= 2:
        recs.update({
            "true_gain_positive_rate": float((ch.true_gain > 0).mean()),
            "pred_vs_true_gain_pearson": float(pearsonr(ch.pred_gain, ch.true_gain).statistic),
            "pred_vs_true_gain_mae_kg": float(np.mean(np.abs(ch.pred_gain - ch.true_gain))),
            "mean_pred_gain_kg": float(ch.pred_gain.mean()), "mean_true_gain_kg": float(ch.true_gain.mean()),
            "windows_over_volume_limit": int((ch.true_vmax > V_LIMIT_L).sum()),
            "by_decision_time": {int(t): {"pred_gain": float(g.pred_gain.mean()), "true_gain": float(g.true_gain.mean())}
                                 for t, g in ch.groupby("t_dec")},
            "most_common_actions": [
                {**{mv: float(a) for mv, a in zip(ACTION_MVS, idx)}, "count": int(c)}
                for idx, c in ch.groupby([f"a_{mv}" for mv in ACTION_MVS]).size()
                                .sort_values(ascending=False).head(5).items()],
        })
        gt_cols = [f"gt_{mv}" for mv in ACTION_MVS]
        recs["ground_truth"] = {
            "mean_attribution_kg": {mv: float(ch[f"gt_{mv}"].mean()) for mv in ACTION_MVS},
            "top_driver_counts": {mv: int(c) for mv, c in
                                  zip(ACTION_MVS, np.bincount(np.abs(ch[gt_cols].values).argmax(1), minlength=4))},
        }
    expl = {m: fidelity(ch, m) for m in METHODS} if len(ch) >= 2 else {}
    if len(ch) >= 2:
        # top-driver agreement split by the TRUE top driver (where v3 showed the aeration blind spot)
        gt = np.abs(ch[[f"gt_{mv}" for mv in ACTION_MVS]].values).argmax(1)
        for m in METHODS:
            ex = np.abs(ch[[f"{m}_{mv}" for mv in ACTION_MVS]].values).argmax(1)
            expl[m]["top1_by_true_driver"] = {ACTION_MVS[d]: {"n": int((gt == d).sum()),
                                                               "top1_agreement": float((ex[gt == d] == d).mean())}
                                              for d in range(len(ACTION_MVS)) if (gt == d).sum() >= 5}

    summary = {"surrogate_cv": cv, "policies": policies, "comparisons": comparisons,
               "recommendations": recs, "explanation_fidelity": expl,
               "method_names": METHODS, "policy_names": POLICIES,
               "volume_limit_L": V_LIMIT_L, "min_effect_kg_for_sign": MIN_EFFECT_KG,
               "decision_times_h": DECISION_TIMES, "n_test_batches": int(len(bat))}
    json.dump(summary, open(f"{RESULTS_DIR}/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
