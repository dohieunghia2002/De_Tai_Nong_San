"""Spraying decision layer on top of the segmentation output (numpy only, runs anywhere).

Each 600x600 tile is split into square cells of `cell` pixels (200 px = 0.5 m at the 2.5 mm GSD of WeedsGalore).
  truth  w     : labelled weed fraction of the cell; the cell needs spraying if w >= tau
  score  s     : mean predicted P(weed) over the cell, i.e. the expected weed fraction
Decisions per cell: NO_SPRAY, SPRAY or DEFER (scout the cell; scouting is assumed to reveal the true need).

Class-conditional split conformal thresholds, calibrated on held-out cells that played no part in training or model
selection (Vovk et al.; as used per class by Melki et al. 2025 for weeding), give for an exchangeable new cell
  P(s < lam_lo | need)   <= alpha   -> NO_SPRAY only when s < lam_lo   (missed spraying is bounded)
  P(s > lam_hi | no need) <= beta   -> SPRAY only when s > lam_hi      (unneeded spraying is bounded)
and cells in between are deferred. Cells of one tile are spatially correlated, so exchangeability, and with it the
guarantee, holds only approximately; the empirical rates are reported next to alpha and beta.
"""
import math

import numpy as np
import pandas as pd

NO_SPRAY, SPRAY, DEFER = 0, 1, 2
_EPS = 1e-9  # guards floor/ceil against floating-point error in alpha * (n + 1)


def cell_view(a, cell):
    n, h, w = a.shape
    if h % cell or w % cell:
        raise ValueError(f'tile size {h}x{w} is not a multiple of the cell size {cell}')
    return a.reshape(n, h // cell, cell, w // cell, cell)


def cell_table(ids, y3, p_weed, cell):
    """One row per cell: tile id, acquisition date, true weed and crop fractions, and the score s."""
    ids = np.asarray(ids)
    if len(ids) != len(y3) or y3.shape != p_weed.shape:
        raise ValueError(f'misaligned inputs: ids {len(ids)}, labels {y3.shape}, probabilities {p_weed.shape}')
    w = cell_view(y3 == 2, cell).mean(axis=(2, 4))
    crop = cell_view(y3 == 1, cell).mean(axis=(2, 4))
    # float64: a float32 mean turns an exact 1 % cover (400 / 40000) into 0.0099999998 and flips a decision at tau
    s = cell_view(p_weed.astype(np.float64), cell).mean(axis=(2, 4))
    tile = np.repeat(np.arange(len(ids)), w.shape[1] * w.shape[2])
    return pd.DataFrame({'tile': ids[tile], 'date': [t[:10] for t in ids[tile]], 'cell_size': cell,
                         'w': w.ravel(), 'crop': crop.ravel(), 's': s.ravel()})


def lower_threshold(pos_scores, alpha):
    """floor(alpha (n+1))-th smallest score of calibration cells that need spraying; -inf when there are too few,
    so that 'need' is never ruled out."""
    s = np.sort(np.asarray(pos_scores, dtype=float))
    k = math.floor(alpha * (len(s) + 1) + _EPS)
    return float(s[k - 1]) if k >= 1 else -np.inf


def upper_threshold(neg_scores, beta):
    """ceil((1-beta)(n+1))-th smallest score of calibration cells that do not need spraying; +inf when there are
    too few, so that 'no need' is never ruled out."""
    s = np.sort(np.asarray(neg_scores, dtype=float))
    k = math.ceil((1 - beta) * (len(s) + 1) - _EPS)
    return float(s[k - 1]) if k <= len(s) else np.inf


def decide_conformal(s, lam_lo, lam_hi):
    s = np.asarray(s, dtype=float)
    need_possible = s >= lam_lo
    no_need_possible = s <= lam_hi
    d = np.full(s.shape, DEFER)
    d[need_possible & ~no_need_possible] = SPRAY
    d[~need_possible & no_need_possible] = NO_SPRAY
    return d  # both possible, or neither (lam_hi < s < lam_lo): DEFER


def decide_threshold(s, t):
    return np.where(np.asarray(s, dtype=float) >= t, SPRAY, NO_SPRAY)


def decision_metrics(d, need, w, crop):
    d, need, w, crop = map(np.asarray, (d, need, w, crop))
    spray, defer, no_spray = d == SPRAY, d == DEFER, d == NO_SPRAY
    effective = spray | (defer & need)  # deferred cells are scouted and sprayed only if needed

    def share(mask, among=None):
        if among is None:
            return float(mask.mean()) if mask.size else math.nan
        return float(mask[among].mean()) if among.any() else math.nan

    return {
        'n_cells': int(d.size),
        'infestation': share(need),
        'sprayed': share(spray),
        'deferred': share(defer),
        'effective_sprayed': share(effective),
        'spray_reduction': 1 - share(effective),
        'missed_need_rate': share(no_spray, need),       # P(NO_SPRAY | needs spraying)
        'false_spray_rate': share(spray, ~need),         # P(SPRAY | does not need spraying)
        'missed_weed_area': float(w[no_spray].sum() / w.sum()) if w.sum() > 0 else math.nan,
        'crop_area_sprayed': float(crop[spray].sum() / crop.sum()) if crop.sum() > 0 else math.nan,
        'spray_surplus': share(spray) - share(need),     # Melki et al. 2025, eq. 13 (before scouting)
    }


def policy_decisions(cal, test, tau, alpha, beta):
    """Thresholds from `cal` cells, decisions of every policy on `test` cells (DataFrames from cell_table)."""
    need_cal = cal['w'].to_numpy() >= tau
    s_cal = cal['s'].to_numpy()
    lam_lo = lower_threshold(s_cal[need_cal], alpha)
    lam_hi = upper_threshold(s_cal[~need_cal], beta)
    s = test['s'].to_numpy()
    need = test['w'].to_numpy() >= tau
    policies = {
        'broadcast': np.full(s.shape, SPRAY),
        'oracle': np.where(need, SPRAY, NO_SPRAY),
        'plug_in': decide_threshold(s, tau),                 # spray if the predicted weed fraction reaches tau
        'conformal_miss_only': decide_threshold(s, lam_lo),  # spray unless 'need' is ruled out (no deferral)
        'conformal_3way': decide_conformal(s, lam_lo, lam_hi),
    }
    info = {'lam_lo': lam_lo, 'lam_hi': lam_hi, 'n_cal_need': int(need_cal.sum()),
            'n_cal_no_need': int((~need_cal).sum())}
    return policies, info


def evaluate_policies(cal, test, tau, alpha, beta):
    """Calibrate on `cal` cells and evaluate every policy on `test` cells."""
    policies, info = policy_decisions(cal, test, tau, alpha, beta)
    need = test['w'].to_numpy() >= tau
    w, crop = test['w'].to_numpy(), test['crop'].to_numpy()
    return [{'policy': k, 'tau': tau, 'alpha': alpha, 'beta': beta, **info, **decision_metrics(d, need, w, crop)}
            for k, d in policies.items()]


def evaluate_crossfit(cells, fold, tau, alpha, beta):
    """Each fold is decided with thresholds calibrated on the other folds, then all cells are scored together, so
    calibration cells never come from the evaluated locations. `fold` gives one fold label per row of `cells`.
    lam_lo / lam_hi are tuples of the thresholds applied to each fold, in sorted fold order."""
    fold = np.asarray(fold)
    if len(np.unique(fold)) < 2:
        return []  # nothing left to calibrate on (e.g. a date whose tiles all fall in one fold)
    parts = []
    for k in np.unique(fold):
        ev = cells[fold == k]
        policies, info = policy_decisions(cells[fold != k], ev, tau, alpha, beta)
        parts.append((ev, policies, info))
    ev_all = pd.concat([p[0] for p in parts])
    need = ev_all['w'].to_numpy() >= tau
    w, crop = ev_all['w'].to_numpy(), ev_all['crop'].to_numpy()
    info = {'lam_lo': tuple(p[2]['lam_lo'] for p in parts), 'lam_hi': tuple(p[2]['lam_hi'] for p in parts),
            'n_cal_need': min(p[2]['n_cal_need'] for p in parts),
            'n_cal_no_need': min(p[2]['n_cal_no_need'] for p in parts)}
    rows = []
    for name in parts[0][1]:
        d = np.concatenate([p[1][name] for p in parts])
        rows.append({'policy': name, 'tau': tau, 'alpha': alpha, 'beta': beta, **info,
                     **decision_metrics(d, need, w, crop)})
    return rows
