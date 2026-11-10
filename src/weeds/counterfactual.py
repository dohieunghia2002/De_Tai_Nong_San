"""Decision-level counterfactual explanations of the spraying decisions, computed from the P(weed) map alone.

A cell's decision is a threshold rule on its score s = mean P(weed) over the cell (src.weeds.decision), so it can be
explained by what would have to change for the decision to change (counterfactual explanations, Wachter et al. 2018,
Harvard Journal of Law & Technology):
  SPRAY      the fewest patches of the cell's fixed 20 x 20 px grid (5 cm x 5 cm) whose removal ends SPRAY, and gives
             NO_SPRAY;
  DEFER      the predicted weed area to add for SPRAY, and the patches / area to remove for NO_SPRAY;
  NO_SPRAY   the predicted weed area to add for SPRAY.
Only patches inside the cell are removed (the model also sees context outside it). Removal takes the patches with the
most predicted weed first and assumes, without running the model again, that a removed patch becomes weed-free and
the rest of the cell is unchanged: s drops by the patch's P(weed) mass / cell^2. The decision is monotone in s, so
under that assumption largest-first gives the fewest patches. The deletion test of src.weeds.explain removes the same
patches (its 'predicted_weed' ranking) and rescores the image with the model, which measures how far the estimate is
off (compare_with_deletion), at its fixed removal sizes only. Adding weed follows from the rule alone and cannot be
checked with the model.
Areas are areas of predicted weed: P(weed) summed over pixels, times the pixel area.

  python -m src.weeds.counterfactual --probs <probs_test.npz> --decisions <decisions.csv> --out <dir>
                                     [--deletion <xai_deletion.csv>]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.weeds.analyze import PRIMARY
from src.weeds.decision import DEFER, NO_SPRAY, SPRAY, decide_conformal

GSD_CM = 0.25  # ground sampling distance of WeedsGalore: 2.5 mm per pixel (DJI P4 Multispectral at 5 m)
PATCH = 20  # 20 px = 5 cm, the patch of the deletion test in src.weeds.explain
SCORE_TOL = 1e-5  # the decisions table and the probability map may come from two GPU runs of the same model
SPRAYNESS = {NO_SPRAY: 0, DEFER: 1, SPRAY: 2}


def ranked_patches(p_cell, patch):
    """Patches of one cell's P(weed) map (cell, cell) ranked by their P(weed) mass, largest first: their indices in
    row-major order of the cell's patch grid, and their masses (float64)."""
    c = p_cell.shape[0]
    if p_cell.shape != (c, c) or c % patch:
        raise ValueError(f'the cell map must be square with a side divisible by the patch ({patch} px): {p_cell.shape}')
    k = c // patch
    m = p_cell.astype(np.float64).reshape(k, patch, k, patch).sum((1, 3)).ravel()
    order = np.argsort(-m, kind='stable')
    return order, m[order]


def removal_path(s, masses, cell):
    """Estimated score after removing the first 0, 1, ..., all ranked patches."""
    return s - np.concatenate([[0.0], np.cumsum(masses)]) / cell ** 2


def first_index(mask):
    i = np.flatnonzero(mask)
    return float(i[0]) if len(i) else np.nan


def _cell_map(p_weed, row, cell, cell_idx):
    cols = p_weed.shape[2] // cell
    r, c = divmod(int(cell_idx), cols)
    return r, c, p_weed[row, r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]


def _check_inputs(ids, p_weed, decisions, cell, patch):
    n, h, w = p_weed.shape
    if len(ids) != n:
        raise ValueError(f'{len(ids)} ids for {n} probability maps')
    if h % cell or w % cell or cell % patch:
        raise ValueError(f'the cell ({cell} px) must divide the tile ({h} x {w}), and the patch ({patch} px) the cell')
    missing = set(decisions['tile']) - set(ids)
    if missing:
        raise ValueError(f'no probability map for tiles {sorted(missing)}')


def counterfactual_table(ids, p_weed, decisions, cell, patch=PATCH, gsd_cm=GSD_CM, score_tol=SCORE_TOL):
    """One row per cell of the decision table (columns tile, date, s, lam_lo, lam_hi, decision, and cell, the cell's
    row-major index in its tile; when absent, the row order within each tile). p_weed: (n, H, W) maps of the tiles ids.
    Areas in cm^2 of predicted weed; k_* in patches (NaN when no removal reaches the decision); `patches` lists the
    ranked patches (row-major indices in the tile's patch grid) up to the larger finite k, so that the first k_not_spray
    of them end SPRAY. Increases are infima: where the rule's inequality is strict, an arbitrarily small extra is
    needed."""
    ids = list(ids)
    _check_inputs(ids, p_weed, decisions, cell, patch)
    t = decisions.copy()
    if 'cell' not in t:
        t['cell'] = t.groupby('tile').cumcount()
    row_of = {tile: i for i, tile in enumerate(ids)}
    cell_cm2 = cell ** 2 * gsd_cm ** 2
    k, grid_cols = cell // patch, p_weed.shape[2] // patch
    rows = []
    for _, d in t.iterrows():
        r, c, p_cell = _cell_map(p_weed, row_of[d['tile']], cell, d['cell'])
        order, masses = ranked_patches(p_cell, patch)
        s = masses.sum() / cell ** 2
        if abs(s - d['s']) > score_tol:
            raise ValueError(f'{d["tile"]} cell {d["cell"]}: score {s:.6f} from the map, {d["s"]:.6f} in the table')
        lo, hi = float(d['lam_lo']), float(d['lam_hi'])
        path = removal_path(s, masses, cell)
        dec = decide_conformal(path, lo, hi)
        if dec[0] != d['decision']:
            raise ValueError(f'{d["tile"]} cell {d["cell"]}: decision {dec[0]} from the map, {d["decision"]} in the '
                             'table')
        k_not, k_no = first_index(dec != SPRAY), first_index(dec == NO_SPRAY)

        def removed_cm2(kk):
            return (s - path[int(kk)]) * cell_cm2 if np.isfinite(kk) else np.nan

        upto = int(np.nanmax([k_not, k_no, 0.0]))
        tile_patch = (r * k + order // k) * grid_cols + c * k + order % k
        rows.append({'tile': d['tile'], 'cell': int(d['cell']), 'date': d['date'], 'decision': int(d['decision']),
                     's': s, 'lam_lo': lo, 'lam_hi': hi, 'weed_cm2': s * cell_cm2,
                     'add_to_spray_cm2': max(0.0, max(lo, hi) - s) * cell_cm2,
                     'remove_to_no_spray_cm2': max(0.0, s - min(lo, hi)) * cell_cm2,
                     'k_not_spray': k_not, 'removed_cm2_not_spray': removed_cm2(k_not),
                     'k_no_spray': k_no, 'removed_cm2_no_spray': removed_cm2(k_no),
                     'patches': ' '.join(str(int(i)) for i in tile_patch[:upto])})
    return pd.DataFrame(rows)


def compare_with_deletion(table, ids, p_weed, deletion, cell, patch=PATCH, gsd_cm=GSD_CM):
    """The estimated score after removing the top-n patches next to the model's rescoring in the deletion test of
    src.weeds.explain, whose 'predicted_weed' ranking removes the same patches (n = ceil(fraction x patches per
    tile)). One row per tested cell and fraction."""
    ids = list(ids)
    row_of = {tile: i for i, tile in enumerate(ids)}
    grid = (p_weed.shape[1] // patch, p_weed.shape[2] // patch)
    cell_cm2 = cell ** 2 * gsd_cm ** 2
    t = table.set_index(['tile', 'cell'])
    paths, rows = {}, []
    for _, d in deletion[deletion['method'] == 'predicted_weed'].iterrows():
        key = (d['tile'], int(d['cell']))
        c = t.loc[key]
        if key not in paths:
            _, _, p_cell = _cell_map(p_weed, row_of[key[0]], cell, key[1])
            masses = ranked_patches(p_cell, patch)[1]
            paths[key] = removal_path(masses.sum() / cell ** 2, masses, cell)
        path = paths[key]
        n = int(math.ceil(d['fraction'] * grid[0] * grid[1]))  # the same float expression as explain.deletion_tile
        s_hat = path[min(n, len(path) - 1)]
        rows.append({'tile': key[0], 'cell': key[1], 'date': c['date'], 'decision': int(c['decision']),
                     'fraction': d['fraction'], 'n_patches': n, 's': c['s'], 's_after_model': d['s_after'],
                     's_after_estimate': s_hat, 'lam_lo': c['lam_lo'], 'lam_hi': c['lam_hi']})
    out = pd.DataFrame(rows)
    out['error_cm2'] = (out['s_after_estimate'] - out['s_after_model']) * cell_cm2  # < 0: estimate drops more
    out['decision_model'] = decide_conformal(out['s_after_model'], out['lam_lo'], out['lam_hi'])
    out['decision_estimate'] = decide_conformal(out['s_after_estimate'], out['lam_lo'], out['lam_hi'])
    return out


def _q(x):
    x = pd.Series(x, dtype=float)
    finite = x[np.isfinite(x)]
    res = {'n': len(x), 'not_reachable': int(x.isna().sum() + np.isinf(x).sum())}
    if len(finite):
        res.update({k: float(v) for k, v in zip(('min', 'p25', 'median', 'p75', 'max'),
                                                finite.quantile([0, .25, .5, .75, 1]))})
    return res


def summarize(table):
    sp, de, ns = (table[table['decision'] == v] for v in (SPRAY, DEFER, NO_SPRAY))
    out = {'cells': len(table), 'decisions': {'spray': len(sp), 'no_spray': len(ns), 'defer': len(de)},
           'spray': {'k_not_spray': _q(sp['k_not_spray']), 'removed_cm2_not_spray': _q(sp['removed_cm2_not_spray']),
                     'k_no_spray': _q(sp['k_no_spray']),
                     **{f'k_not_spray_le_{k}': int((sp['k_not_spray'] <= k).sum()) for k in (1, 2, 4)}},
           'defer': {'add_to_spray_cm2': _q(de['add_to_spray_cm2']),
                     'remove_to_no_spray_cm2': _q(de['remove_to_no_spray_cm2']), 'k_no_spray': _q(de['k_no_spray'])},
           'no_spray': {'add_to_spray_cm2': _q(ns['add_to_spray_cm2'])},
           'by_date': {}}
    for date, g in table.groupby('date'):
        gs, gn = g[g['decision'] == SPRAY], g[g['decision'] == NO_SPRAY]
        out['by_date'][date] = {'spray': len(gs), 'spray_k_not_spray_median': float(gs['k_not_spray'].median())
                                if len(gs) else None,
                                'no_spray': len(gn), 'no_spray_add_to_spray_cm2_median':
                                    float(gn['add_to_spray_cm2'].median()) if len(gn) else None}
    return out


def summarize_comparison(cmp):
    lenient = cmp['decision_estimate'].map(SPRAYNESS) < cmp['decision_model'].map(SPRAYNESS)
    was_spray = cmp['decision'] == SPRAY
    drop_ratio = (cmp['s'] - cmp['s_after_estimate']) / (cmp['s'] - cmp['s_after_model'])
    by_fraction = {}
    for f, g in cmp.groupby('fraction'):
        e = g['error_cm2'].abs()
        by_fraction[str(f)] = {'rows': len(g), 'abs_error_cm2_median': float(e.median()),
                               'abs_error_cm2_p95': float(e.quantile(.95)), 'abs_error_cm2_max': float(e.max()),
                               'drop_ratio_median': float(drop_ratio[g.index].median())}
    agree = cmp['decision_model'] == cmp['decision_estimate']
    # rows where neither changes the decision agree trivially; agreement where at least one of them changes it
    changed = (cmp['decision_model'] != cmp['decision']) | (cmp['decision_estimate'] != cmp['decision'])
    return {'rows': len(cmp), 'cells': int(cmp[['tile', 'cell']].drop_duplicates().shape[0]),
            'decision_agreement': float(agree.mean()), 'disagreements': int((~agree).sum()),
            'rows_decision_changed': int(changed.sum()),
            'decision_agreement_where_changed': float(agree[changed].mean()) if changed.any() else None,
            'disagreements_estimate_less_spray': int((~agree & lenient).sum()),
            'spray_end_agreement': float(((cmp['decision_model'] != SPRAY) == (cmp['decision_estimate'] != SPRAY))
                                         [was_spray].mean()),
            'spray_rows': int(was_spray.sum()), 'by_fraction': by_fraction}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probs', required=True, help='probs_test.npz of the model that made the decisions')
    ap.add_argument('--decisions', required=True, help='decisions.csv (src.weeds.explain) of the same model')
    ap.add_argument('--deletion', help='xai_deletion.csv of the same model, to check the estimate against the model')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cell', type=int, default=PRIMARY['cell_size'])
    ap.add_argument('--patch', type=int, default=PATCH)
    args = ap.parse_args()
    with np.load(args.probs, allow_pickle=False) as z:
        ids, p_weed = list(z['ids']), z['p_weed']
    decisions = pd.read_csv(args.decisions)
    t0 = time.perf_counter()
    table = counterfactual_table(ids, p_weed, decisions, args.cell, args.patch)
    seconds = time.perf_counter() - t0
    summary = {'inputs': {'probs': args.probs, 'decisions': args.decisions, 'deletion': args.deletion,
                          'cell_px': args.cell, 'patch_px': args.patch, 'gsd_cm': GSD_CM},
               **summarize(table)}
    if args.deletion:
        summary['check_against_model'] = summarize_comparison(
            compare_with_deletion(table, ids, p_weed, pd.read_csv(args.deletion), args.cell, args.patch))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'counterfactual_cells.csv', index=False)
    (out / 'counterfactual_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f'{len(table)} cells in {seconds:.2f} s ({1000 * seconds / len(table):.2f} ms per cell, this machine)')


if __name__ == '__main__':
    main()
