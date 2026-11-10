"""Model check of the counterfactual explanations of src.weeds.counterfactual.

src.weeds.counterfactual estimates from the P(weed) map alone, without running the model again, the fewest 20 x 20 px
patches of a cell whose removal ends SPRAY (k_not_spray) and gives NO_SPRAY (k_no_spray). Here the model rescores the
tile after the first k = 1, 2, ... ranked patches of the cell are replaced by bare soil (the values of
src.weeds.explain), for every SPRAY and DEFER cell. Patches are ranked by the model's own P(weed) mass inside the cell,
as the 'predicted_weed' ranking of the deletion test, i.e. in the order the explanation names them. The scan stops
once the model gives NO_SPRAY and k has passed it and both estimates by `extra` patches, or when every patch of the
cell is gone. The first k at which the model's decision changes checks the explanation as it is shown ("remove these
k patches and the cell is no longer sprayed"). It is not the fewest patches for the model: with context, another set
of patches could change the decision with fewer. The scan also shows whether the decision changes back at a larger k,
since the model's score need not fall monotonically. Run on Colab GPU:

  python -m src.weeds.counterfactual_check --data <dir with weedsgalore_test.npz> --ckpt <dlv3p_msi_3.pth>
      --decisions <decisions.csv of src.weeds.explain> --soil <soil_baseline.json of src.weeds.explain> --out <dir>

Per-tile results go to <out>/parts/, so an interrupted run resumes; settings are fixed per output folder.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.weeds.analyze import PRIMARY, clean
from src.weeds.authors import load_checkpoint
from src.weeds.counterfactual import GSD_CM, PATCH, first_index, ranked_patches, removal_path
from src.weeds.data import BANDS, load_npz
from src.weeds.decision import DEFER, NO_SPRAY, SPRAY, decide_conformal
from src.weeds.explain import autocast, cell_scores, check_grid, occlude, weed_prob

CHECK_VERSION = 1  # bump when the scan changes: parts of another version are not merged


def weed_map(model, x, amp=True):
    """The model's P(weed) map (H, W) of one tile x (bands, H, W), float32."""
    with torch.no_grad(), autocast(x, amp):
        return weed_prob(model(x[None]))[0]


def cell_ranking(pred, rc, cell, patch):
    """For cell rc = (row, col) of a P(weed) map: its patches ranked by P(weed) mass, largest first, as row-major
    indices in the tile's patch grid; the cell's score s; and the estimated scores after removing 0, 1, ... of them."""
    r, c = rc
    order, masses = ranked_patches(pred[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell].double().cpu().numpy(),
                                   patch)
    side, grid_cols = cell // patch, pred.shape[1] // patch
    s = float(pred[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell].double().mean())
    return (r * side + order // side) * grid_cols + c * side + order % side, s, removal_path(s, masses, cell)


def estimated_k(est, lam_lo, lam_hi):
    d = decide_conformal(est, lam_lo, lam_hi)
    return first_index(d != SPRAY), first_index(d == NO_SPRAY)


def scan_target(no_spray_at, k_estimates, extra):
    """How far the scan must go once the model has given NO_SPRAY at k = no_spray_at."""
    return max([no_spray_at] + [int(k) for k in k_estimates if np.isfinite(k)]) + extra


@torch.no_grad()
def verify_tile(model, x, base, cells, cell, patch=PATCH, batch=8, extra=2, amp=True):
    """The scan for some cells of one tile: decision-table rows with columns cell, decision, lam_lo, lam_hi. x: the
    tile (bands, H, W) in [0, 1] on the model's device; base: bare-soil value per band. Returns one row per scored
    (cell, k), one row per cell, and the number of forward passes."""
    check_grid(x.shape[1], x.shape[2], cell, patch)
    pred = weed_map(model, x, amp)
    n_cols, grid = x.shape[2] // cell, (x.shape[1] // patch, x.shape[2] // patch)
    passes, step_rows, cell_rows = 1, [], []
    for _, c in cells.iterrows():
        lo, hi = float(c['lam_lo']), float(c['lam_hi'])
        rc = divmod(int(c['cell']), n_cols)
        patches, s0, est = cell_ranking(pred, rc, cell, patch)
        k_est = estimated_k(est, lo, hi)
        scores = [s0]
        no_at = 0 if decide_conformal(np.array([s0]), lo, hi)[0] == NO_SPRAY else None
        k = 0
        while k < len(patches) and (no_at is None or k < scan_target(no_at, k_est, extra)):
            ks = range(k + 1, min(len(patches), k + batch) + 1)
            imgs = []
            for kk in ks:
                mask = torch.zeros(grid[0] * grid[1], dtype=torch.bool)
                mask[torch.from_numpy(patches[:kk].astype(np.int64))] = True
                imgs.append(occlude(x, base, mask.view(grid), patch))
            s = cell_scores(model, torch.stack(imgs), cell, amp)[:, rc[0], rc[1]].tolist()
            passes += len(imgs)
            for kk, sv in zip(ks, s):
                scores.append(sv)
                if no_at is None and decide_conformal(np.array([sv]), lo, hi)[0] == NO_SPRAY:
                    no_at = kk
            k = ks[-1]
        scores = np.array(scores)
        dec_model = decide_conformal(scores, lo, hi)
        dec_est = decide_conformal(est[:len(scores)], lo, hi)
        for kk in range(len(scores)):
            step_rows.append({'cell': int(c['cell']), 'k': kk, 's_model': float(scores[kk]),
                              's_estimate': float(est[kk]), 'decision_model': int(dec_model[kk]),
                              'decision_estimate': int(dec_est[kk])})
        k_not, k_no = first_index(dec_model != SPRAY), first_index(dec_model == NO_SPRAY)
        cell_rows.append({
            'cell': int(c['cell']), 'decision': int(c['decision']), 'decision_rescored': int(dec_model[0]),
            's_model': s0, 'lam_lo': lo, 'lam_hi': hi,
            'k_not_spray_estimate': k_est[0], 'k_no_spray_estimate': k_est[1],
            'k_not_spray_model': k_not, 'k_no_spray_model': k_no, 'k_scanned': len(scores) - 1,
            'reverts_to_spray': bool(np.isfinite(k_not) and (dec_model[int(k_not):] == SPRAY).any()),
            'reverts_from_no_spray': bool(np.isfinite(k_no) and (dec_model[int(k_no):] != NO_SPRAY).any()),
            'max_score_increase': float(max(np.diff(scores).max(initial=0.0), 0.0)),
            'patches': ' '.join(str(int(p)) for p in patches[:len(scores) - 1])})
    return step_rows, cell_rows, passes


def planned_passes(model, tiles, cells, cell, patch, batch, extra, amp=True):
    """Forward passes the scan needs if the model agrees with the estimate: one pass per tile for its P(weed) map,
    then per cell the batches up to the estimated k plus `extra` (every patch when the estimate never gives NO_SPRAY).
    More passes are needed where the model needs more patches than the estimate, fewer where the estimate never
    gives NO_SPRAY but the model does.
    tiles: {tile id: tile tensor on the model's device}; cells: decision-table rows of those tiles."""
    total = 0
    for tile, x in tiles.items():
        pred = weed_map(model, x, amp)
        total += 1
        for _, c in cells[cells['tile'] == tile].iterrows():
            patches, _, est = cell_ranking(pred, divmod(int(c['cell']), x.shape[2] // cell), cell, patch)
            k_est = estimated_k(est, float(c['lam_lo']), float(c['lam_hi']))
            finite = [int(k) for k in k_est if np.isfinite(k)]
            target = min(len(patches), max(finite) + extra if finite else len(patches))
            total += min(len(patches), int(math.ceil(target / batch)) * batch)
    return total


def _compare(model_k, est_k):
    ok = model_k.notna() & est_k.notna()
    d = (model_k - est_k)[ok].astype(int)
    return {'cells': int(len(model_k)), 'compared': int(ok.sum()), 'model_not_reached': int(model_k.isna().sum()),
            'estimate_not_reached': int(est_k.isna().sum()), 'exact': int((d == 0).sum()),
            'estimate_too_small': int((d > 0).sum()), 'estimate_too_large': int((d < 0).sum()),
            'model_minus_estimate': {str(v): int(n) for v, n in d.value_counts().sort_index().items()}}


def summarize_check(cells, steps, cell, gsd_cm=GSD_CM):
    """cells: one row per checked cell; steps: one row per (tile, cell, k) scored (k = 0 is the unoccluded tile)."""
    cell_cm2 = cell ** 2 * gsd_cm ** 2
    sp = cells[cells['decision_rescored'] == SPRAY]
    tested = cells[cells['decision_rescored'].isin([SPRAY, DEFER])]
    fragile = sp[sp['k_not_spray_estimate'] <= 2]
    st = steps.merge(cells[['tile', 'cell', 'decision_rescored']], on=['tile', 'cell'])
    st = st[st['k'] > 0]
    err = (st['s_estimate'] - st['s_model']) * cell_cm2  # < 0: the estimate falls further than the model
    agree = st['decision_model'] == st['decision_estimate']
    changed = (st['decision_model'] != st['decision_rescored']) | (st['decision_estimate'] != st['decision_rescored'])
    inc = cells['max_score_increase'] * cell_cm2
    return {'cells': len(cells),
            'decisions': {name: int((cells['decision'] == v).sum()) for v, name in ((SPRAY, 'spray'),
                                                                                    (DEFER, 'defer'))},
            'rescored_decision_differs': int((cells['decision'] != cells['decision_rescored']).sum()),
            'k_not_spray': _compare(sp['k_not_spray_model'], sp['k_not_spray_estimate']),
            'k_not_spray_fragile': _compare(fragile['k_not_spray_model'], fragile['k_not_spray_estimate']),
            'k_no_spray': _compare(tested['k_no_spray_model'], tested['k_no_spray_estimate']),
            'reverts_to_spray': int(cells['reverts_to_spray'].sum()),
            'reverts_from_no_spray': int(cells['reverts_from_no_spray'].sum()),
            'max_score_increase_cm2': {'median': float(inc.median()), 'max': float(inc.max())},
            'curve': {'rows': len(st), 'error_cm2_median': float(err.median()),
                      'abs_error_cm2_median': float(err.abs().median()),
                      'abs_error_cm2_p95': float(err.abs().quantile(.95)), 'abs_error_cm2_max': float(err.abs().max()),
                      'decision_agreement': float(agree.mean()), 'rows_decision_changed': int(changed.sum()),
                      'decision_agreement_where_changed': float(agree[changed].mean()) if changed.any() else None}}


def run(args, dev):
    out = Path(args.out)
    parts = out / 'parts'
    parts.mkdir(parents=True, exist_ok=True)
    soil = json.loads(Path(args.soil).read_text())
    if list(soil['bands']) != list(BANDS):
        raise ValueError(f'soil values are for bands {soil["bands"]}, the model takes {BANDS}')
    # the inputs are part of the settings, so that parts made for other decisions or soil values are never merged
    config = {'version': CHECK_VERSION, 'ckpt': Path(args.ckpt).name, 'cell': args.cell, 'patch': args.patch,
              'extra': args.extra, 'amp': args.amp, 'soil': [float(v) for v in soil['values']],
              'decisions_sha256': hashlib.sha256(Path(args.decisions).read_bytes()).hexdigest()}
    cfg = out / 'cfcheck_config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError(f'{out} holds results made with other settings ({cfg.read_text()}); this run has {config}: '
                         'use a new output folder')
    cfg.write_text(json.dumps(config))
    base = torch.tensor(soil['values'], dtype=torch.float32, device=dev)
    decisions = pd.read_csv(args.decisions)
    todo = decisions[decisions['decision'].isin([SPRAY, DEFER])]
    model = load_checkpoint(args.ckpt, len(BANDS), 3, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    test = load_npz(Path(args.data) / 'weedsgalore_test.npz')
    index = {t: i for i, t in enumerate(test['ids'])}

    def tile_tensor(t):
        return torch.from_numpy(test['x'][index[t]].astype(np.float32) / 65535.0).to(dev)

    all_tiles = list(dict.fromkeys(todo['tile']))
    tiles = all_tiles[:args.limit_tiles or None]
    cell = args.cell
    t0, done_passes, done_s = time.time(), 0, 0.0
    for tile in tiles:
        part = parts / f'{tile}.json'
        if part.exists():
            continue
        if time.time() - t0 > args.budget_hours * 3600:
            print('time budget reached: rerun to resume', flush=True)
            return
        t1 = time.time()
        steps, cells, passes = verify_tile(model, tile_tensor(tile), base, todo[todo['tile'] == tile], cell,
                                           args.patch, args.batch, args.extra, args.amp)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        sec = time.time() - t1
        done_passes, done_s = done_passes + passes, done_s + sec
        tmp = part.with_suffix('.tmp')
        tmp.write_text(json.dumps(clean({'steps': steps, 'cells': cells, 'passes': passes, 'seconds': sec})))
        tmp.replace(part)
        print(f'{tile}: {len(cells)} cells, {passes} forward passes in {sec:.0f} s', flush=True)
    if args.limit_tiles:
        if done_passes:
            planned = planned_passes(model, {t: tile_tensor(t) for t in all_tiles}, todo, cell, args.patch,
                                     args.batch, args.extra, args.amp)
            print(f'estimated full run: about {planned * done_s / done_passes / 60:.0f} min ({planned} forward '
                  f'passes at {done_s / done_passes:.3f} s each; more if the model needs more patches than the '
                  'estimate)', flush=True)
        return
    if not all((parts / f'{t}.json').exists() for t in all_tiles):
        return
    steps, cells, passes, seconds = [], [], 0, 0.0
    for tile in all_tiles:
        d = json.loads((parts / f'{tile}.json').read_text())
        steps += [{'tile': tile, **r} for r in d['steps']]
        cells += [{'tile': tile, **r} for r in d['cells']]
        passes, seconds = passes + d['passes'], seconds + d['seconds']
    steps = pd.DataFrame(steps)
    cells = pd.DataFrame(cells).merge(todo[['tile', 'cell', 'date']], on=['tile', 'cell'])
    for col in [c for c in cells if c.startswith('k_') and c.endswith(('_estimate', '_model'))]:
        cells[col] = cells[col].astype(float)  # None (not reached) -> NaN
    steps.to_csv(out / 'cfcheck_steps.csv', index=False)
    cells.to_csv(out / 'cfcheck_cells.csv', index=False)
    summary = {'config': config, 'soil_baseline': dict(zip(BANDS, soil['values'])),
               **summarize_check(cells, steps, cell),
               'forward_passes': passes, 'seconds': seconds, 'seconds_per_cell': seconds / len(cells)}
    (out / 'cfcheck_summary.json').write_text(json.dumps(clean(summary), indent=1))
    print(json.dumps(clean({k: summary[k] for k in ('cells', 'rescored_decision_differs', 'k_not_spray',
                                                    'k_no_spray', 'reverts_to_spray', 'curve')}), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--decisions', required=True, help='decisions.csv written by src.weeds.explain for this model')
    ap.add_argument('--soil', required=True, help='soil_baseline.json written by src.weeds.explain')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cell', type=int, default=PRIMARY['cell_size'])
    ap.add_argument('--patch', type=int, default=PATCH)
    ap.add_argument('--extra', type=int, default=2, help='patches scanned past NO_SPRAY and both estimates')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--no-amp', dest='amp', action='store_false', help='float32 throughout (slower, more memory)')
    ap.add_argument('--budget-hours', type=float, default=3.0)
    ap.add_argument('--limit-tiles', type=int, default=0, help='check only the first N tiles (smoke test)')
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
    run(args, dev)


if __name__ == '__main__':
    main()
