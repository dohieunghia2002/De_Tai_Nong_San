"""Explanations of the per-cell spraying decisions, and a test of whether they are faithful.

A cell's decision depends only on its score s = mean P(weed) over the cell (src.weeds.decision), so explaining the
decision means explaining s: which input pixels and which spectral bands raise it. The model has a large receptive
field, so pixels outside the cell can matter too; attributions cover the whole tile.

Attributions of s, compared:
  integrated_gradients   Sundararajan et al. 2017 (ICML), from a 'bare soil' baseline: every band of every pixel set
                         to the band's median over background pixels of the training split. Attributions per pixel
                         and band sum to s(x) - s(baseline) (completeness, checked and reported).
  grad_cam               Selvaraju et al. 2017 (ICCV), on the decoder features (stride 4) that feed the classifier.
  predicted_weed         the model's own P(weed) inside the cell, a reference that needs no explanation method.
Band importance: each band in turn is replaced by its bare-soil value over the whole tile, and the drop in s recorded.
Gradients are taken of the cell's SUM of P(weed) (cell^2 times s) and scaled back: under float16 autocast the
gradient of a 40 000-pixel mean underflows to zero.

Faithfulness (deletion test, Petsiuk et al. 2018, with random controls), for the SPRAY and DEFER cells: the patches
an explanation ranks highest are replaced by bare soil and s is recomputed; the same number of patches drawn at
random over the whole tile, and inside the cell, gives the controls. A faithful explanation removes the weed evidence:
s drops more than under the random controls, and SPRAY decisions flip sooner. Run on Colab GPU:

  python -m src.weeds.explain --data <dir with weedsgalore_<split>.npz> --ckpt <dlv3p_msi_3.pth> --out <dir>
                              [--expect <go_no_go.json reported for this model>]

The decisions are computed once (decisions.csv) and reused by later sessions; per-tile results go to <out>/parts/, so
an interrupted run resumes, and settings are fixed per output folder (xai_config.json). When every tile is done the
parts are merged into xai_cells.csv, xai_deletion.csv, xai_maps.npz and xai_summary.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.weeds.analyze import PRIMARY, clean, primary_decisions, test_location_folds
from src.weeds.authors import load_checkpoint
from src.weeds.data import BANDS, load_npz, to_3class
from src.weeds.decision import DEFER, NO_SPRAY, SPRAY, cell_table, decide_conformal, decision_metrics
from src.weeds.train import evaluate

XAI_VERSION = 2  # bump when an explanation or the deletion test changes: parts of another version are not merged
WEED = 2
METHODS = ('integrated_gradients', 'grad_cam', 'predicted_weed')
CONTROLS = ('random_tile', 'random_cell')
FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.1)  # share of the tile's patches removed in the deletion test
MAP_POOL = 4  # attribution maps are stored average-pooled 4 x 4 (600 -> 150 px)


def autocast(x, amp):
    return torch.autocast('cuda', dtype=torch.float16, enabled=amp and x.is_cuda)


def check_grid(height, width, cell, patch):
    if height % cell or width % cell or cell % patch:
        raise ValueError(f'the cell ({cell} px) must divide the tile ({height} x {width}), and the patch ({patch} px) '
                         'the cell')


def soil_values(x, y, step=4):
    """Bare-soil value per band, scaled to [0, 1] like the model input: the median over background pixels (label 0)
    of uint16 tiles x (n, bands, H, W) with labels y (n, H, W), on every step-th pixel."""
    xs, bg = x[:, :, ::step, ::step], y[:, ::step, ::step] == 0
    return np.array([np.median(xs[:, b][bg]) for b in range(x.shape[1])], dtype=np.float32) / 65535.0


def weed_prob(logits):
    return torch.softmax(logits.float(), dim=1)[:, WEED]


def cell_scores(model, x, cell, amp=True):
    """Score s of every cell, (n, H / cell, W / cell), for a batch x (n, bands, H, W) scaled to [0, 1]."""
    with autocast(x, amp):
        logits = model(x)
    return F.avg_pool2d(weed_prob(logits)[:, None], cell)[:, 0]


def cell_sum(p, rc, cell):
    """Sum of a map (..., H, W) over cell rc = (row, col)."""
    r, c = rc
    return p[..., r * cell:(r + 1) * cell, c * cell:(c + 1) * cell].sum((-2, -1))


def integrated_gradients(model, x, base, rc, cell, steps=32, batch=4, amp=True):
    """Attributions (bands, H, W) of the score of cell rc for one tile x (bands, H, W), from the bare-soil image, by
    the midpoint rule over `steps` points; also s(x) - s(baseline), which the attributions should sum to."""
    base_img = base.view(-1, 1, 1).expand_as(x)
    diff = x - base_img
    grad_sum = torch.zeros_like(x)
    alphas = (torch.arange(steps, device=x.device, dtype=torch.float32) + 0.5) / steps
    for i in range(0, steps, batch):
        xi = (base_img + alphas[i:i + batch].view(-1, 1, 1, 1) * diff).detach().requires_grad_(True)
        with autocast(xi, amp):
            logits = model(xi)
        total = cell_sum(weed_prob(logits), rc, cell).sum()  # cell^2 * s per image, kept large for float16
        grad_sum += torch.autograd.grad(total, xi)[0].float().sum(0)
    with torch.no_grad():
        s_x, s_base = cell_scores(model, torch.stack([x, base_img]), cell, amp)[:, rc[0], rc[1]].tolist()
    return diff * grad_sum / (steps * cell * cell), s_x - s_base


def grad_cams(model, x, cells_rc, cell, amp=True):
    """Grad-CAM map (H, W) of the score of each cell in cells_rc, on decoder features computed once for the tile."""
    with torch.no_grad(), autocast(x, amp):
        feat = model.features(x[None])
    maps = []
    for rc in cells_rc:
        f = feat.float().detach().requires_grad_(True)
        with autocast(x, amp):
            logits = model.head(f, x.shape[-2:])
        g = torch.autograd.grad(cell_sum(weed_prob(logits), rc, cell).sum(), f)[0].float() / (cell * cell)
        cam = F.relu((g.mean((2, 3), keepdim=True) * f).sum(1, keepdim=True))
        maps.append(F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0].detach())
    return maps


def patch_sums(m, patch):
    """Sum of a map (H, W) over each patch: (H / patch, W / patch)."""
    return F.avg_pool2d(m[None, None].float(), patch)[0, 0] * patch * patch


def top_mask(scores, n):
    """The n patches with the highest scores, as a boolean mask of the patch grid."""
    mask = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    mask[torch.topk(scores.flatten(), n).indices] = True
    return mask.view_as(scores)


def random_mask(rng, shape, n, region=None):
    """n patches drawn without replacement, from the whole grid or from the patches where `region` is True."""
    cand = np.flatnonzero(np.ones(shape, bool) if region is None else region)
    mask = np.zeros(int(np.prod(shape)), bool)
    mask[rng.choice(cand, size=n, replace=False)] = True
    return torch.from_numpy(mask.reshape(shape))


def occlude(x, base, mask, patch):
    """x (bands, H, W) with the patches of `mask` (H / patch, W / patch) replaced by the bare-soil values."""
    up = mask.to(x.device).repeat_interleave(patch, 0).repeat_interleave(patch, 1)
    return torch.where(up[None], base.view(-1, 1, 1), x)


def cell_region(shape, rc, cell, patch):
    """Patches of the grid that lie inside cell rc."""
    k = cell // patch
    region = np.zeros(shape, bool)
    region[rc[0] * k:(rc[0] + 1) * k, rc[1] * k:(rc[1] + 1) * k] = True
    return region


@torch.no_grad()
def deletion_tile(model, x, base, tested, cell, patch, rng, fractions=FRACTIONS, draws=10, batch=8, amp=True):
    """Deletion test for the tested cells of one tile. tested: {cell index: ((row, col), {method: map (H, W)})}.
    At each fraction, every tested cell is scored after removing the top patches of each of its maps, and random
    patches inside it (`draws` times); random patches over the whole tile (`draws` times) are drawn once per tile,
    since one occluded image gives every cell's score. One row per (cell, occluded image)."""
    check_grid(x.shape[1], x.shape[2], cell, patch)
    grid = (x.shape[1] // patch, x.shape[2] // patch)
    ranked = {k: {name: patch_sums(m, patch) for name, m in maps.items()} for k, (_, maps) in tested.items()}
    jobs = []  # (cell index, or None for an occlusion shared by every tested cell; method; fraction; draw; mask)
    for f in fractions:
        n = int(math.ceil(f * grid[0] * grid[1]))
        jobs += [(None, 'random_tile', f, d, random_mask(rng, grid, n)) for d in range(draws)]
        for k, (rc, _) in tested.items():
            region = cell_region(grid, rc, cell, patch)
            jobs += [(k, name, f, 0, top_mask(sc, n)) for name, sc in ranked[k].items()]
            jobs += [(k, 'random_cell', f, d, random_mask(rng, grid, min(n, int(region.sum())), region))
                     for d in range(draws)]
    rows = []
    for i in range(0, len(jobs), batch):
        part = jobs[i:i + batch]
        s = cell_scores(model, torch.stack([occlude(x, base, j[-1], patch) for j in part]), cell, amp).cpu()
        for (k, name, f, d, _), sj in zip(part, s):
            for kk in (tested if k is None else [k]):
                rc = tested[kk][0]
                rows.append({'cell': kk, 'method': name, 'fraction': f, 'draw': d, 's_after': float(sj[rc])})
    return rows


@torch.no_grad()
def band_ablation(model, x, base, cell, amp=True):
    """Drop in every cell's score when one band at a time is set to its bare-soil value over the whole tile:
    (bands, H / cell, W / cell)."""
    xb = x.repeat(len(base), 1, 1, 1)
    for b in range(len(base)):
        xb[b, b] = base[b]
    s0 = cell_scores(model, x[None], cell, amp)[0]
    return s0[None] - cell_scores(model, xb, cell, amp)


def explain_tile(model, x, base, cells, cell, patch, rng, steps=32, draws=10, ig_batch=4, batch=8, amp=True):
    """Explanations for the cells of one tile. x: tile (bands, H, W) in [0, 1] on the model's device; cells: rows of
    the decision table for this tile (columns cell, decision). Returns one row per cell, the deletion rows of the
    SPRAY and DEFER cells, maps {name: (n_cells, H / 4, W / 4)} in float32, and the time spent."""
    check_grid(x.shape[1], x.shape[2], cell, patch)
    t0 = time.time()
    n_cols = x.shape[2] // cell
    with torch.no_grad():
        with autocast(x, amp):
            pred = weed_prob(model(x[None]))[0]
        s_model = F.avg_pool2d(pred[None, None], cell)[0, 0]
    ablation = band_ablation(model, x, base, cell, amp)
    rcs = [divmod(int(k), n_cols) for k in cells['cell']]
    cams = grad_cams(model, x, rcs, cell, amp)
    cell_rows, ig_maps, cam_maps, tested = [], [], [], {}
    for (_, c), rc, cam in zip(cells.iterrows(), rcs, cams):
        attr, gap = integrated_gradients(model, x, base, rc, cell, steps, ig_batch, amp)
        ig_map = attr.sum(0)
        ig_maps.append(F.avg_pool2d(ig_map[None, None], MAP_POOL)[0, 0].cpu().numpy())
        cam_maps.append(F.avg_pool2d(cam[None, None], MAP_POOL)[0, 0].cpu().numpy())
        k = int(c['cell'])
        cell_rows.append({'cell': k, 's_model': float(s_model[rc]), 'ig_gap': gap, 'ig_sum': float(ig_map.sum()),
                          **{f'ig_{b}': v for b, v in zip(BANDS, attr.sum((1, 2)).tolist())},
                          **{f'ablate_{b}': float(ablation[i][rc]) for i, b in enumerate(BANDS)}})
        if c['decision'] in (SPRAY, DEFER):
            in_cell = torch.zeros_like(pred)
            in_cell[rc[0] * cell:(rc[0] + 1) * cell, rc[1] * cell:(rc[1] + 1) * cell] = 1
            tested[k] = (rc, {'integrated_gradients': ig_map, 'grad_cam': cam, 'predicted_weed': pred * in_cell})
    t1 = time.time()
    dels = deletion_tile(model, x, base, tested, cell, patch, rng, draws=draws, batch=batch, amp=amp) if tested else []
    maps = {'integrated_gradients': np.stack(ig_maps), 'grad_cam': np.stack(cam_maps)}
    return cell_rows, dels, maps, {'explain_s': t1 - t0, 'deletion_s': time.time() - t1, 'tested': len(tested)}


def decisions_for_test(model, test, dev):
    """The model's decisions in the primary setting, from its probabilities on the test split (src.weeds.analyze)."""
    _, probs = evaluate(model, test, dev, keep_probs=True)
    cells = cell_table(test['ids'], to_3class(test['y']), probs[:, 1], PRIMARY['cell_size'])
    return primary_decisions(cells, test_location_folds(list(test['ids'])))


def decision_check(t, expected):
    """Decision metrics and thresholds of table t next to those reported for the same model (a go_no_go.json)."""
    m = decision_metrics(t['decision'].to_numpy(), t['w'].to_numpy() >= PRIMARY['tau'], t['w'].to_numpy(),
                         t['crop'].to_numpy())
    lam = t.groupby('fold')[['lam_lo', 'lam_hi']].first()
    got = {'spray_reduction': m['spray_reduction'], 'deferred': m['deferred'],
           'missed_need_rate': m['missed_need_rate'], 'false_spray_rate': m['false_spray_rate'],
           'lam_lo': lam['lam_lo'].tolist(), 'lam_hi': lam['lam_hi'].tolist()}
    if expected is None:
        return {'computed': got}
    want = {'spray_reduction': expected['checks']['C2_spray_reduction']['value'],
            'deferred': expected['checks']['C3_deferred']['value'],
            **{k: expected['info'][k] for k in ('missed_need_rate', 'false_spray_rate', 'lam_lo', 'lam_hi')}}
    diff = max(float(np.max(np.abs(np.subtract(got[k], want[k])))) for k in want)
    return {'computed': got, 'reported': want, 'max_abs_diff': diff}


def faithfulness(deletion, cells):
    """Per method and removed fraction: mean relative drop of s, share of cells where the drop beats the mean drop of
    each random control, and the share of SPRAY cells that are no longer SPRAY. SPRAY is judged on the rescored s
    (s_model) with the cell's thresholds, so that a cell sitting on a threshold does not count as flipped by
    rounding alone."""
    if deletion.empty:  # no SPRAY or DEFER cell among those explained
        return {}
    df = deletion.merge(cells[['tile', 'cell', 's_model', 'lam_lo', 'lam_hi']], on=['tile', 'cell'])
    df['spray_before'] = decide_conformal(df['s_model'], df['lam_lo'], df['lam_hi']) == SPRAY
    df['drop'] = df['s_model'] - df['s_after']
    df['rel_drop'] = df['drop'] / df['s_model']
    df['flipped'] = decide_conformal(df['s_after'], df['lam_lo'], df['lam_hi']) != SPRAY
    per_cell = df.groupby(['tile', 'cell', 'spray_before', 'method', 'fraction'], as_index=False)[
        ['drop', 'rel_drop', 'flipped']].mean()
    out = {}
    for f, g in per_cell.groupby('fraction'):
        rand = {ctrl: g[g['method'] == ctrl].set_index(['tile', 'cell'])['drop'] for ctrl in CONTROLS}
        for m in METHODS + CONTROLS:
            gm = g[g['method'] == m].set_index(['tile', 'cell'])
            res = {'cells': len(gm), 'mean_rel_drop': float(gm['rel_drop'].mean()),
                   'spray_flip_rate': float(gm[gm['spray_before']]['flipped'].mean())}
            if m in METHODS:
                for ctrl in CONTROLS:
                    res[f'beats_{ctrl}'] = float((gm['drop'] > rand[ctrl].loc[gm.index]).mean())
            out.setdefault(m, {})[str(f)] = res
    return out


def run(args, dev):
    out = Path(args.out)
    parts = out / 'parts'
    parts.mkdir(parents=True, exist_ok=True)
    config = {'version': XAI_VERSION, 'ckpt': Path(args.ckpt).name,
              **{k: getattr(args, k) for k in ('steps', 'patch', 'draws', 'seed', 'amp')}}
    if (out / 'xai_config.json').exists() and json.loads((out / 'xai_config.json').read_text()) != config:
        raise ValueError(f'{out} holds results made with other settings ({(out / "xai_config.json").read_text()}); '
                         f'this run has {config}: use a new output folder')
    (out / 'xai_config.json').write_text(json.dumps(config))

    model = load_checkpoint(args.ckpt, len(BANDS), 3, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    if (out / 'soil_baseline.json').exists():
        soil = np.array(json.loads((out / 'soil_baseline.json').read_text())['values'], dtype=np.float32)
    else:
        tr = load_npz(Path(args.data) / 'weedsgalore_train.npz')
        soil = soil_values(tr['x'], tr['y'])
        del tr
        (out / 'soil_baseline.json').write_text(json.dumps({'bands': BANDS, 'values': soil.tolist(),
                                                            'source': 'median of background pixels, train split'}))
    base = torch.from_numpy(soil).to(dev)

    test = load_npz(Path(args.data) / 'weedsgalore_test.npz')
    if (out / 'decisions.csv').exists():  # fixed by the first session, so every tile explains the same decisions
        cells = pd.read_csv(out / 'decisions.csv')
    else:
        cells = decisions_for_test(model, test, dev)
        expected = json.loads(Path(args.expect).read_text()) if args.expect else None
        check = decision_check(cells, expected)
        (out / 'decision_check.json').write_text(json.dumps(clean(check), indent=1))
        if check.get('max_abs_diff', 0) > 1e-6:
            print(f'WARNING: decisions differ from the reported ones (max abs diff {check["max_abs_diff"]:.2e}); '
                  'see decision_check.json', flush=True)
        cells.to_csv(out / 'decisions.csv', index=False)
    counts = {name: int((cells['decision'] == k).sum()) for k, name in ((SPRAY, 'spray'), (NO_SPRAY, 'no_spray'),
                                                                         (DEFER, 'defer'))}
    print(f'{len(cells)} cells, decisions {counts}', flush=True)

    tiles = list(dict.fromkeys(cells['tile']))[:args.limit_tiles or None]
    if not all((parts / f'{t}.npz').exists() for t in tiles):
        (out / 'xai_summary.json').unlink(missing_ok=True)  # only a finished run has a summary
    index = {t: i for i, t in enumerate(test['ids'])}
    t0 = time.time()
    for k, tile in enumerate(tiles):
        part = parts / f'{tile}.npz'
        if part.exists():
            continue
        if time.time() - t0 > args.budget_hours * 3600:
            print('time budget reached: rerun to resume', flush=True)
            return
        x = torch.from_numpy(test['x'][index[tile]].astype(np.float32) / 65535.0).to(dev)
        tile_cells = cells[cells['tile'] == tile]
        rows, dels, maps, timing = explain_tile(model, x, base, tile_cells, PRIMARY['cell_size'], args.patch,
                                                np.random.default_rng([args.seed, k]), args.steps, args.draws,
                                                args.ig_batch, args.batch, args.amp)
        for r, (_, c) in zip(rows, tile_cells.iterrows()):
            r['decision'] = int(c['decision'])  # kept with the part, checked when merging
        tmp = parts / f'{tile}.tmp.npz'
        np.savez_compressed(tmp, cells=json.dumps(rows), deletion=json.dumps(dels), **maps)
        tmp.replace(part)
        err = max(abs(r['ig_sum'] - r['ig_gap']) for r in rows)
        print(f'{tile}: {len(rows)} cells ({timing["tested"]} tested) in {timing["explain_s"]:.0f} s + deletion '
              f'{timing["deletion_s"]:.0f} s | integrated gradients completeness: max abs error {err:.2e}', flush=True)
        if args.limit_tiles:
            tested_all = int(cells['decision'].isin([SPRAY, DEFER]).sum())
            per_tested = timing['deletion_s'] / max(timing['tested'], 1)
            est = timing['explain_s'] * cells['tile'].nunique() + per_tested * tested_all
            print(f'estimated full run: {est / 60:.0f} min ({cells["tile"].nunique()} tiles, {tested_all} tested '
                  'cells)', flush=True)

    cell_rows, del_rows, maps = [], [], {m: [] for m in ('integrated_gradients', 'grad_cam')}
    for tile in tiles:
        with np.load(parts / f'{tile}.npz') as p:
            cell_rows += [{'tile': tile, **r} for r in json.loads(str(p['cells']))]
            del_rows += [{'tile': tile, **r} for r in json.loads(str(p['deletion']))]
            for m in maps:
                maps[m].append(p[m])
    merged = pd.DataFrame(cell_rows).rename(columns={'decision': 'decision_part'})
    xai = cells.merge(merged, on=['tile', 'cell'])
    if (xai['decision'] != xai['decision_part']).any():
        raise ValueError('parts were computed for other decisions than decisions.csv: use a new output folder')
    xai = xai.drop(columns='decision_part')
    deletion = pd.DataFrame(del_rows)
    xai.to_csv(out / 'xai_cells.csv', index=False)
    deletion.to_csv(out / 'xai_deletion.csv', index=False)
    # maps are stacked in the order of cell_rows: one map per (tile, cell)
    np.savez_compressed(out / 'xai_maps.npz', tile=np.array([r['tile'] for r in cell_rows]),
                        cell=np.array([r['cell'] for r in cell_rows]), **{m: np.concatenate(v) for m, v in maps.items()})
    err = (xai['ig_sum'] - xai['ig_gap']).abs()
    rescored = decide_conformal(xai['s_model'], xai['lam_lo'], xai['lam_hi'])
    summary = {'config': config, 'cells': len(xai), 'decisions': counts,
               'decision_check': json.loads((out / 'decision_check.json').read_text())
               if (out / 'decision_check.json').exists() else None,
               'rescored_decision_differs': int((rescored != xai['decision']).sum()),
               'score_check_max_abs_diff': float((xai['s_model'] - xai['s']).abs().max()),
               'soil_baseline': dict(zip(BANDS, soil.tolist())),
               'ig_completeness_abs_error': {'median': float(err.median()), 'max': float(err.max())},
               'ig_completeness_rel_error_median': float((err / xai['ig_gap'].abs().clip(lower=1e-6)).median()),
               'faithfulness': faithfulness(deletion, xai)}
    (out / 'xai_summary.json').write_text(json.dumps(clean(summary), indent=1))
    print(json.dumps(clean({k: summary[k] for k in ('decisions', 'rescored_decision_differs',
                                                    'ig_completeness_abs_error')}), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--expect', default=None, help='go_no_go.json reported for this model, to check the decisions')
    ap.add_argument('--steps', type=int, default=32, help='integrated gradients: points on the path')
    ap.add_argument('--patch', type=int, default=20, help='deletion test: patch size in pixels (5 cm)')
    ap.add_argument('--draws', type=int, default=10, help='deletion test: random draws per control and fraction')
    ap.add_argument('--ig-batch', type=int, default=4)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-amp', dest='amp', action='store_false', help='float32 throughout (slower, more memory)')
    ap.add_argument('--budget-hours', type=float, default=3.0)
    ap.add_argument('--limit-tiles', type=int, default=0, help='explain only the first N tiles (smoke test)')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
    run(args, dev)


if __name__ == '__main__':
    main()
