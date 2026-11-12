"""Data and computations behind the progress demo (demo/app.py). No Streamlit here, so tests can import it.

Every number the demo shows is read from results/ or recomputed from the saved probability maps and the dataset's
labels with the project's own functions (src.weeds); none is typed in by hand. When a recomputed decision setting is
also in a run's decision_metrics.csv, saved_difference measures how far the two are apart.
"""
import json
import math
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
for _p in (ROOT_DIR / 'scripts', ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_counterfactual_figure as CF  # noqa: E402  (inputs and drawing helpers of the explanation figure)
from export_weeds_images import CLASSES, B, G, NIR, R, band_limits, composite, date_vi  # noqa: E402,F401
from src.weeds.analyze import (ALPHA, BETA, CAL_TEST_2FOLD, CAL_VAL, CRITERIA, PRIMARY, SCOPES,  # noqa: E402,F401
                               test_location_folds)
from src.weeds.counterfactual import GSD_CM, PATCH  # noqa: E402,F401
from src.weeds.data import (BANDS, DATES, PRE_HERBICIDE_DATES, ROOT, SEMANTIC_CLASSES, SPLITS, TILE,  # noqa: E402,F401
                            location_blocks, read_png, read_tile, split_ids, tile_date, to_3class)
from src.weeds.decision import (DEFER, NO_SPRAY, SPRAY, cell_table, crossfit_decisions,  # noqa: E402,F401
                                evaluate_crossfit, evaluate_policies, policy_decisions)

ZIP = Path(os.environ.get('WEEDSGALORE_ZIP', ROOT_DIR / 'data/weedsgalore/weedsgalore-dataset.zip'))
RESULTS = ROOT_DIR / 'results'
IMAGES = ROOT_DIR / 'data/weedsgalore'  # preview images and figures made by scripts/ (local only, not in git)

# Segmentation models whose probability maps are saved; 'batch' is documented in CLAUDE.md where train_done.json
# predates the field.
MODELS = {  # labels short enough for the sidebar's select box
    'authors_msi': {'label': 'Tác giả, 5 dải phổ (chính)', 'dir': 'weeds_authors/dlv3p_msi_3'},
    'authors_rgb': {'label': 'Tác giả, chỉ ảnh RGB', 'dir': 'weeds_authors/dlv3p_rgb_3'},
    'own_b8': {'label': 'Đề tài, lần 2 (batch 8)', 'dir': 'weeds_pilot_b8', 'batch': 8},
    'own_b2': {'label': 'Đề tài, lần 1 (batch 2)', 'dir': 'weeds_pilot', 'batch': 2},
}
MAIN_MODEL = 'authors_msi'
POLICIES = ('conformal_3way', 'plug_in', 'conformal_miss_only', 'oracle', 'broadcast')
POLICY_VI = {
    'conformal_3way': 'Đề tài: hai ngưỡng conformal (có CHƯA CHẮC)',
    'plug_in': 'Ngưỡng cố định bằng τ',
    'conformal_miss_only': 'Một ngưỡng conformal, chỉ chặn bỏ sót',
    'oracle': 'Biết trước đáp án (mức tốt nhất có thể)',
    'broadcast': 'Phun cả ruộng',
}
DECISION_VI = {d: name for d, (name, _) in CF.DECISION.items()}
DECISION_COLOR = {d: colour for d, (_, colour) in CF.DECISION.items()}
SCOPE_VI = {'pre-herbicide pooled': 'Gộp 25/5, 30/5, 6/6 (chính)',  # the 3 flights before spraying; fits the sidebar
            **{d: f'Riêng đợt bay {date_vi(d + "_")[:5]}' for d in DATES}}
CALIBRATION_VI = {CAL_TEST_2FOLD: 'Chéo 2 phần theo vị trí trên tập kiểm tra (phương án chính)',
                  CAL_VAL: 'Tập kiểm định val (đã dùng để chọn mô hình, phương án phụ)'}
METRICS = ('n_cells', 'infestation', 'sprayed', 'deferred', 'effective_sprayed', 'spray_reduction',
           'missed_need_rate', 'false_spray_rate', 'missed_weed_area', 'crop_area_sprayed', 'spray_surplus')
PRIMARY_SETTING = {'cell': PRIMARY['cell_size'], 'tau': PRIMARY['tau'], 'alpha': ALPHA, 'beta': BETA,
                   'scope': PRIMARY['scope'], 'calibration': PRIMARY['calibration']}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def model_dir(key):
    return RESULTS / MODELS[key]['dir']


def available_models():
    """Models with both saved probability maps (they are not in git: results/**/probs_*.npz)."""
    return [k for k in MODELS if all((model_dir(k) / f'probs_{s}.npz').exists() for s in ('val', 'test'))]


def load_probs(key, split):
    """Tile ids, P(crop) and P(weed) maps (float16, as saved) of a model on a split."""
    with np.load(model_dir(key) / f'probs_{split}.npz', allow_pickle=False) as z:
        return z['ids'].astype(str), z['p_crop'], z['p_weed']


def dataset_ids(zip_path=ZIP):
    """Tile ids of each split, in the order of the split files."""
    with zipfile.ZipFile(zip_path) as zf:
        return {s: split_ids(zf, s) for s in SPLITS}


def check_order(ids, split_file_ids):
    """The saved maps must follow the split file, as src.weeds.analyze.load_cells requires."""
    if list(ids) != list(split_file_ids)[:len(ids)]:
        raise ValueError('probability maps and split file are not in the same tile order')


def read_masks(ids, zip_path=ZIP):
    """Semantic masks (values 0-5) of the tiles, stacked in the order of ids."""
    with zipfile.ZipFile(zip_path) as zf:
        return np.stack([read_png(zf, f'{ROOT}/{tile_date(t)}/semantics/{t}.png') for t in ids]).astype(np.uint8)


def read_bands(tile, zip_path=ZIP):
    """(5, 600, 600) uint16 bands in BANDS order and the semantic mask of one tile."""
    with zipfile.ZipFile(zip_path) as zf:
        return read_tile(zf, tile)


def dataset_band_limits(zip_path=ZIP):
    """Brightness stretch of the preview images (scripts/export_weeds_images.py): 1st / 99th percentile per band."""
    with zipfile.ZipFile(zip_path) as zf:
        return band_limits(zf, sorted(t for s in SPLITS for t in split_ids(zf, s)))


def locations(test_ids):
    """Field location (1-8) and calibration part (0 / 1) of every test tile, as src.weeds.analyze assigns them."""
    fold_of = test_location_folds(list(test_ids))
    loc = {t: i + 1 for i, block in enumerate(location_blocks(list(test_ids))) for t in block}
    return loc, fold_of


# ---------------------------------------------------------------- decisions

def decide(test_cells, val_cells, fold_of, tau, alpha, beta, dates, calibration):
    """Every policy's decisions on the test cells of `dates` and their metrics, as src.weeds.analyze computes them.

    test_cells / val_cells: src.weeds.decision.cell_table of one model and cell size. Returns None when the setting
    cannot be calibrated (one location part only, or no val cell of those dates); otherwise (cells, metrics, parts):
      cells    the test cells with 'cell' (row-major index in the tile), 'fold', 'need', the thresholds applied to
               each cell and one decision column per policy;
      metrics  one row per policy in POLICIES order, with the columns of decision_metrics.csv plus counts;
      parts    one dict per calibration part (thresholds and number of calibration cells)."""
    t = test_cells[test_cells['date'].isin(dates)].reset_index(drop=True).copy()
    if t.empty:
        return None
    t['cell'] = t.groupby('tile').cumcount()
    t['fold'] = t['tile'].map(fold_of).to_numpy()
    if calibration == CAL_TEST_2FOLD:
        res = crossfit_decisions(t, t['fold'].to_numpy(), tau, alpha, beta)
        if res is None:
            return None
        policies, per_row, parts = res
        rows = evaluate_crossfit(t, t['fold'].to_numpy(), tau, alpha, beta)
        t['lam_lo'], t['lam_hi'] = per_row['lam_lo'], per_row['lam_hi']
    elif calibration == CAL_VAL:
        v = val_cells[val_cells['date'].isin(dates)]
        if v.empty:
            return None
        policies, info = policy_decisions(v, t, tau, alpha, beta)
        rows = evaluate_policies(v, t, tau, alpha, beta)
        t['lam_lo'], t['lam_hi'] = info['lam_lo'], info['lam_hi']
        parts = [info]
    else:
        raise ValueError(f'unknown calibration: {calibration}')
    need = (t['w'] >= tau).to_numpy()
    t['need'] = need
    metrics = pd.DataFrame(rows).set_index('policy')
    for name, d in policies.items():
        t[name] = d
        metrics.loc[name, 'n_need'] = need.sum()
        metrics.loc[name, 'n_deferred'] = (d == DEFER).sum()
        metrics.loc[name, 'n_missed'] = ((d == NO_SPRAY) & need).sum()
        metrics.loc[name, 'n_false_spray'] = ((d == SPRAY) & ~need).sum()
    return t, metrics.loc[list(POLICIES)].reset_index(), parts


def saved_metrics(key):
    p = model_dir(key) / 'decision_metrics.csv'
    return pd.read_csv(p) if p.exists() else None


def saved_difference(metrics, saved, cell, tau, alpha, beta, scope, calibration):
    """Largest absolute difference between recomputed metrics (decide) and the rows of decision_metrics.csv for the
    same setting; None when the file does not hold that setting (it covers alpha = beta = 0.1 and a grid of tau)."""
    if saved is None or not (math.isclose(alpha, ALPHA) and math.isclose(beta, BETA)):
        return None
    s = saved[(saved['cell_size'] == cell) & (saved['scope'] == scope) & (saved['calibration'] == calibration)
              & np.isclose(saved['tau'], tau, rtol=0, atol=1e-12)]
    if s.empty:
        return None
    m = metrics.set_index('policy')
    s = s.set_index('policy')
    if set(m.index) != set(s.index):
        return math.inf
    a, b = m.loc[s.index, list(METRICS)].astype(float), s[list(METRICS)].astype(float)
    diff = (a - b).abs().where(~(a.isna() & b.isna()), 0.0)
    return float(diff.to_numpy().max()) if not diff.isna().to_numpy().any() else math.inf


def cell_cm2(cell):
    return (cell * GSD_CM) ** 2


# ---------------------------------------------------------------- segmentation

def predicted_classes(p_crop, p_weed):
    """Most probable class per pixel (0 background, 1 crop, 2 weed) from the saved probability maps."""
    pc, pw = p_crop.astype(np.float32), p_weed.astype(np.float32)
    return np.stack([1 - pc - pw, pc, pw]).argmax(0).astype(np.uint8)


def iou_per_class(pred, true, n=3):
    """IoU of each class in %, NaN for a class absent from both."""
    out = []
    for k in range(n):
        union = ((pred == k) | (true == k)).sum()
        out.append(100 * ((pred == k) & (true == k)).sum() / union if union else np.nan)
    return np.array(out)


def authors_vs_paper():
    """The authors' published models evaluated with the project's code (fp32) next to the paper's test mIoU."""
    d = read_json(RESULTS / 'weeds_authors/authors_vs_paper.json')
    return pd.DataFrame([{'model': k, 'bands': r['bands'], 'classes': r['classes'], 'ours': r['fp32_test']['miou'],
                          'paper': r['paper']['miou'], 'diff': r['fp32_test_minus_paper_miou']}
                         for k, r in d.items()])


def test_iou(key):
    """Test IoU (%) of a model as the report gives it: for the authors' checkpoints the float32 pass (as their
    evaluate.py and the paper; the file's 'test' key holds the float16 pass, up to 0.005 points off), for the
    project's own runs the 'test' key."""
    d = read_json(model_dir(key) / 'segmentation_metrics.json')
    return d['fp32']['test'] if 'fp32' in d else d['test']


def segmentation_table():
    """Test IoU (%) of every model with saved results (test_iou)."""
    rows = []
    for k, m in MODELS.items():
        if (model_dir(k) / 'segmentation_metrics.json').exists():
            t = test_iou(k)
            rows.append({'key': k, 'model': m['label'], 'miou': t['miou'], 'background': t['background'],
                         'crop': t['crop'], 'weed': t['weed']})
    return pd.DataFrame(rows)


def training_runs():
    """Every training run of the project: results/weeds_runs/runs_summary.csv plus the two pilot runs (their
    segmentation_metrics.json, best.json, train_done.json and go_no_go.json)."""
    runs = pd.read_csv(RESULTS / 'weeds_runs/runs_summary.csv')
    pilots = []
    for key, schedule in (('own_b2', 'early stopping (30)'), ('own_b8', 'early stopping (30)')):
        d = model_dir(key)
        if not (d / 'segmentation_metrics.json').exists():
            continue
        seg, best = read_json(d / 'segmentation_metrics.json')['test'], read_json(d / 'best.json')
        done, go = read_json(d / 'train_done.json'), read_json(d / 'go_no_go.json')
        pilots.append({'run': d.name, 'schedule': schedule, 'batch': done.get('batch', MODELS[key]['batch']),
                       'seed': done.get('seed', 0), 'epochs_run': done['epochs_run'], 'selection': 'best val',
                       'epoch': best['epoch'], 'val_miou': best['val_miou'], 'test_miou': seg['miou'],
                       'test_crop': seg['crop'], 'test_weed': seg['weed'],
                       'spray_reduction': go['checks']['C2_spray_reduction']['value'],
                       'deferred': go['checks']['C3_deferred']['value'],
                       'missed_need_rate': go['info']['missed_need_rate'],
                       'false_spray_rate': go['info']['false_spray_rate']})
    return pd.concat([pd.DataFrame(pilots), runs], ignore_index=True)


def training_history(run):
    """Per-epoch validation mIoU of a training run (history.json)."""
    for d in (RESULTS / run, RESULTS / 'weeds_runs' / run):
        if (d / 'history.json').exists():
            return pd.DataFrame(read_json(d / 'history.json'))
    return None


# ---------------------------------------------------------------- explanations

def explanation_cells():
    """SPRAY and DEFER cells with the model-checked counterfactuals, integrated gradients per band and the true weed
    cover (make_counterfactual_figure.load_inputs), and the rescoring steps."""
    return CF.load_inputs(ROOT_DIR)


def no_spray_cells():
    """NO_SPRAY cells: the predicted weed area to add for SPRAY (by the threshold rule; adding weed to an image cannot
    be checked with the model), with the true weed cover."""
    cf = pd.read_csv(RESULTS / 'weeds_counterfactual/counterfactual_cells.csv', keep_default_na=False, na_values=[''])
    dec = pd.read_csv(RESULTS / 'weeds_xai/decisions.csv')
    t = cf.merge(dec[['tile', 'cell', 'w', 'crop']], on=['tile', 'cell'])
    return t[t['decision'] == NO_SPRAY].reset_index(drop=True)


def load_xai_maps():
    """Attribution maps of every decided cell (whole tile, average-pooled): tile, cell, IG and Grad-CAM."""
    with np.load(RESULTS / 'weeds_xai/xai_maps.npz', allow_pickle=False) as z:
        return {'tile': z['tile'].astype(str), 'cell': z['cell'].astype(int),
                'ig': z['integrated_gradients'], 'grad_cam': z['grad_cam']}


def in_cell_share(maps, cells, cell=PRIMARY['cell_size']):
    """Share of each map's positive part that falls inside the cell it explains (a cell is 1/9 of the tile)."""
    pool = TILE // maps.shape[-1]
    c, per_row = cell // pool, TILE // cell
    out = np.full(len(maps), np.nan)
    for i, (m, k) in enumerate(zip(maps, cells)):
        pos = np.clip(m.astype(np.float64), 0, None)
        r, col = divmod(int(k), per_row)
        if pos.sum() > 0:
            out[i] = pos[r * c:(r + 1) * c, col * c:(col + 1) * c].sum() / pos.sum()
    return out


def within_tile_correlation(maps, tiles):
    """Per tile: mean Pearson correlation over all pairs of its cells' maps (1: the same map for every cell)."""
    res = {}
    for t in dict.fromkeys(tiles):
        m = maps[tiles == t].reshape((tiles == t).sum(), -1).astype(np.float64)
        if len(m) > 1:
            res[t] = float(np.nanmean(np.corrcoef(m)[np.triu_indices(len(m), 1)]))
    return pd.Series(res)


def band_shares(xai_cells):
    """Integrated gradients per band as a share of their sum, for SPRAY and DEFER cells; and the top band."""
    t = xai_cells[xai_cells['decision'] != NO_SPRAY]
    ig = t[[f'ig_{b}' for b in BANDS]].to_numpy()
    shares = pd.DataFrame(ig / t['ig_sum'].to_numpy()[:, None], columns=list(BANDS))
    top = pd.Series(np.array(BANDS)[ig.argmax(1)])
    return shares, top
