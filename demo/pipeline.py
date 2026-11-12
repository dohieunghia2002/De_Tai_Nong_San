"""Run the whole system on one tile, on this machine's CPU: segmentation with the authors' published model
(dlv3p_msi_3, 5 bands, 3 classes), the spraying decision of every cell, and the explanations of one cell.
No Streamlit here, so tests can import it.

Every step reuses the project's code: the model loader of src.weeds.authors, the decision rule of src.weeds.decision,
the counterfactual estimate of src.weeds.counterfactual, the model check of src.weeds.counterfactual_check and the
integrated gradients / Grad-CAM of src.weeds.explain. The reported results were computed on Colab with float16
autocast; here the model runs in float32 on CPU, so scores can differ in the last digits.

Thresholds: a test tile of the pooled pre-herbicide dates gets the thresholds of the other location part (the
cross-fit of the report, so its decisions are the reported ones); any other tile gets thresholds calibrated on all
those test cells, with which it shares no cell.
"""
import io
import re
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import core as C
from src.weeds.authors import CKPT_ZIP_BYTES, CKPT_URL  # noqa: F401  (CKPT_URL: shown on the page)
from src.weeds.counterfactual import counterfactual_table, ranked_patches, removal_path
from src.weeds.decision import cell_table, decide_conformal, lower_threshold, upper_threshold

MODEL_NAME = 'dlv3p_msi_3'
CKPT_DIR = C.IMAGES / 'ckpts'           # data/ is not in git
CKPT_ZIP = CKPT_DIR / 'ckpts.zip'
CKPT = CKPT_DIR / f'{MODEL_NAME}.pth'
CELL = C.PRIMARY['cell_size']
TAU = C.PRIMARY['tau']
IG_STEPS = 32                           # as the Colab run (results/weeds_xai/xai_config.json)
MAX_PIXELS = 4000 * 4000                # refuse larger images before decoding them
MAX_ZIP_BYTES = 200 * 2 ** 20           # uncompressed size of an uploaded zip, checked before reading it
# '<tile>_<band>.png', also with the ' (1)' a browser appends to a file downloaded twice
BAND_FILE = re.compile(r'^(?P<tile>.*?)[_-](?P<band>R|G|B|NIR|RE)(?: \(\d+\))?\.png$', re.IGNORECASE)


# ---------------------------------------------------------------- model

def extract_checkpoint(zip_path=CKPT_ZIP, out=CKPT):
    """Copy the one checkpoint the demo needs out of the authors' ckpts.zip (checked by its size, as the notebooks
    do). Only that member is read, streamed to a fixed path. Raises ValueError with a message for the page when the
    zip is incomplete (an interrupted download) or damaged."""
    size = Path(zip_path).stat().st_size
    if size != CKPT_ZIP_BYTES:
        raise ValueError(f'`ckpts.zip` chưa tải đủ hoặc bị hỏng: có {size / 1e6:.0f} MB, cần {CKPT_ZIP_BYTES / 1e6:.0f} '
                         f'MB ({CKPT_ZIP_BYTES} byte). Tải lại (hoặc tải tiếp) file này từ {CKPT_URL}.')
    tmp = Path(out).with_suffix('.part')
    try:
        with zipfile.ZipFile(zip_path) as zf, zf.open(f'ckpts/{MODEL_NAME}.pth') as src, open(tmp, 'wb') as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    except (zipfile.BadZipFile, KeyError) as e:
        tmp.unlink(missing_ok=True)
        raise ValueError(f'Không đọc được `{MODEL_NAME}.pth` trong `ckpts.zip` ({e}). Tải lại file từ {CKPT_URL}.')
    tmp.replace(out)
    return out


def load_model(path=CKPT):
    """The authors' model in evaluation mode on CPU, loaded with weights_only (a third-party file)."""
    import torch
    from src.weeds.authors import load_checkpoint
    model = load_checkpoint(path, len(C.BANDS), 3, torch.device('cpu'))
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def to_tensor(x):
    """uint16 tile (bands, H, W) -> float32 tensor scaled to [0, 1], as the model was trained."""
    import torch
    return torch.from_numpy(x.astype(np.float32) / 65535.0)


def segment(model, x):
    """Class probabilities (3, H, W) float32 (background, crop, weed) of one uint16 tile, and the seconds taken."""
    import torch
    t0 = time.perf_counter()
    with torch.no_grad():
        p = torch.softmax(model(to_tensor(x)[None]).float(), dim=1)[0].numpy()
    return p, time.perf_counter() - t0


def soil_baseline():
    """Bare-soil value per band in [0, 1] (results/weeds_xai/soil_baseline.json, from the training split)."""
    return np.array(C.read_json(C.RESULTS / 'weeds_xai/soil_baseline.json')['values'], dtype=np.float32)


# ---------------------------------------------------------------- input

def unpack_zips(files):
    """Replace every uploaded .zip by the .png files inside it (read in memory, never extracted to disk; names reduced
    to their last part). The total uncompressed size is checked before anything is read."""
    out = []
    for name, data in files:
        if not name.lower().endswith('.zip'):
            out.append((name, data))
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                # only band files: a zip may also hold a label image, or the '__MACOSX/._*' metadata macOS adds
                members = [i for i in zf.infolist() if not i.is_dir() and '__MACOSX' not in i.filename
                           and not Path(i.filename).name.startswith('._') and BAND_FILE.match(Path(i.filename).name)]
                if sum(i.file_size for i in members) > MAX_ZIP_BYTES:
                    raise ValueError(f'File "{name}" giải nén ra quá lớn.')
                out += [(Path(i.filename).name, zf.read(i)) for i in members]
        except zipfile.BadZipFile:
            raise ValueError(f'Không mở được file "{name}" như một file zip.') from None
    return out


def read_band_files(files):
    """Five 16-bit PNG files of one tile, named like WeedsGalore's (<tile>_R.png ... <tile>_RE.png), uploaded as they
    are or in a zip. files: (name, bytes) pairs. Returns the uint16 bands (5, H, W) in BANDS order and the tile id the
    names share (None if they differ). Raises ValueError with a message for the page (Vietnamese, the audience)."""
    bands, tiles = {}, set()
    for name, data in unpack_zips(files):
        m = BAND_FILE.match(Path(name).name)
        if not m:
            raise ValueError(f'Không nhận ra dải phổ của file "{name}": tên file phải kết thúc bằng _R, _G, _B, _NIR '
                             'hoặc _RE và có đuôi .png, như trong WeedsGalore.')
        band = m['band'].upper()
        if band in bands:
            raise ValueError(f'Dải {band} được upload hai lần.')
        try:
            img = Image.open(io.BytesIO(data))
            if img.width * img.height > MAX_PIXELS:
                raise ValueError(f'File "{name}" quá lớn ({img.width} × {img.height} điểm ảnh).')
            a = np.array(img)
        except (OSError, SyntaxError, Image.DecompressionBombError) as e:  # not an image, or a damaged one
            raise ValueError(f'Không đọc được file "{name}" như một ảnh PNG ({type(e).__name__}).') from None
        if a.ndim != 2:
            raise ValueError(f'File "{name}" không phải ảnh một kênh (mỗi file chỉ chứa một dải phổ).')
        if a.dtype != np.uint16:
            raise ValueError(f'File "{name}" là ảnh {a.dtype}, cần ảnh 16 bit như dữ liệu gốc: mô hình được huấn luyện '
                             'trên ảnh 16 bit.')
        bands[band] = a
        tiles.add(m['tile'])
    missing = [b for b in C.BANDS if b not in bands]
    if missing:
        raise ValueError('Còn thiếu dải: ' + ', '.join(missing) + '.')
    if len({a.shape for a in bands.values()}) != 1:
        raise ValueError('Năm dải không cùng kích thước.')
    return np.stack([bands[b] for b in C.BANDS]), (tiles.pop() if len(tiles) == 1 else None)


def sample_files(tile_id, zip_path=C.ZIP):
    """The five band PNGs of a WeedsGalore tile, as stored in the dataset zip, packed in a zip to try the upload."""
    buf = io.BytesIO()
    with zipfile.ZipFile(zip_path) as zf, zipfile.ZipFile(buf, 'w') as out:
        for b in C.BANDS:
            out.writestr(f'{tile_id}_{b}.png', zf.read(f'{C.ROOT}/{C.tile_date(tile_id)}/images/{tile_id}_{b}.png'))
    return buf.getvalue()


def fit_tile(x):
    """The model and the grid work on WeedsGalore tiles (600 x 600 px = 1.5 m x 1.5 m): a larger image is cropped to
    its top-left tile. Returns the tile and whether it was cropped."""
    h, w = x.shape[1:]
    if h < C.TILE or w < C.TILE:
        raise ValueError(f'Ảnh {w} × {h} điểm ảnh nhỏ hơn một ô ảnh WeedsGalore ({C.TILE} × {C.TILE}).')
    return x[:, :C.TILE, :C.TILE], (h, w) != (C.TILE, C.TILE)


def match_dataset(x, tile_id, ids, read=C.read_bands):
    """(split, semantic mask) when the uploaded bands are exactly a WeedsGalore tile, else (None, None)."""
    split = next((s for s, tiles in ids.items() if tile_id in tiles), None)
    if split is None:
        return None, None
    bands, mask = read(tile_id)
    return (split, mask) if bands.shape == x.shape and np.array_equal(bands, x) else (None, None)


# ---------------------------------------------------------------- decisions

def thresholds(tile_id, split, test_cells, fold_of, tau=TAU, alpha=C.ALPHA, beta=C.BETA):
    """Conformal thresholds for one tile, from the saved test cells of the primary scope (src.weeds.decision).
    Returns lam_lo, lam_hi and how they were calibrated: 'crossfit' (a test tile: the other location part) or 'all'
    (every test cell of those dates)."""
    pool = test_cells[test_cells['date'].isin(C.SCOPES[C.PRIMARY['scope']])]
    if split == 'test' and tile_id in set(pool['tile']):
        cal, how = pool[pool['tile'].map(fold_of) != fold_of[tile_id]], 'crossfit'
    else:
        cal, how = pool, 'all'
    need = (cal['w'] >= tau).to_numpy()
    s = cal['s'].to_numpy()
    return {'lam_lo': lower_threshold(s[need], alpha), 'lam_hi': upper_threshold(s[~need], beta), 'how': how,
            'n_need': int(need.sum()), 'n_no_need': int((~need).sum()), 'locations': int(cal['tile'].nunique())}


def decide_tile(p_weed, mask, tile_id, thr, cell=CELL, tau=TAU):
    """One row per cell (row-major): score s, decision and, when the label is known, true weed cover and need."""
    y3 = C.to_3class(mask) if mask is not None else np.zeros(p_weed.shape, np.uint8)
    t = cell_table([tile_id or 'upload'], y3[None], p_weed[None], cell)
    t['cell'] = np.arange(len(t))
    t['lam_lo'], t['lam_hi'] = thr['lam_lo'], thr['lam_hi']
    t['decision'] = decide_conformal(t['s'].to_numpy(), thr['lam_lo'], thr['lam_hi'])
    if mask is None:
        t[['w', 'crop']] = np.nan
        t['need'] = pd.NA
    else:
        t['need'] = t['w'] >= tau
    return t


def explain_estimate(p_weed, cells):
    """The fast counterfactual of every cell (src.weeds.counterfactual, from the P(weed) map alone)."""
    tile = cells['tile'].iloc[0]
    return counterfactual_table([tile], p_weed[None], cells[['tile', 'date', 's', 'lam_lo', 'lam_hi', 'decision',
                                                            'cell']], CELL)


def estimate_steps(p_weed, cell_idx, lam_lo, lam_hi, cell=CELL):
    """Estimated score and decision after removing 0, 1, 2, ... ranked patches of one cell (no model run), in the
    columns of results/weeds_cfcheck/cfcheck_steps.csv so the figure code draws it."""
    r, c = divmod(int(cell_idx), p_weed.shape[1] // cell)
    _, masses = ranked_patches(p_weed[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell], C.PATCH)
    est = removal_path(masses.sum() / cell ** 2, masses, cell)
    dec = decide_conformal(est, lam_lo, lam_hi)
    return pd.DataFrame({'k': np.arange(len(est)), 's_model': est, 's_estimate': est, 'decision_model': dec,
                         'decision_estimate': dec})


CHECK_BATCH = 2  # images per forward call of the check: CPU time grows with the images, so this only saves overhead


def model_check(model, x, row, batch=CHECK_BATCH):
    """Rescore the tile with the model after removing the cell's ranked patches one by one (the Colab check of
    src.weeds.counterfactual_check, here on CPU). The scan stops as soon as the model gives NO_SPRAY and both estimates
    are reached (extra=0: the Colab run scanned 2 patches more, for its statistics only). row: a decide_tile row.
    Returns (steps, cell result, forward passes, seconds)."""
    import torch
    from src.weeds.counterfactual_check import verify_tile
    t0 = time.perf_counter()
    cells = pd.DataFrame([{'cell': int(row['cell']), 'decision': int(row['decision']), 'lam_lo': row['lam_lo'],
                           'lam_hi': row['lam_hi']}])
    steps, result, passes = verify_tile(model, to_tensor(x), torch.from_numpy(soil_baseline()), cells, CELL,
                                        batch=batch, extra=0, amp=False)
    return pd.DataFrame(steps), result[0], passes, time.perf_counter() - t0


def planned_passes(est_row, batch=CHECK_BATCH):
    """Images the model check scores if the model agrees with the estimate: the map, then the estimated k rounded up
    to whole batches. The model often needs a few patches more than the estimate, so this is a lower bound."""
    ks = [k for k in (est_row['k_not_spray'], est_row['k_no_spray']) if np.isfinite(k)]
    n = (CELL // C.PATCH) ** 2
    target = min(n, int(max(ks)) if ks else n)
    return 1 + min(n, -(-target // batch) * batch)


def attributions(model, x, cell_idx):
    """Integrated gradients (5, H, W) from bare soil, IG_STEPS steps, one image at a time (little memory), the score
    gap they should add up to, and the Grad-CAM map (H, W) of one cell (src.weeds.explain). Also the seconds taken."""
    import torch
    from src.weeds.explain import grad_cams, integrated_gradients
    t0 = time.perf_counter()
    xt, rc = to_tensor(x), divmod(int(cell_idx), C.TILE // CELL)
    ig, gap = integrated_gradients(model, xt, torch.from_numpy(soil_baseline()), rc, CELL, steps=IG_STEPS, batch=1,
                                   amp=False)
    cam = grad_cams(model, xt, [rc], CELL, amp=False)[0]
    return ig.numpy(), float(gap), cam.numpy(), time.perf_counter() - t0
