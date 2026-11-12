"""The pipeline page of the demo (demo/pipeline.py): reading uploaded band files, thresholds and decisions of one tile,
and, with the authors' checkpoint on this machine, the model's output against the Colab run. Upload tests use
synthetic images (they test the reader, not results); the rest needs the dataset zip and the saved results."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('streamlit')
pytest.importorskip('torch')
from PIL import Image  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / 'demo'))
import core as C  # noqa: E402
import pipeline as PL  # noqa: E402

HAVE_RESULTS = C.ZIP.exists() and C.MAIN_MODEL in C.available_models()
needs_results = pytest.mark.skipif(not HAVE_RESULTS, reason='dataset zip or saved probability maps not found')
needs_model = pytest.mark.skipif(not (HAVE_RESULTS and PL.CKPT.exists()), reason=f'checkpoint not found: {PL.CKPT}')


def png(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format='PNG')
    return buf.getvalue()


def band_files(tile='2023-05-30_0001', shape=(600, 600), dtype=np.uint16, bands=C.BANDS):
    rng = np.random.default_rng(0)
    return [(f'{tile}_{b}.png', png(rng.integers(0, 60000, shape).astype(dtype))) for b in bands]


def test_read_band_files_orders_the_bands_and_keeps_16_bit():
    files = band_files()
    x, tile = PL.read_band_files(files[::-1])  # upload order does not matter
    assert x.shape == (5, 600, 600) and x.dtype == np.uint16 and tile == '2023-05-30_0001'
    for i, (_, data) in enumerate(files):
        assert np.array_equal(x[i], np.array(Image.open(io.BytesIO(data))))
    _, tile = PL.read_band_files([(name.lower(), data) for name, data in files])  # case of the suffix
    assert tile == '2023-05-30_0001'
    # a file downloaded twice: the browser names it '<tile>_R (1).png'
    x2, tile = PL.read_band_files([(name.replace('.png', ' (1).png'), data) for name, data in files])
    assert tile == '2023-05-30_0001' and np.array_equal(x2, x)


def zipped(files, folder=''):
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name, data in files:
            zf.writestr(folder + name, data)
    return buf.getvalue()


def test_read_band_files_takes_a_zip_of_the_five_files():
    files = band_files()
    x, tile = PL.read_band_files(files)
    xz, tile_z = PL.read_band_files([('mau.zip', zipped(files, 'thu_muc/'))])  # also inside a folder of the zip
    assert tile_z == tile and np.array_equal(xz, x)
    with pytest.raises(ValueError, match='zip'):
        PL.read_band_files([('hong.zip', b'not a zip')])
    with pytest.raises(ValueError, match='thiếu dải'):
        PL.read_band_files([('mau.zip', zipped(files[:3]))])


@pytest.mark.parametrize('files, message', [
    (band_files(bands=C.BANDS[:4]), 'thiếu dải: RE'),
    (band_files() + band_files()[:1], 'hai lần'),
    (band_files(dtype=np.uint8), '16 bit'),
    ([('anh.png', png(np.zeros((600, 600), np.uint16)))], 'Không nhận ra dải'),
    (band_files()[:4] + band_files(shape=(500, 600))[4:], 'cùng kích thước'),
    ([('t_R.png', b'not an image')] + band_files('t')[1:], 'Không đọc được'),
])
def test_read_band_files_refuses_bad_uploads(files, message):
    with pytest.raises(ValueError, match=message):
        PL.read_band_files(files)


def test_extract_checkpoint_refuses_an_incomplete_download(tmp_path):
    partial = tmp_path / 'ckpts.zip'
    partial.write_bytes(b'PK' + b'\0' * 1000)
    with pytest.raises(ValueError, match='chưa tải đủ'):
        PL.extract_checkpoint(partial, tmp_path / 'm.pth')
    assert not (tmp_path / 'm.pth').exists()


def test_fit_tile_crops_larger_images_and_refuses_smaller():
    x, cropped = PL.fit_tile(np.zeros((5, 600, 600), np.uint16))
    assert x.shape == (5, 600, 600) and not cropped
    x, cropped = PL.fit_tile(np.arange(5 * 700 * 650, dtype=np.uint16).reshape(5, 700, 650) % 1000)
    assert x.shape == (5, 600, 600) and cropped
    with pytest.raises(ValueError):
        PL.fit_tile(np.zeros((5, 599, 600), np.uint16))


@needs_results
def test_sample_download_uploads_back_as_the_same_tile():
    """The page's flow for trying the upload: download the five files of a tile, upload them again."""
    import zipfile
    ids = C.dataset_ids()
    tile = ids['test'][0]
    with zipfile.ZipFile(io.BytesIO(PL.sample_files(tile))) as zf:
        files = [(name, zf.read(name)) for name in zf.namelist()]
    assert sorted(n for n, _ in files) == sorted(f'{tile}_{b}.png' for b in C.BANDS)
    x, tile_id = PL.read_band_files(files)
    x, cropped = PL.fit_tile(x)
    assert tile_id == tile and not cropped
    assert np.array_equal(x, C.read_bands(tile)[0])
    assert PL.match_dataset(x, tile_id, ids)[0] == 'test'


@needs_results
def test_match_dataset_needs_identical_pixels():
    ids = C.dataset_ids()
    tile = ids['test'][0]
    x, mask = C.read_bands(tile)
    split, m = PL.match_dataset(x, tile, ids)
    assert split == 'test' and np.array_equal(m, mask)
    changed = x.copy()
    changed[0, 0, 0] ^= 1
    assert PL.match_dataset(changed, tile, ids) == (None, None)
    assert PL.match_dataset(x, 'not_a_tile', ids) == (None, None)


@needs_results
def test_saved_maps_through_the_page_logic_give_the_reported_decisions():
    """With the Colab probability map of a test tile, thresholds and decisions are those of results/weeds_xai."""
    ids = C.dataset_ids()
    tiles, _, p_weed = C.load_probs(C.MAIN_MODEL, 'test')
    test_cells = C.cell_table(tiles, C.to_3class(C.read_masks(tiles)), p_weed, PL.CELL)
    fold_of = C.locations(ids['test'])[1]
    saved = pd.read_csv(C.RESULTS / 'weeds_xai/decisions.csv')
    for tile in saved['tile'].unique()[:6]:
        i = list(tiles).index(tile)
        thr = PL.thresholds(tile, 'test', test_cells, fold_of)
        assert thr['how'] == 'crossfit'
        cells = PL.decide_tile(p_weed[i].astype(np.float32), C.read_bands(tile)[1], tile, thr)
        s = saved[saved['tile'] == tile].sort_values('cell')
        assert np.array_equal(cells['decision'], s['decision'])
        assert np.allclose(cells['lam_lo'], s['lam_lo']) and np.allclose(cells['lam_hi'], s['lam_hi'])
        assert np.allclose(cells['w'], s['w'])
        est = PL.explain_estimate(p_weed[i].astype(np.float32), cells)
        assert len(est) == len(cells)
    # a tile outside the calibration cells (val) gets the thresholds of every pooled test cell
    thr = PL.thresholds(ids['val'][0], 'val', test_cells, fold_of)
    assert thr['how'] == 'all' and thr['n_need'] + thr['n_no_need'] == 216


@needs_results
def test_estimate_steps_follow_the_counterfactual_table():
    tiles, _, p_weed = C.load_probs(C.MAIN_MODEL, 'test')
    cf = pd.read_csv(C.RESULTS / 'weeds_counterfactual/counterfactual_cells.csv', keep_default_na=False,
                     na_values=[''])
    r = cf[cf['decision'] == C.SPRAY].iloc[0]
    steps = PL.estimate_steps(p_weed[list(tiles).index(r['tile'])].astype(np.float32), r['cell'], r['lam_lo'],
                              r['lam_hi'])
    assert steps['decision_model'].iloc[0] == C.SPRAY
    assert (steps['decision_model'] != C.SPRAY).idxmax() == r['k_not_spray']
    assert (steps['decision_model'] == C.NO_SPRAY).idxmax() == r['k_no_spray']


@needs_model
def test_model_on_cpu_reproduces_the_colab_map():
    """The authors' model run here (float32, CPU) gives the Colab map (float16, GPU) up to rounding, and the same
    decisions on that tile. About 10 s on a laptop CPU."""
    tiles, _, p_weed = C.load_probs(C.MAIN_MODEL, 'test')
    saved = pd.read_csv(C.RESULTS / 'weeds_xai/decisions.csv')
    tile = saved['tile'].iloc[0]
    x, mask = C.read_bands(tile)
    probs, _ = PL.segment(PL.load_model(), x)
    assert probs.shape == (3, 600, 600) and np.allclose(probs.sum(0), 1, atol=1e-4)
    assert np.abs(probs[2] - p_weed[list(tiles).index(tile)].astype(np.float32)).max() < 0.02
    ids = C.dataset_ids()
    test_cells = C.cell_table(tiles, C.to_3class(C.read_masks(tiles)), p_weed, PL.CELL)
    cells = PL.decide_tile(probs[2], mask, tile, PL.thresholds(tile, 'test', test_cells, C.locations(ids['test'])[1]))
    s = saved[saved['tile'] == tile].sort_values('cell')
    assert np.abs(cells['s'].to_numpy() - s['s'].to_numpy()).max() < 1e-3
    assert np.array_equal(cells['decision'], s['decision'])
