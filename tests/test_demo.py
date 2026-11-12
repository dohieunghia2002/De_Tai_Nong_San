"""Progress demo (demo/): its recomputed decisions must equal the saved results, and every page must run.
Needs the dataset zip and the unpacked Colab results (probs_*.npz, xai_maps.npz are not in git); skipped otherwise.
The page test drives the real Streamlit app headless (about 30 s, longer on the very first run), so it runs only with
WEEDS_SLOW_TESTS=1."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('streamlit')
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / 'demo'))
import core as C  # noqa: E402

HAVE_RESULTS = C.ZIP.exists() and C.MAIN_MODEL in C.available_models()
needs_results = pytest.mark.skipif(not HAVE_RESULTS, reason='dataset zip or saved probability maps not found')


def tables(key, cell):
    ids = C.dataset_ids()
    out = {}
    for split in ('val', 'test'):
        tiles, _, p_weed = C.load_probs(key, split)
        C.check_order(tiles, ids[split])
        out[split] = C.cell_table(tiles, C.to_3class(C.read_masks(tiles)), p_weed, cell)
    return out, C.locations(ids['test'])[1]


@needs_results
@pytest.mark.parametrize('cell, tau, scope, calibration', [
    (200, 0.01, 'pre-herbicide pooled', C.CAL_TEST_2FOLD),   # the primary setting
    (100, 0.005, 'pre-herbicide pooled', C.CAL_TEST_2FOLD),
    (200, 0.02, '2023-05-30', C.CAL_TEST_2FOLD),
    (200, 0.01, 'pre-herbicide pooled', C.CAL_VAL),
])
def test_decisions_equal_saved_results(cell, tau, scope, calibration):
    t, fold_of = tables(C.MAIN_MODEL, cell)
    cells, metrics, parts = C.decide(t['test'], t['val'], fold_of, tau, C.ALPHA, C.BETA, C.SCOPES[scope], calibration)
    diff = C.saved_difference(metrics, C.saved_metrics(C.MAIN_MODEL), cell, tau, C.ALPHA, C.BETA, scope, calibration)
    assert diff is not None and diff < 1e-9
    m = metrics.set_index('policy')
    for name in C.POLICIES:  # the counts the pages show agree with the rates of decision_metrics.csv
        r = m.loc[name]
        assert r['n_deferred'] == round(r['deferred'] * r['n_cells'])
        if r['n_need']:
            assert r['n_missed'] == round(r['missed_need_rate'] * r['n_need'])
        if r['n_cells'] > r['n_need']:
            assert r['n_false_spray'] == round(r['false_spray_rate'] * (r['n_cells'] - r['n_need']))
    assert len(parts) == (2 if calibration == C.CAL_TEST_2FOLD else 1)
    assert set(cells['need']) <= {True, False} and (cells['need'] == (cells['w'] >= tau)).all()


@needs_results
def test_primary_decisions_equal_the_explained_ones():
    """The cells the explanation pages describe (results/weeds_xai/decisions.csv) are the primary decisions."""
    t, fold_of = tables(C.MAIN_MODEL, C.PRIMARY['cell_size'])
    s = C.PRIMARY_SETTING
    cells = C.decide(t['test'], t['val'], fold_of, s['tau'], s['alpha'], s['beta'], C.SCOPES[s['scope']],
                     s['calibration'])[0]
    saved = pd.read_csv(C.RESULTS / 'weeds_xai/decisions.csv')
    merged = cells.merge(saved, on=['tile', 'cell'], suffixes=('', '_saved'))
    assert len(merged) == len(cells) == len(saved)
    assert (merged['conformal_3way'] == merged['decision']).all()
    assert np.allclose(merged['lam_lo'], merged['lam_lo_saved']) and np.allclose(merged['lam_hi'], merged['lam_hi_saved'])


def test_saved_difference_is_none_off_the_saved_grid():
    saved = pd.DataFrame([{'cell_size': 200, 'scope': 'x', 'calibration': 'c', 'policy': p, 'tau': 0.01,
                           **{k: 0.5 for k in C.METRICS}} for p in C.POLICIES])
    metrics = saved.drop(columns=['cell_size', 'scope', 'calibration'])
    assert C.saved_difference(metrics, saved, 200, 0.01, C.ALPHA, C.BETA, 'x', 'c') == 0.0
    assert C.saved_difference(metrics, saved, 200, 0.015, C.ALPHA, C.BETA, 'x', 'c') is None   # tau not saved
    assert C.saved_difference(metrics, saved, 200, 0.01, 0.05, C.BETA, 'x', 'c') is None        # alpha not saved
    assert C.saved_difference(metrics, None, 200, 0.01, C.ALPHA, C.BETA, 'x', 'c') is None
    changed = metrics.copy()
    changed.loc[0, 'deferred'] = 0.25
    assert math.isclose(C.saved_difference(changed, saved, 200, 0.01, C.ALPHA, C.BETA, 'x', 'c'), 0.25)


def test_in_cell_share_and_within_tile_correlation():
    m = np.zeros((2, 150, 150), dtype=np.float32)
    m[0, 50:100, 50:100] = 1          # all inside cell 4 (the centre)
    m[1] = 1                          # spread evenly over the tile
    share = C.in_cell_share(m, np.array([4, 4]))
    assert share[0] == 1 and math.isclose(share[1], 1 / 9)
    rng = np.random.default_rng(0)
    same = np.repeat(rng.random((1, 150, 150)), 3, axis=0)
    corr = C.within_tile_correlation(np.concatenate([same, rng.random((3, 150, 150))]),
                                     np.array(['a'] * 3 + ['b'] * 3))
    assert math.isclose(corr['a'], 1) and abs(corr['b']) < 0.05


@needs_results
def test_every_page_runs():
    if os.environ.get('WEEDS_SLOW_TESTS') != '1':
        pytest.skip('drives the Streamlit app: set WEEDS_SLOW_TESTS=1')
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT_DIR / 'demo/app.py'), default_timeout=300)
    pages = ['overview', 'dataset', 'segmentation', 'decision', 'explanation', 'faithfulness']
    ckpts = C.IMAGES / 'ckpts'
    if (ckpts / 'dlv3p_msi_3.pth').exists() or (ckpts / 'ckpts.zip').exists():  # the pipeline page runs the model
        pages.insert(0, 'pipeline')
    for page in pages:
        at.switch_page(f'views/{page}.py')
        at.run()
        assert not at.exception, (page, [e.message for e in at.exception])
        assert not at.error, (page, [e.value for e in at.error])
    at.switch_page('views/decision.py')
    at.session_state['dec_tau'] = 0.015          # a setting that is not saved: the page must say so
    at.run()
    assert not at.exception and any('Đang khám phá' in w.value for w in at.warning)
