"""WeedsGalore reader against the real dataset zip. The zip is not in the repo (337 MB): set WEEDSGALORE_ZIP to its
path, otherwise these tests are skipped."""
import os
from pathlib import Path

import numpy as np
import pytest

from src.weeds.data import BANDS, DATES, EXPECTED_TILES, load_split, to_3class
from src.weeds.decision import cell_table, evaluate_policies

ZIP = Path(os.environ.get('WEEDSGALORE_ZIP', 'data/weedsgalore/weedsgalore-dataset.zip'))
pytestmark = pytest.mark.skipif(not ZIP.exists(), reason=f'dataset zip not found: {ZIP}')


@pytest.fixture(scope='module')
def val():
    return load_split(ZIP, 'val')


def test_val_split_shapes_and_values(val):
    assert len(val['ids']) == EXPECTED_TILES['val']
    assert val['x'].shape == (26, len(BANDS), 600, 600) and val['x'].dtype == np.uint16
    assert set(np.unique(val['y'])) <= set(range(6))
    assert {t[:10] for t in val['ids']} == set(DATES)


def test_three_class_merge(val):
    y3 = to_3class(val['y'])
    assert set(np.unique(y3)) <= {0, 1, 2}
    assert (y3 == 2).sum() == (val['y'] >= 2).sum()
    assert (y3 == 1).sum() == (val['y'] == 1).sum()


def test_perfect_probabilities_end_to_end(val):
    # labels used as probabilities: the cell score must equal the true weed fraction and nothing may be missed
    y3 = to_3class(val['y'])
    t = cell_table(val['ids'], y3, (y3 == 2).astype(np.float32), 200)
    assert len(t) == 26 * 9
    np.testing.assert_allclose(t['s'], t['w'], atol=1e-6)
    rows = {r['policy']: r for r in evaluate_policies(t, t, tau=0.01, alpha=0.1, beta=0.1)}
    assert rows['conformal_3way']['missed_need_rate'] == 0
    assert rows['conformal_3way']['false_spray_rate'] == 0
