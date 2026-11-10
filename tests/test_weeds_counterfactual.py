"""Counterfactual explanations of the decisions: hand-computed cases on a small tile."""
import math

import numpy as np
import pandas as pd
import pytest

from src.weeds.counterfactual import (compare_with_deletion, counterfactual_table, ranked_patches, summarize,
                                      summarize_comparison)
from src.weeds.decision import DEFER, NO_SPRAY, SPRAY

TILE = '2023-05-30_0001'
CELL, PATCH = 4, 2  # 8 x 8 tile: 2 x 2 cells of 2 x 2 patches; the patch grid of the tile is 4 x 4


def small_case():
    p = np.zeros((1, 8, 8), dtype=np.float16)
    p[0, 0:2, 0:2] = 1.0     # cell 0, patch (0, 0): mass 4
    p[0, 0:2, 2:4] = 0.5     # cell 0, patch (0, 1): mass 2
    p[0, 2:4, 0:2] = 0.25    # cell 0, patch (1, 0): mass 1        -> s = 7 / 16
    p[0, 6:8, 4:6] = 0.5     # cell 3, patch (1, 0): mass 2        -> s = 2 / 16
    p[0, 4:6, 0:2] = 1.0     # cell 2, patch (0, 0): mass 4        -> s = 4 / 16
    lam = {0: (0.3, 0.1), 1: (0.3, 0.1), 2: (-np.inf, 0.1), 3: (0.2, 0.1)}  # (lam_lo, lam_hi); cell 3 crossed
    s = {0: 7 / 16, 1: 0.0, 2: 4 / 16, 3: 2 / 16}
    dec = {0: SPRAY, 1: NO_SPRAY, 2: SPRAY, 3: DEFER}
    t = pd.DataFrame({'tile': TILE, 'date': TILE[:10], 'cell': range(4), 's': [s[k] for k in range(4)],
                      'lam_lo': [lam[k][0] for k in range(4)], 'lam_hi': [lam[k][1] for k in range(4)],
                      'decision': [dec[k] for k in range(4)]})
    return p, t


def test_ranked_patches():
    order, masses = ranked_patches(small_case()[0][0, :4, :4], PATCH)
    assert list(order) == [0, 1, 2, 3]
    assert masses.tolist() == [4.0, 2.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        ranked_patches(np.zeros((4, 6)), PATCH)


def test_counterfactual_table_hand_computed():
    p, t = small_case()
    out = counterfactual_table([TILE], p, t, CELL, PATCH, gsd_cm=1.0).set_index('cell')
    # SPRAY, s = 7/16: removing patch (0,0) gives 3/16, below lam_lo but above lam_hi (DEFER); then 1/16 (NO_SPRAY)
    assert out.loc[0, 'k_not_spray'] == 1 and out.loc[0, 'k_no_spray'] == 2
    assert out.loc[0, 'removed_cm2_not_spray'] == pytest.approx(4.0)
    assert out.loc[0, 'removed_cm2_no_spray'] == pytest.approx(6.0)
    assert out.loc[0, 'add_to_spray_cm2'] == 0.0
    assert out.loc[0, 'remove_to_no_spray_cm2'] == pytest.approx((7 / 16 - 0.1) * 16)
    assert out.loc[0, 'patches'] == '0 1'
    assert out.loc[0, 'weed_cm2'] == pytest.approx(7.0)
    # NO_SPRAY with nothing predicted: only adding weed changes it
    assert out.loc[1, 'k_not_spray'] == 0 and out.loc[1, 'k_no_spray'] == 0
    assert out.loc[1, 'add_to_spray_cm2'] == pytest.approx(0.3 * 16)
    assert out.loc[1, 'patches'] == ''
    # lam_lo = -inf (too few calibration cells): 'need' is never ruled out, so NO_SPRAY cannot be reached
    assert out.loc[2, 'k_not_spray'] == 1 and math.isnan(out.loc[2, 'k_no_spray'])
    assert math.isnan(out.loc[2, 'removed_cm2_no_spray']) and math.isinf(out.loc[2, 'remove_to_no_spray_cm2'])
    assert out.loc[2, 'patches'] == '8'      # cell (1, 0), patch (0, 0) -> tile patch row 2, column 0
    # DEFER between crossed thresholds (lam_hi 0.1 < s = 0.125 < lam_lo 0.2)
    assert out.loc[3, 'k_not_spray'] == 0 and out.loc[3, 'k_no_spray'] == 1
    assert out.loc[3, 'add_to_spray_cm2'] == pytest.approx((0.2 - 0.125) * 16)
    assert out.loc[3, 'remove_to_no_spray_cm2'] == pytest.approx((0.125 - 0.1) * 16)
    assert out.loc[3, 'patches'] == '14'     # cell (1, 1), patch (1, 0) -> tile patch row 3, column 2
    s = summarize(out.reset_index())
    assert s['decisions'] == {'spray': 2, 'no_spray': 1, 'defer': 1}
    assert s['spray']['k_no_spray']['not_reachable'] == 1 and s['spray']['k_not_spray_le_1'] == 2


def test_default_area_unit():
    p, t = small_case()
    out = counterfactual_table([TILE], p, t, CELL, PATCH).set_index('cell')  # 2.5 mm pixels: 0.0625 cm^2 each
    assert out.loc[0, 'weed_cm2'] == pytest.approx(7 * 0.0625)


def test_cell_order_without_cell_column():
    p, t = small_case()
    a = counterfactual_table([TILE], p, t, CELL, PATCH)
    b = counterfactual_table([TILE], p, t.drop(columns='cell'), CELL, PATCH)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize('column, value', [('s', 7 / 16 + 1e-3), ('decision', DEFER)])
def test_mismatch_with_the_decision_table_is_an_error(column, value):
    p, t = small_case()
    t.loc[0, column] = value
    with pytest.raises(ValueError):
        counterfactual_table([TILE], p, t, CELL, PATCH)


def test_missing_tile_is_an_error():
    p, t = small_case()
    with pytest.raises(ValueError):
        counterfactual_table(['2023-05-30_0002'], p, t, CELL, PATCH)


def test_compare_with_deletion():
    p, t = small_case()
    table = counterfactual_table([TILE], p, t, CELL, PATCH, gsd_cm=1.0)
    # 16 patches per tile: fraction 1/16 removes 1 patch, 2/16 removes 2
    deletion = pd.DataFrame({'tile': TILE, 'cell': [0, 0, 0], 'method': ['predicted_weed', 'predicted_weed',
                                                                         'random_cell'],
                             'fraction': [1 / 16, 2 / 16, 1 / 16], 'draw': 0, 's_after': [0.35, 1 / 16, 0.0]})
    cmp = compare_with_deletion(table, [TILE], p, deletion, CELL, PATCH, gsd_cm=1.0)
    assert len(cmp) == 2                                           # random controls are ignored
    assert cmp['s_after_estimate'].tolist() == pytest.approx([3 / 16, 1 / 16])
    assert cmp['decision_estimate'].tolist() == [DEFER, NO_SPRAY]
    assert cmp['decision_model'].tolist() == [SPRAY, NO_SPRAY]    # the model kept more weed after one patch
    assert cmp['error_cm2'].tolist() == pytest.approx([(3 / 16 - 0.35) * 16, 0.0])
    s = summarize_comparison(cmp)
    assert s['decision_agreement'] == 0.5 and s['disagreements_estimate_less_spray'] == 1
    assert s['rows_decision_changed'] == 2 and s['decision_agreement_where_changed'] == 0.5
    assert s['spray_end_agreement'] == 0.5 and s['spray_rows'] == 2
