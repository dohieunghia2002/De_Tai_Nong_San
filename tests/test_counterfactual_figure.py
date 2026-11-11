"""Logic of the counterfactual figure (scripts/make_counterfactual_figure.py): the fixed rules that pick the examples,
the thresholds it names, and the sentences. Synthetic rows: they test the rules, never results."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('PIL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import make_counterfactual_figure as F  # noqa: E402
from src.weeds.decision import DEFER, NO_SPRAY, SPRAY  # noqa: E402


def row(tile, cell, date, dec, s, k_not_e, k_not_m, k_no_e, k_no_m, reverts=False):
    return {'tile': tile, 'cell': cell, 'date': date, 'decision': dec, 'decision_rescored': dec, 's_model': s,
            'k_not_spray_estimate': k_not_e, 'k_not_spray_model': k_not_m, 'k_no_spray_estimate': k_no_e,
            'k_no_spray_model': k_no_m, 'reverts_to_spray': reverts, 'reverts_from_no_spray': False}


def test_pick_examples_follows_its_rules():
    early, last = '2023-05-30', '2023-06-06'
    cells = pd.DataFrame([
        row('a', 0, early, SPRAY, 0.020, 1, 1, 2, 2),
        row('a', 1, early, SPRAY, 0.030, 1, 1, 1, 1, reverts=True),   # most weed, but its decision changes back
        row('a', 2, early, SPRAY, 0.025, 1, 1, 1, 1),                 # -> fragile
        row('a', 3, early, SPRAY, 0.050, 2, 2, 3, 3),                 # two patches: not fragile
        row('b', 0, early, DEFER, 0.012, 0, 0, 1, 1),
        row('b', 1, early, DEFER, 0.013, 0, 0, 2, 3),                 # needs 3 patches: not the usual case
        row('b', 2, early, DEFER, 0.011, 0, 0, 1, 1),
        *[row('c', i, last, SPRAY, 0.1, k, k + 2, k + 1, k + 3) for i, k in enumerate((10, 20, 30))],
        row('c', 3, last, SPRAY, 0.1, 40, 41, 41, 42),                # short by one patch only
        row('d', 0, early, SPRAY, 0.1, 5, 9, 6, 9),                   # short by four, but not the last flight
    ])
    picks = dict(F.pick_examples(cells))
    assert (picks['fragile']['tile'], picks['fragile']['cell']) == ('a', 2)
    assert (picks['defer']['tile'], picks['defer']['cell']) == ('b', 0)
    assert (picks['short']['tile'], picks['short']['cell']) == ('c', 1)  # median of k = 12, 22, 32
    assert [k for k, _ in F.pick_examples(cells[cells['decision'] == SPRAY])] == ['fragile', 'short']


@pytest.mark.parametrize('lo, hi', [(0.0132, 0.0099), (0.008, 0.012)])  # crossed (as in the results), not crossed
def test_thresholds_name_the_binding_one(lo, hi):
    no_thr, sp_thr = F.thresholds_cm2({'lam_lo': lo, 'lam_hi': hi})
    assert no_thr == pytest.approx(min(lo, hi) * 2500) and sp_thr == pytest.approx(max(lo, hi) * 2500)


def cell_row(dec, s, k_not, k_no):
    ig = {'R': -0.004, 'G': 0.002, 'B': -0.001, 'NIR': 0.003, 'RE': 0.012}  # adds up to 0.012
    return pd.Series({'tile': 't', 'cell': 0, 'decision_rescored': dec, 's_model': s, 'lam_lo': 0.0132,
                      'lam_hi': 0.0099, 'k_not_spray_model': k_not, 'k_no_spray_model': k_no,
                      'k_not_spray_estimate': 1.0, 'w': 0.02, 'ig_sum': sum(ig.values()),
                      **{f'ig_{b}': v for b, v in ig.items()}})


def steps_for(scores, decisions, est):
    return pd.DataFrame({'k': range(len(scores)), 's_model': scores, 's_estimate': est, 'decision_model': decisions})


def test_text_for_a_spray_cell():
    c = cell_row(SPRAY, 0.016, 1, 2)
    steps = steps_for([0.016, 0.011, 0.008], [SPRAY, DEFER, NO_SPRAY], [0.016, 0.010, 0.007])
    why, what, tail = F.text_lines('fragile', c, steps, 0.0)
    assert 'từ ngưỡng phun 33 cm²' in why and '40 cm²' in why            # 0.016 x 2500 = 40
    assert 'bỏ vùng số 1 (15 cm² cỏ dự đoán)' in what                      # (0.016 - 0.010) x 2500
    assert 'CHƯA CHẮC; bỏ 2 vùng thì KHÔNG PHUN' in what and 'Ước lượng nhanh: 1 vùng' in what
    assert 'RE (+30 cm²) và NIR (+8 cm²)' in tail and 'R kéo xuống nhiều nhất (−10 cm²)' in tail
    assert 'ô thực sự cần phun' in tail                                    # w = 2% >= 1%


def test_text_for_a_defer_cell_and_patch_words():
    c = cell_row(DEFER, 0.011, 0, 1)
    steps = steps_for([0.011, 0.007], [DEFER, NO_SPRAY], [0.011, 0.0068])
    why, what, _ = F.text_lines('defer', c, steps, 5.5)
    assert 'giữa ngưỡng không phun 25 cm² và ngưỡng phun 33 cm²' in why
    assert 'thì ô thành KHÔNG PHUN' in what and 'thêm khoảng 5,5 cm² cỏ thì PHUN' in what
    assert 'nên xem vùng này trước' in what
    assert F.patch_words(1) == 'vùng số 1' and F.patch_words(9) == '9 vùng số 1–9'
    assert F.patch_words(29) == '29 vùng có viền'  # not numbered on the images past NUMBERED


def test_simple_lines():
    # one patch takes the cell to DEFER, the next one to NO_SPRAY
    c = cell_row(SPRAY, 0.016, 1, 2)
    steps = steps_for([0.016, 0.011, 0.008], [SPRAY, DEFER, NO_SPRAY], [0.016, 0.010, 0.007])
    why, change, truth = F.simple_lines(c, steps)
    assert '40 cm²' in why and 'từ ngưỡng phun 33 cm²' in why and why.endswith('→ PHUN.')
    assert change == ('Bỏ vùng số 1 → CHƯA CHẮC; bỏ thêm vùng số 2 → KHÔNG PHUN. '
                      'Chỉ thiếu một đám cỏ nhỏ là quyết định đã đổi.')
    assert 'ô thật sự cần phun' in truth
    # many patches, straight to NO_SPRAY
    c = cell_row(SPRAY, 0.15, 29, 29)
    steps = steps_for([0.15] + [0.012] * 28 + [0.009], [SPRAY] * 29 + [NO_SPRAY], [0.15] * 30)
    assert F.simple_lines(c, steps)[1] == ('Phải bỏ tới 29 vùng có viền (29% diện tích ô) mới hết PHUN '
                                           '(thành KHÔNG PHUN), nên quyết định phun khó đổi.')
    # DEFER, and a cell that does not need spraying
    c = cell_row(DEFER, 0.011, 0, 1)
    c['w'] = 0.005
    why, change, truth = F.simple_lines(c, steps_for([0.011, 0.007], [DEFER, NO_SPRAY], [0.011, 0.0068]))
    assert 'giữa ngưỡng không phun 25 cm² và ngưỡng phun 33 cm² → CHƯA CHẮC' in why
    assert change == 'Bỏ vùng số 1 → KHÔNG PHUN. Người kiểm tra nên xem vùng này trước.'
    assert 'ô thực ra không cần phun' in truth


def test_patch_boxes_are_inside_the_cell():
    # cell 4 = row 1, column 1 of the 3 x 3 grid; its first patch row is 10 and first patch column 10 of 30
    c = {'cell': 4, 'patches': f'{10 * 30 + 10} {19 * 30 + 19} {12 * 30 + 15}'}
    assert F.patch_boxes(c) == [(0, 0), (180, 180), (100, 40)]
    assert np.all(np.array(F.patch_boxes(c)) < F.CELL)
