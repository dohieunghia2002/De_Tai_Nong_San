"""Decision layer of the WeedsGalore pilot: hand-computed cases and the exact leave-one-out conformal property."""
import math

import numpy as np
import pandas as pd
import pytest

from src.weeds.analyze import PRIMARY, go_no_go
from src.weeds.data import location_blocks
from src.weeds.decision import (ANY_WEED, DEFER, NO_SPRAY, SPRAY, cell_table, crossfit_decisions, decide_conformal,
                                decide_threshold, decision_metrics, evaluate_crossfit, evaluate_policies,
                                lower_threshold, needs_spray, predicted_weed, upper_threshold)


def test_needs_spray_any_weed_and_thresholds():
    w = np.array([0.0, 1 / 40000, 0.001, 0.0031, 0.02])  # no weed, one weed pixel of a 200 px cell, ...
    assert needs_spray(w, ANY_WEED).tolist() == [False, True, True, True, True]  # w >= 0 would count the empty cell
    assert needs_spray(w, 0.001).tolist() == [False, False, True, True, True]
    assert needs_spray(w, 0.0031).tolist() == [False, False, False, True, True]
    # an exact cover of 124 / 40000 pixels, computed as cell_table does, reaches tau = 0.0031
    y = np.zeros((1, 200, 200), np.uint8)
    y.reshape(-1)[:124] = 2
    t = cell_table(np.array(['2023-05-30_0001']), y, np.zeros(y.shape, np.float32), np.zeros(y.shape, np.float32), 200)
    assert needs_spray(t['w'], 0.0031).tolist() == [True]
    rows = evaluate_policies(pd.DataFrame({'w': [0.0, 0.01], 's': [0.0, 0.01]}),
                             pd.DataFrame({'w': [0.0, 0.01], 's': [0.0, 0.01], 'crop': [0.0, 0.0]}), ANY_WEED, 0.1, 0.1)
    assert {r['policy']: r['infestation'] for r in rows}['oracle'] == 0.5  # the empty cell does not need spraying


def test_cell_table_fractions():
    # one 4x4 tile, 2x2 cells; labels: 0 background, 1 crop, 2 weed
    y = np.array([[[2, 2, 0, 0],
                   [2, 1, 0, 0],
                   [1, 1, 2, 0],
                   [0, 0, 0, 0]]], dtype=np.uint8)
    p = np.zeros_like(y, dtype=np.float32)
    p[0, 0, 0] = 0.8   # weed is the most probable class of this pixel only (background 0.2)
    p[0, 0, 1] = 0.3   # background (0.7) stays the most probable class
    t = cell_table(np.array(['2023-05-30_0001']), y, np.zeros_like(p), p, 2)
    assert list(t['w']) == [0.75, 0.0, 0.0, 0.25]       # cells in row-major order
    assert list(t['crop']) == [0.25, 0.0, 0.5, 0.0]
    assert t['s'].tolist() == [0.25, 0.0, 0.0, 0.0]     # one of the 4 pixels of cell 0 is predicted weed
    assert set(t['date']) == {'2023-05-30'}


def test_score_at_exact_threshold_is_not_rounded_down():
    # a 200 x 200 cell with exactly 400 weed pixels (1 %) and perfect float16 probabilities must reach tau = 0.01
    y = np.zeros((1, 200, 200), dtype=np.uint8)
    y.reshape(-1)[:400] = 2
    p = (y == 2).astype(np.float16)
    t = cell_table(np.array(['2023-05-30_0001']), y, np.zeros_like(p), p, 200)
    assert t['w'].iloc[0] == 0.01 and t['s'].iloc[0] == 0.01
    assert decide_threshold(t['s'], 0.01)[0] == SPRAY


def test_cell_table_rejects_bad_shapes():
    with pytest.raises(ValueError):
        cell_table(np.array(['a']), np.zeros((1, 6, 6), np.uint8), np.zeros((1, 6, 6)), np.zeros((1, 6, 6)), 4)
    with pytest.raises(ValueError):
        cell_table(np.array(['a', 'b']), np.zeros((1, 4, 4), np.uint8), np.zeros((1, 4, 4)), np.zeros((1, 4, 4)), 2)
    with pytest.raises(ValueError):  # P(crop) and P(weed) maps of different shapes
        cell_table(np.array(['a']), np.zeros((1, 4, 4), np.uint8), np.zeros((1, 4, 2)), np.zeros((1, 4, 4)), 2)


def test_predicted_weed_is_the_argmax_of_the_stored_probabilities():
    # argmax over (background = 1 - crop - weed, crop, weed), ties to the earlier class, after float16 rounding
    pc = np.array([0.0, 0.5, 0.40, 0.2, 0.3, 0.1], np.float32)
    pw = np.array([0.8, 0.5, 0.45, 0.5, 0.4, 0.3], np.float32)
    assert predicted_weed(pc, pw).tolist() == [True, False, True, True, True, False]
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=10000).astype(np.float16)  # background, crop, weed as train.evaluate keeps them
    pc, pw = p[:, 1], p[:, 2]
    a, b = pc.astype(np.float32), pw.astype(np.float32)
    assert (predicted_weed(pc, pw) == (np.stack([1 - a - b, a, b]).argmax(0) == 2)).all()
    assert (predicted_weed(pc.astype(np.float32), pw) == predicted_weed(pc, pw)).all()  # float32 input rounded


def test_lower_threshold_ranks():
    s = np.arange(1, 10) / 10                  # 0.1 .. 0.9, n = 9
    assert lower_threshold(s, 0.1) == 0.1      # floor(0.1 * 10) = 1st smallest
    assert lower_threshold(s, 0.2) == 0.2      # 2nd smallest
    assert lower_threshold(s[:8], 0.1) == -math.inf  # floor(0.1 * 9) = 0: cannot rule 'need' out
    # alpha * (n + 1) = 0.29 * 100 evaluates to 28.999999999999996 in floating point; the rank must still be 29
    assert lower_threshold(np.arange(99), 0.29) == 28


def test_upper_threshold_ranks():
    s = np.arange(1, 10) / 10
    assert upper_threshold(s, 0.1) == 0.9      # ceil(0.9 * 10) = 9th smallest
    assert upper_threshold(s, 0.2) == 0.8
    assert upper_threshold(s[:8], 0.1) == math.inf   # ceil(0.9 * 9) = 9 > 8


@pytest.mark.parametrize('alpha', [0.05, 0.1, 0.2])
def test_conformal_leave_one_out_exact(alpha):
    # with n + 1 distinct scores, leaving each one out in turn reproduces the exact miss frequency k / (n + 1)
    s = np.arange(30) / 30.0
    miss = [s[i] < lower_threshold(np.delete(s, i), alpha) for i in range(len(s))]
    k = math.floor(alpha * len(s) + 1e-9)
    assert np.mean(miss) == pytest.approx(k / len(s))
    assert np.mean(miss) <= alpha
    false = [s[i] > upper_threshold(np.delete(s, i), alpha) for i in range(len(s))]
    assert np.mean(false) <= alpha


def test_decide_conformal_regions():
    s = [0.1, 0.4, 0.6]
    assert list(decide_conformal(s, lam_lo=0.3, lam_hi=0.5)) == [NO_SPRAY, DEFER, SPRAY]
    # thresholds crossed (lam_hi < lam_lo): the gap between them is deferred
    assert list(decide_conformal([0.3, 0.5, 0.7], lam_lo=0.6, lam_hi=0.4)) == [NO_SPRAY, DEFER, SPRAY]
    # unusable calibration: nothing can be ruled out, so everything is deferred
    assert set(decide_conformal(s, -math.inf, math.inf)) == {DEFER}
    assert list(decide_threshold(s, 0.4)) == [NO_SPRAY, SPRAY, SPRAY]


def test_decision_metrics_hand_computed():
    d = np.array([SPRAY, NO_SPRAY, DEFER, DEFER, NO_SPRAY])
    need = np.array([True, True, True, False, False])
    w = np.array([0.5, 0.2, 0.1, 0.0, 0.0])
    crop = np.array([0.1, 0.0, 0.0, 0.3, 0.1])
    m = decision_metrics(d, need, w, crop)
    assert m['infestation'] == pytest.approx(0.6)
    assert m['sprayed'] == pytest.approx(0.2)
    assert m['deferred'] == pytest.approx(0.4)
    assert m['effective_sprayed'] == pytest.approx(0.4)   # the sprayed cell + the deferred cell that needs it
    assert m['spray_reduction'] == pytest.approx(0.6)
    assert m['missed_need_rate'] == pytest.approx(1 / 3)
    assert m['false_spray_rate'] == pytest.approx(0.0)
    assert m['missed_weed_area'] == pytest.approx(0.2 / 0.8)
    assert m['crop_area_sprayed'] == pytest.approx(0.1 / 0.5)
    assert m['spray_surplus'] == pytest.approx(0.2 - 0.6)


def test_evaluate_policies_perfect_scores():
    # scores equal to the true weed fraction: no policy that respects tau may miss or over-spray
    w = np.array([0.0, 0.001, 0.003, 0.02, 0.05, 0.2, 0.0, 0.004, 0.03, 0.1, 0.0, 0.002])
    cal = pd.DataFrame({'w': np.tile(w, 3), 's': np.tile(w, 3), 'crop': 0.0})
    test = pd.DataFrame({'w': w, 's': w, 'crop': 0.0})
    rows = {r['policy']: r for r in evaluate_policies(cal, test, tau=0.01, alpha=0.1, beta=0.1)}
    for name in ('oracle', 'plug_in', 'conformal_3way'):
        assert rows[name]['missed_need_rate'] == 0, name
        assert rows[name]['false_spray_rate'] == 0, name
    assert rows['broadcast']['spray_reduction'] == 0
    assert rows['oracle']['spray_reduction'] == pytest.approx(1 - np.mean(w >= 0.01))
    assert rows['conformal_3way']['n_cal_need'] == 3 * int(np.sum(w >= 0.01))


def test_location_blocks_from_split_order():
    ids = ['2023-06-15_0735', '2023-06-06_0736', '2023-05-30_0720', '2023-05-25_0750',
           '2023-06-06_0770', '2023-05-30_0750', '2023-05-25_0786']
    assert location_blocks(ids) == [ids[:4], ids[4:]]


def test_crossfit_uses_only_the_other_fold():
    # 30 cells per fold (15 need, 15 no need); fold 0 scores are shifted up by 0.5.
    # Fold 0 decided with fold 1 thresholds (lam_lo 0.05, lam_hi 0): its 15 no-need cells (s = 0.5) are sprayed.
    # Fold 1 decided with fold 0 thresholds (lam_lo 0.55, lam_hi 0.5): its 15 need cells (s = 0.05) are missed.
    # Self-calibration would give no errors, so 0.5 / 0.5 proves each fold uses the other fold's thresholds.
    w = np.array([0.0, 0.0, 0.0, 0.05, 0.05, 0.05] * 10)
    fold = np.repeat([0, 1], 30)
    s = w + np.where(fold == 0, 0.5, 0.0)
    cells = pd.DataFrame({'w': w, 's': s, 'crop': 0.0})
    rows = {r['policy']: r for r in evaluate_crossfit(cells, fold, tau=0.01, alpha=0.1, beta=0.1)}
    r = rows['conformal_3way']
    assert r['n_cells'] == 60                                   # every cell decided exactly once
    # thresholds are listed per evaluated fold: fold 0 uses fold 1's, fold 1 uses fold 0's
    assert r['lam_lo'] == pytest.approx((0.05, 0.55)) and r['lam_hi'] == pytest.approx((0.0, 0.5))
    assert r['false_spray_rate'] == pytest.approx(0.5)
    assert r['missed_need_rate'] == pytest.approx(0.5)
    assert evaluate_crossfit(cells, np.zeros(60), tau=0.01, alpha=0.1, beta=0.1) == []


def test_crossfit_decisions_are_aligned_with_rows():
    # folds interleaved row by row: decisions and thresholds must follow each row, not the fold order
    w = np.array([0.0, 0.0, 0.0, 0.05, 0.05, 0.05] * 10)
    fold = np.tile([0, 1], 30)
    s = w + np.where(fold == 0, 0.5, 0.0)
    cells = pd.DataFrame({'w': w, 's': s, 'crop': 0.0})
    policies, per_row, per_fold = crossfit_decisions(cells, fold, tau=0.01, alpha=0.1, beta=0.1)
    for k, info in enumerate(per_fold):
        assert np.all(per_row['lam_lo'][fold == k] == info['lam_lo'])
        assert np.all(per_row['lam_hi'][fold == k] == info['lam_hi'])
    np.testing.assert_array_equal(policies['conformal_3way'],
                                  decide_conformal(s, per_row['lam_lo'], per_row['lam_hi']))
    assert crossfit_decisions(cells, np.zeros(60), 0.01, 0.1, 0.1) is None


def test_go_no_go_compares_miou_with_the_paper_value_for_the_same_input():
    key = {k: PRIMARY[k] for k in ('score', 'cell_size', 'tau', 'scope', 'calibration')}
    df = pd.DataFrame([{**key, 'policy': p, 'spray_reduction': 0.5, 'deferred': 0.1, 'missed_need_rate': 0.0,
                        'false_spray_rate': 0.0, 'infestation': 0.5, 'n_cal_need': 10, 'n_cal_no_need': 10,
                        'lam_lo': (0.01,), 'lam_hi': (0.02,)} for p in (PRIMARY['policy'], 'oracle')])
    ours = go_no_go(df, {'test': {'miou': 76.0}})['checks']['C1_test_miou']   # our runs: 5-band paper value 82.90
    assert ours['required'] == '>= 77.90' and not ours['pass']
    rgb = go_no_go(df, {'test': {'miou': 76.0}, 'paper_test': {'miou': 79.33}})['checks']['C1_test_miou']
    assert rgb['required'] == '>= 74.33' and rgb['pass']


def test_go_no_go_reads_the_thresholds_back_from_csv(tmp_path):
    # decision_metrics.csv stores the per-fold thresholds as text: go_no_go must still write numbers (the XAI notebook
    # subtracts them), also for an infinite threshold (too few calibration cells)
    key = {k: PRIMARY[k] for k in ('score', 'cell_size', 'tau', 'scope', 'calibration')}
    df = pd.DataFrame([{**key, 'policy': p, 'spray_reduction': 0.5, 'deferred': 0.1, 'missed_need_rate': 0.0,
                        'false_spray_rate': 0.0, 'infestation': 0.5, 'n_cal_need': 10, 'n_cal_no_need': 10,
                        'lam_lo': (0.0023526254758238, 0.0014994372025132), 'lam_hi': (0.002, math.inf)}
                       for p in (PRIMARY['policy'], 'oracle')])
    df.to_csv(tmp_path / 'decision_metrics.csv', index=False)
    back = pd.read_csv(tmp_path / 'decision_metrics.csv')
    assert isinstance(back['lam_lo'].iloc[0], str)  # what made the notebook fail on 14/11/2026
    seg = {'test': {'miou': 80.0}}
    for frame in (df, back):
        info = go_no_go(frame, seg)['info']
        assert info['lam_lo'] == [0.0023526254758238, 0.0014994372025132] and info['lam_hi'] == [0.002, math.inf]


def test_go_no_go_refuses_metrics_of_the_former_score():
    # decision_metrics.csv written before 15/11/2026 has no 'score' column: its s was the mean P(weed)
    key = {k: PRIMARY[k] for k in ('cell_size', 'tau', 'scope', 'calibration')}
    df = pd.DataFrame([{**key, 'policy': p, 'spray_reduction': 0.5, 'deferred': 0.1, 'missed_need_rate': 0.0,
                        'false_spray_rate': 0.0, 'infestation': 0.5, 'n_cal_need': 10, 'n_cal_no_need': 10,
                        'lam_lo': (0.01,), 'lam_hi': (0.02,)} for p in (PRIMARY['policy'], 'oracle')])
    with pytest.raises(ValueError, match='cell score'):
        go_no_go(df, {'test': {'miou': 80.0}})


def test_go_no_go_c2_is_a_share_of_the_oracle_reduction():
    key = {k: PRIMARY[k] for k in ('score', 'cell_size', 'tau', 'scope', 'calibration')}

    def c2(ours, oracle):
        df = pd.DataFrame([{**key, 'policy': p, 'spray_reduction': v, 'deferred': 0.1, 'missed_need_rate': 0.0,
                            'false_spray_rate': 0.0, 'infestation': 1 - oracle, 'n_cal_need': 10, 'n_cal_no_need': 10,
                            'lam_lo': (0.01,), 'lam_hi': (0.02,)} for p, v in ((PRIMARY['policy'], ours),
                                                                               ('oracle', oracle))])
        return go_no_go(df, {'test': {'miou': 80.0}})['checks']['C2_spray_reduction']

    # oracle 50 % (tau = 1 % on the test cells): exactly the former fixed criterion, 30 %
    assert c2(0.30, 0.5) == {'value': 0.30, 'required': '>= 0.3', 'pass': True}
    assert not c2(0.29, 0.5)['pass']
    # oracle 10.2 % (tau = 0.1 %): 60 % of it
    assert c2(0.157, 0.102)['required'] == '>= 0.0612' and c2(0.157, 0.102)['pass']
    assert not c2(0.06, 0.102)['pass']


def test_primary_decisions_match_the_crossfit_metrics():
    # the per-cell decisions behind the explanations and the figure give the metrics analyze reports
    from src.weeds.analyze import SCOPES, primary_decisions
    from src.weeds.decision import evaluate_crossfit
    rng = np.random.default_rng(0)
    tiles = [f'2023-05-30_{i:04d}' for i in range(8)] + ['2023-06-15_0001']  # the last one is after herbicide
    w = rng.choice([0.0, 0.002, 0.008, 0.02, 0.05], size=9 * len(tiles))
    cells = pd.DataFrame({'tile': np.repeat(tiles, 9), 'date': [t[:10] for t in np.repeat(tiles, 9)],
                          'cell_size': 200, 'w': w, 'crop': 0.1, 's': w + rng.normal(0, 0.004, w.size)})
    fold_of = {t: i % 2 for i, t in enumerate(tiles)}
    t = primary_decisions(cells, fold_of)
    assert set(t['date']) <= set(SCOPES[PRIMARY['scope']]) and len(t) == 72
    assert t.groupby('tile')['cell'].apply(list).map(lambda c: c == list(range(9))).all()
    ref = {r['policy']: r for r in evaluate_crossfit(t, t['fold'].to_numpy(), PRIMARY['tau'], 0.1, 0.1)}
    m = decision_metrics(t['decision'], t['w'] >= PRIMARY['tau'], t['w'], t['crop'])
    for k in ('deferred', 'spray_reduction', 'missed_need_rate', 'false_spray_rate'):
        assert m[k] == pytest.approx(ref[PRIMARY['policy']][k], nan_ok=True)
    assert tuple(t.groupby('fold')['lam_lo'].first()) == pytest.approx(ref[PRIMARY['policy']]['lam_lo'])
