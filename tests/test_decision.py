import numpy as np

from src.decision import decide, decision_metrics
from src.step2_decision import evaluate, pilot_comparison, pilot_reproduced


def test_decide():
    assert list(decide([13, 14.5, 15.5, 17], 1.0, 15)) == ['not_ready', 'defer', 'defer', 'ready']
    assert list(decide([13, 14.5, 15.5, 17], 0.0, 15)) == ['not_ready', 'not_ready', 'ready', 'ready']


def test_decision_metrics_hand_computed():
    # truth ready: F T T F | said ready: T T F F | decided (|p - 15| >= 1): F T F T
    m = decision_metrics(y=[14, 16, 16, 14], p=[15.5, 16.5, 14.8, 13], half_width=1.0, T=15)
    assert m['truly_ready'] == 0.5
    assert m['accuracy_no_defer'] == 0.5
    assert m['false_ready_no_defer'] == 0.25
    assert m['missed_ready_no_defer'] == 0.25
    assert m['interval_coverage'] == 0.5  # |p - y| = 1.5, 0.5, 1.2, 1.0
    assert m['decided'] == 0.5
    assert m['accuracy_decided'] == 1.0
    assert m['false_ready_decided'] == 0.0


def test_step2_reproduces_pilot_up_to_documented_difference(df):
    comparison = pilot_comparison(evaluate(df))
    assert pilot_reproduced(comparison), [c for c in comparison if not c['match']]


def test_step2_with_pilot_quantile_matches_pilot_exactly(df):
    # pilot_dss.py: np.quantile with linear interpolation at level ceil(0.9 (n + 1)) / n
    def pilot_quantile(residuals, alpha):
        r = np.abs(np.asarray(residuals, dtype=float))
        return np.quantile(r, np.ceil((1 - alpha) * (r.size + 1)) / r.size)

    comparison = pilot_comparison(evaluate(df, quantile=pilot_quantile), known={})
    assert all(c['match'] for c in comparison), [c for c in comparison if not c['match']]
