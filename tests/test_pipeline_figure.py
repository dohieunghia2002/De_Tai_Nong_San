"""Text of the pipeline figure (scripts/make_pipeline_figure.py): each explanation must quote the threshold that
actually decided the cell under decision.decide_conformal, for crossed and uncrossed thresholds."""
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip('PIL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import make_pipeline_figure as F  # noqa: E402
from src.weeds.decision import DEFER, NO_SPRAY, SPRAY, decide_conformal  # noqa: E402


def cell(s, lo, hi):
    return {'cell': 0, 's': s, 'lam_lo': lo, 'lam_hi': hi, 'decision': int(decide_conformal([s], lo, hi)[0])}


@pytest.mark.parametrize('lo, hi', [(0.0097, 0.0078), (0.005, 0.02)])  # crossed (as in run 2), not crossed
def test_explanation_quotes_the_binding_threshold(lo, hi):
    low, high = F.pct(min(lo, hi)), F.pct(max(lo, hi))
    spray, no_spray, defer = cell(0.03, lo, hi), cell(0.001, lo, hi), cell((lo + hi) / 2, lo, hi)
    assert (spray['decision'], no_spray['decision'], defer['decision']) == (SPRAY, NO_SPRAY, DEFER)
    assert high in F.explanation(spray) and low not in F.explanation(spray)
    assert low in F.explanation(no_spray) and high not in F.explanation(no_spray)
    text = F.explanation(defer)
    assert text.index(low) < text.index(high)  # 'giữa ngưỡng không phun <low> và ngưỡng phun <high>'


def test_explanation_with_too_few_calibration_cells():
    c = cell(0.01, -math.inf, 0.02)  # 'need' can never be ruled out
    assert c['decision'] == DEFER
    text = F.explanation(c)
    assert 'inf' not in text and 'quá ít ô' in text and 'ngưỡng không phun' in text


def test_same_treats_missing_values_alike():
    assert F.same(float('nan'), None) and F.same(0.1, 0.1 + 1e-12)
    assert not F.same(0.1, None) and not F.same(0.1, 0.2)
