"""Model check of the counterfactual explanations (src.weeds.counterfactual_check), on CPU in seconds. A per-pixel
fake model (no context) makes the estimate exact, so the scan's k can be computed by hand; a small random DeepLabv3+
(with context) checks which pixels are removed. The arrays are synthetic: they test the computations, never results."""
import argparse
import json
import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from src.weeds import counterfactual_check as C  # noqa: E402
from src.weeds import explain as E  # noqa: E402
from src.weeds.data import BANDS  # noqa: E402
from src.weeds.decision import DEFER, NO_SPRAY, SPRAY  # noqa: E402
from src.weeds.model import DeepLabV3Plus  # noqa: E402

CELL, PATCH, SIZE = 32, 8, 64  # 2 x 2 cells of 4 x 4 patches; the tile's patch grid is 8 x 8
BASE = torch.tensor([0.2, 0.25, 0.3, 0.35, 0.4])


class PixelModel(torch.nn.Module):
    """P(weed) of each pixel from its own band R only: about 1 where R = 1, about 2e-5 on bare soil (R = 0.2)."""

    def forward(self, x):
        w = 100 * (x[:, 0] - 0.3)
        z = torch.zeros_like(w)
        return torch.stack([z, z, w], 1)


def weedy_tile():
    x = BASE.view(-1, 1, 1).expand(5, SIZE, SIZE).clone()
    x[0, 0:8, 0:8] = 1.0      # cell 0, patch (0, 0): weed mass about 64 pixels
    x[0, 0:4, 8:16] = 1.0     # cell 0, patch (0, 1): about 32
    x[0, 8:10, 0:8] = 1.0     # cell 0, patch (1, 0): about 16
    x[0, 32:40, 32:40] = 1.0  # cell 3, patch (0, 0): about 64
    return x


def cells_table():
    # cell 0: s ~ 112/1024; after 1 patch ~ 48/1024 (SPRAY), 2 ~ 16/1024 (DEFER), 3 ~ 0 (NO_SPRAY)
    # cell 3: lam_lo = -inf, so 'need' is never ruled out and NO_SPRAY cannot be reached
    return pd.DataFrame({'tile': 't', 'date': '2023-05-30', 'cell': [0, 3], 'decision': [SPRAY, SPRAY],
                         'lam_lo': [0.03, -np.inf], 'lam_hi': [0.01, 0.01]})


def test_scan_with_a_per_pixel_model():
    steps, cells, passes = C.verify_tile(PixelModel(), weedy_tile(), BASE, cells_table(), CELL, PATCH, batch=2,
                                         extra=2, amp=False)
    st, ce = pd.DataFrame(steps), pd.DataFrame(cells).set_index('cell')
    # without context the estimate is the model, up to the bare-soil P(weed) of about 2e-5 per pixel
    assert (st['s_model'] - st['s_estimate']).abs().max() < 1e-4
    assert (st['decision_model'] == st['decision_estimate']).all()
    c0 = ce.loc[0]
    assert (c0['k_not_spray_estimate'], c0['k_no_spray_estimate']) == (2, 3)
    assert (c0['k_not_spray_model'], c0['k_no_spray_model']) == (2, 3)
    # NO_SPRAY at k = 3 is found in the batch (3, 4); the scan must pass k = 3 + extra = 5, so it ends after (5, 6)
    assert c0['k_scanned'] == 6 and list(st[st['cell'] == 0]['k']) == list(range(7))
    assert c0['patches'].split()[:3] == ['0', '1', '8']   # cell 0's patches (0,0), (0,1), (1,0) in the 8 x 8 grid
    c3 = ce.loc[3]
    assert c3['k_not_spray_model'] == 1 and math.isnan(c3['k_no_spray_model']) and math.isnan(c3['k_no_spray_estimate'])
    assert c3['k_scanned'] == 16                          # NO_SPRAY never comes: every patch of the cell is removed
    assert c3['patches'].split()[0] == '36'               # cell (1, 1), patch (0, 0) -> grid row 4, column 4
    assert passes == 1 + 6 + 16
    assert not ce['reverts_to_spray'].any() and not ce['reverts_from_no_spray'].any()
    assert ce['max_score_increase'].max() < 1e-6
    # the plan assumes the model agrees with the estimate: 6 for cell 0, and up to k = 1 + 2 in batches of 2 for cell 3
    assert C.planned_passes(PixelModel(), {'t': weedy_tile()}, cells_table(), CELL, PATCH, 2, 2, amp=False) == 1 + 6 + 4


def test_scan_target():
    assert C.scan_target(3, (2.0, 3.0), 2) == 5
    assert C.scan_target(3, (5.0, np.nan), 2) == 7
    assert C.scan_target(4, (np.nan, np.nan), 0) == 4


def test_removed_patches_are_the_cells_own():
    """With a model that sees context: removing every patch of a cell scores the tile with that cell set to soil."""
    torch.manual_seed(0)
    model = DeepLabV3Plus(5, 3, pretrained_backbone=False).eval()
    x = torch.rand(5, SIZE, SIZE, generator=torch.Generator().manual_seed(1))
    cells = pd.DataFrame({'tile': 't', 'cell': [1], 'decision': [SPRAY], 'lam_lo': [-np.inf], 'lam_hi': [-1.0]})
    steps, ce, _ = C.verify_tile(model, x, BASE, cells, CELL, PATCH, batch=8, extra=2, amp=False)
    assert ce[0]['k_scanned'] == 16 and sorted(int(p) for p in ce[0]['patches'].split()) == [
        r * 8 + c for r in range(4) for c in range(4, 8)]  # cell 1 = rows 0-3, columns 4-7 of the patch grid
    soil_cell = x.clone()
    soil_cell[:, :CELL, CELL:] = BASE.view(-1, 1, 1)
    with torch.no_grad():
        expected = float(E.cell_scores(model, soil_cell[None], CELL, amp=False)[0, 0, 1])
    assert steps[-1]['s_model'] == pytest.approx(expected, abs=1e-6)
    assert steps[0]['s_model'] == pytest.approx(steps[0]['s_estimate'])


def test_scan_matches_one_image_at_a_time():
    """With a model that sees context and whose score need not fall monotonically: an independent ranking and every
    k scored one image at a time give the same patches, scores, first changes and reversals as the batched scan."""
    torch.manual_seed(2)
    model = DeepLabV3Plus(5, 3, pretrained_backbone=False).eval()
    x = torch.rand(5, SIZE, SIZE, generator=torch.Generator().manual_seed(3))

    def score(img):  # mean P(weed) over cell 0, one image per pass
        with torch.no_grad():
            return float(torch.softmax(model(img[None]), 1)[0, 2, :CELL, :CELL].double().mean())

    with torch.no_grad():
        p = torch.softmax(model(x[None]), 1)[0, 2]
    mass = {i * 8 + j: float(p[i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH].double().sum())
            for i in range(4) for j in range(4)}  # cell 0 = rows 0-3, columns 0-3 of the 8 x 8 patch grid
    ranking = sorted(mass, key=lambda k: -mass[k])
    curve = []
    for k in range(17):
        img = x.clone()
        for pid in ranking[:k]:
            i, j = divmod(pid, 8)
            img[:, i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH] = BASE.view(-1, 1, 1)
        curve.append(score(img))
    curve = np.array(curve)
    span = curve[0] - curve.min()
    assert span > 1e-3  # the thresholds below sit inside the curve's range
    lo, hi = curve[0] - 0.3 * span, curve[0] - 0.7 * span
    assert np.abs(np.subtract.outer(curve, [lo, hi])).min() > 1e-5  # no score within rounding of a threshold
    dec = np.where((curve >= lo) & (curve > hi), SPRAY, np.where((curve < lo) & (curve <= hi), NO_SPRAY, DEFER))
    cells = pd.DataFrame({'tile': 't', 'cell': [0], 'decision': [SPRAY], 'lam_lo': [lo], 'lam_hi': [hi]})
    steps, ce, _ = C.verify_tile(model, x, BASE, cells, CELL, PATCH, batch=3, extra=2, amp=False)
    n = ce[0]['k_scanned'] + 1
    assert [int(v) for v in ce[0]['patches'].split()] == ranking[:n - 1]
    assert [r['s_model'] for r in steps] == pytest.approx(curve[:n], abs=1e-6)
    k_not = int(np.flatnonzero(dec != SPRAY)[0])
    assert ce[0]['k_not_spray_model'] == k_not
    no = np.flatnonzero(dec == NO_SPRAY)
    if len(no):
        assert ce[0]['k_no_spray_model'] == no[0] and n - 1 >= no[0] + 2
        assert ce[0]['reverts_from_no_spray'] == bool((dec[no[0]:n] != NO_SPRAY).any())
    else:
        assert math.isnan(ce[0]['k_no_spray_model']) and n == 17
    assert ce[0]['reverts_to_spray'] == bool((dec[k_not:n] == SPRAY).any())
    assert ce[0]['max_score_increase'] == pytest.approx(max(np.diff(curve[:n]).max(), 0.0), abs=1e-6)


def test_summarize_check():
    cells = pd.DataFrame({'tile': 't', 'cell': [0, 1, 2, 3], 'decision': [SPRAY, SPRAY, SPRAY, DEFER],
                          'decision_rescored': [SPRAY, SPRAY, SPRAY, DEFER],
                          'k_not_spray_estimate': [1.0, 2.0, 5.0, 0.0], 'k_not_spray_model': [2.0, 2.0, 4.0, 0.0],
                          'k_no_spray_estimate': [1.0, 3.0, 6.0, 1.0], 'k_no_spray_model': [2.0, np.nan, 6.0, 1.0],
                          'reverts_to_spray': [True, False, False, False],
                          'reverts_from_no_spray': [False, False, False, False],
                          'max_score_increase': [0.0, 0.0, 1e-4, 0.0]})
    steps = pd.DataFrame({'tile': 't', 'cell': [0, 0, 3, 3], 'k': [0, 1, 0, 1], 's_model': [0.1, 0.05, 0.02, 0.0],
                          's_estimate': [0.1, 0.04, 0.02, 0.0], 'decision_model': [SPRAY, SPRAY, DEFER, NO_SPRAY],
                          'decision_estimate': [SPRAY, NO_SPRAY, DEFER, NO_SPRAY]})
    s = C.summarize_check(cells, steps, CELL, gsd_cm=1.0)
    assert s['k_not_spray'] == {'cells': 3, 'compared': 3, 'model_not_reached': 0, 'estimate_not_reached': 0,
                                'exact': 1, 'estimate_too_small': 1, 'estimate_too_large': 1,
                                'model_minus_estimate': {'-1': 1, '0': 1, '1': 1}}
    assert s['k_not_spray_fragile']['cells'] == 2 and s['k_not_spray_fragile']['estimate_too_small'] == 1
    assert s['k_no_spray']['cells'] == 4 and s['k_no_spray']['model_not_reached'] == 1
    assert s['reverts_to_spray'] == 1 and s['max_score_increase_cm2']['max'] == pytest.approx(1e-4 * CELL ** 2)
    # k > 0 rows only: cell 0 disagrees (estimate NO_SPRAY, model SPRAY), cell 3 agrees; both changed
    assert s['curve']['rows'] == 2 and s['curve']['decision_agreement'] == 0.5
    assert s['curve']['rows_decision_changed'] == 2
    assert s['curve']['error_cm2_median'] == pytest.approx((-0.01 + 0.0) / 2 * CELL ** 2)


def test_run_resumes_and_merges(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    ids = ['2023-05-30_0001', '2023-05-30_0002']
    x = np.stack([weedy_tile().numpy(), BASE.view(-1, 1, 1).expand(5, SIZE, SIZE).numpy()])
    np.savez_compressed(data / 'weedsgalore_test.npz', ids=np.array(ids), x=np.round(x * 65535).astype(np.uint16),
                        y=np.zeros((2, SIZE, SIZE), np.uint8))
    decisions = pd.concat([cells_table().assign(tile=ids[0]),
                           pd.DataFrame({'tile': ids[1], 'date': '2023-05-30', 'cell': [0, 1], 'decision':
                                         [NO_SPRAY, DEFER], 'lam_lo': 0.03, 'lam_hi': 1e-6})])
    decisions.to_csv(tmp_path / 'decisions.csv', index=False)
    (tmp_path / 'soil.json').write_text(json.dumps({'bands': list(BANDS), 'values': BASE.tolist()}))
    monkeypatch.setattr(C, 'load_checkpoint', lambda *a: PixelModel())

    def args(out, **kw):
        a = dict(data=str(data), ckpt='fake.pth', decisions=str(tmp_path / 'decisions.csv'),
                 soil=str(tmp_path / 'soil.json'), out=str(out), cell=CELL, patch=PATCH, extra=2, batch=2, amp=False,
                 budget_hours=1.0, limit_tiles=0)
        return argparse.Namespace(**{**a, **kw})

    C.run(args(tmp_path / 'smoke', limit_tiles=1), torch.device('cpu'))  # one tile, then the estimate only
    assert [p.name for p in (tmp_path / 'smoke' / 'parts').iterdir()] == [f'{ids[0]}.json']
    assert not (tmp_path / 'smoke' / 'cfcheck_summary.json').exists()
    out = tmp_path / 'check'
    C.run(args(out), torch.device('cpu'))
    cells = pd.read_csv(out / 'cfcheck_cells.csv')
    assert len(cells) == 3 and set(cells['tile']) == set(ids)  # the NO_SPRAY cell is not checked
    assert set(pd.read_csv(out / 'cfcheck_steps.csv')['tile']) == set(ids)
    summary = json.loads((out / 'cfcheck_summary.json').read_text())
    assert summary['cells'] == 3 and summary['decisions'] == {'spray': 2, 'defer': 1}
    assert summary['rescored_decision_differs'] == 0 and summary['k_not_spray']['exact'] == 2
    # cell 3 (lam_lo = -inf) never reaches NO_SPRAY. The all-soil DEFER cell does under the estimate, which takes a
    # removed patch to P(weed) = 0 and so falls below lam_hi = 1e-6, but not under the model, which still gives bare
    # soil about 2e-5: the kind of gap the check is there to measure
    assert summary['k_no_spray']['estimate_not_reached'] == 1 and summary['k_no_spray']['model_not_reached'] == 2
    assert summary['forward_passes'] > 0
    # a second run reuses every finished tile; other settings in the same folder are refused
    before = {p.name: p.stat().st_mtime_ns for p in (out / 'parts').iterdir()}
    C.run(args(out), torch.device('cpu'))
    assert {p.name: p.stat().st_mtime_ns for p in (out / 'parts').iterdir()} == before
    with pytest.raises(ValueError):
        C.run(args(out, extra=3), torch.device('cpu'))
    decisions.assign(lam_hi=2e-6).to_csv(tmp_path / 'decisions.csv', index=False)  # other decisions, same folder
    with pytest.raises(ValueError):
        C.run(args(out), torch.device('cpu'))
    with pytest.raises(ValueError):
        (tmp_path / 'soil.json').write_text(json.dumps({'bands': ['R', 'G', 'B'], 'values': [0.2, 0.2, 0.2]}))
        C.run(args(tmp_path / 'other'), torch.device('cpu'))
