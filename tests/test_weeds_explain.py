"""Explanations and the deletion test (src.weeds.explain) on a small randomly initialised model (CPU, seconds).
The arrays here are synthetic: they test the computations, never results."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from src.weeds import explain as E  # noqa: E402
from src.weeds.decision import DEFER, NO_SPRAY, SPRAY  # noqa: E402
from src.weeds.model import DeepLabV3Plus  # noqa: E402

CELL, PATCH, SIZE = 32, 8, 64
BASE = torch.tensor([0.2, 0.25, 0.3, 0.35, 0.4])


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    m = DeepLabV3Plus(5, 3, pretrained_backbone=False).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope='module')
def tile():
    return torch.rand(5, SIZE, SIZE, generator=torch.Generator().manual_seed(1))


def soil_image():
    return BASE.view(-1, 1, 1).expand(5, SIZE, SIZE).clone()


def test_features_and_head_are_the_forward_pass(model, tile):
    with torch.no_grad():
        assert torch.equal(model.head(model.features(tile[None]), tile.shape[-2:]), model(tile[None]))


def test_cell_scores_are_mean_weed_probabilities(model, tile):
    with torch.no_grad():
        p = torch.softmax(model(tile[None]), 1)[0, 2]
        s = E.cell_scores(model, tile[None], CELL, amp=False)[0]
    assert s.shape == (2, 2)
    assert s[1, 0] == pytest.approx(float(p[CELL:, :CELL].mean()), rel=1e-5)
    assert float(E.cell_sum(p, (1, 0), CELL)) == pytest.approx(float(s[1, 0]) * CELL * CELL, rel=1e-5)


def test_integrated_gradients_completeness(model, tile):
    attr, gap = E.integrated_gradients(model, tile, BASE, (0, 1), CELL, steps=64, batch=8, amp=False)
    assert attr.shape == tile.shape
    assert float(attr.sum()) == pytest.approx(gap, rel=0.05, abs=1e-4)  # attributions add up to s(x) - s(soil)
    same, gap0 = E.integrated_gradients(model, soil_image(), BASE, (0, 1), CELL, steps=4, amp=False)
    assert gap0 == 0 and float(same.abs().max()) == 0  # nothing to attribute when the tile is bare soil


def test_grad_cams(model, tile):
    cams = E.grad_cams(model, tile, [(0, 0), (1, 1)], CELL, amp=False)
    assert len(cams) == 2 and all(c.shape == (SIZE, SIZE) and float(c.min()) >= 0 for c in cams)
    assert not torch.equal(cams[0], cams[1])  # each cell gets its own map


def test_masks_occlusion_and_grid():
    scores = torch.arange(16.0).view(4, 4)
    assert E.top_mask(scores, 3).flatten().nonzero().flatten().tolist() == [13, 14, 15]
    region = E.cell_region((4, 4), (1, 0), cell=16, patch=8)  # cell (1, 0) of a 32 px tile = patches rows 2-3, cols 0-1
    assert region.sum() == 4 and region[2:, :2].all()
    m1, m2 = (E.random_mask(np.random.default_rng(5), (4, 4), 3, region) for _ in range(2))
    assert torch.equal(m1, m2) and int(m1.sum()) == 3 and bool(m1[torch.from_numpy(~region)].any()) is False
    x = torch.rand(5, 32, 32)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[0, 3] = True
    y = E.occlude(x, BASE, mask, 8)
    assert torch.equal(y[:, :8, 24:], BASE.view(-1, 1, 1).expand(5, 8, 8))
    y[:, :8, 24:] = x[:, :8, 24:]
    assert torch.equal(y, x)  # nothing else changed
    for h, cell, patch in ((600, 200, 30), (600, 200, 16), (64, 24, 8)):
        with pytest.raises(ValueError):
            E.check_grid(h, h, cell, patch)
    E.check_grid(600, 600, 200, 20)


def test_deletion_tile(model, tile):
    maps = {'integrated_gradients': torch.rand(SIZE, SIZE), 'predicted_weed': torch.rand(SIZE, SIZE)}
    tested = {0: ((0, 0), maps), 3: ((1, 1), maps)}
    rows = E.deletion_tile(model, tile, BASE, tested, CELL, PATCH, np.random.default_rng(0), fractions=(0.05, 1.0),
                           draws=2, amp=False)
    df = pd.DataFrame(rows)
    # per fraction: 2 shared random-tile occlusions scored for both cells, and per cell 2 maps + 2 random-in-cell
    assert len(df) == 2 * (2 * 2 + 2 * (2 + 2))
    sizes = df.groupby(['method', 'cell']).size()  # 2 fractions: 1 row each per map, `draws` rows each per control
    assert (sizes.loc[list(maps)] == 2).all() and (sizes.loc[['random_tile', 'random_cell']] == 4).all()
    # removing every patch leaves bare soil, whatever ranked the patches
    with torch.no_grad():
        s_soil = E.cell_scores(model, soil_image()[None], CELL, amp=False)[0]
    full = df[(df['fraction'] == 1.0) & (df['method'] != 'random_cell')]
    for _, r in full.iterrows():
        assert r['s_after'] == pytest.approx(float(s_soil[(0, 0) if r['cell'] == 0 else (1, 1)]), abs=1e-6)


def test_band_ablation(model, tile):
    x = tile.clone()
    x[0] = BASE[0]  # band R already bare soil everywhere: setting it to soil changes nothing
    drop = E.band_ablation(model, x, BASE, CELL, amp=False)
    # (a batch of 5 and a batch of 1 round differently, by about 1e-8)
    assert drop.shape == (5, 2, 2) and float(drop[0].abs().max()) < 1e-6


def test_soil_values():
    x = np.full((2, 2, 8, 8), 1000, dtype=np.uint16)
    y = np.zeros((2, 8, 8), dtype=np.uint8)
    y[:, :4] = 2  # weed on the top half, whose values must not count
    x[:, :, :4] = 60000
    x[:, 1, 4:] = 3000
    assert E.soil_values(x, y, step=1) == pytest.approx([1000 / 65535, 3000 / 65535])


def test_explain_tile_and_faithfulness(model, tile):
    cells = pd.DataFrame({'tile': 't', 'cell': [0, 1, 2, 3], 'decision': [SPRAY, DEFER, NO_SPRAY, SPRAY]})
    rows, dels, maps, timing = E.explain_tile(model, tile, BASE, cells, CELL, PATCH, np.random.default_rng(0), steps=4,
                                              draws=2, amp=False)
    assert [r['cell'] for r in rows] == [0, 1, 2, 3] and timing['tested'] == 3
    assert {r['cell'] for r in dels} == {0, 1, 3}  # SPRAY and DEFER cells only
    assert maps['integrated_gradients'].dtype == np.float32  # float16 would flush these small values to zero
    assert maps['integrated_gradients'].shape == maps['grad_cam'].shape == (4, SIZE // E.MAP_POOL, SIZE // E.MAP_POOL)
    assert {f'ablate_{b}' for b in 'R G B NIR RE'.split()} <= set(rows[0])
    s = np.array([r['s_model'] for r in rows])
    # thresholds around the rescored scores: cell 0 is SPRAY by its own s_model, cell 3 is not
    lam = pd.DataFrame({'cell': [0, 1, 2, 3], 'lam_lo': [s[0] - 1e-3, 1, 1, s[3] + 1e-3],
                        'lam_hi': [s[0] - 2e-3, 1, 1, s[3] + 2e-3]})
    xai = cells.merge(pd.DataFrame(rows).assign(tile='t'), on=['tile', 'cell']).merge(lam, on='cell')
    out = E.faithfulness(pd.DataFrame(dels).assign(tile='t'), xai)
    assert set(out) == set(E.METHODS + E.CONTROLS)
    for m in E.METHODS:
        for res in out[m].values():
            assert res['cells'] == 3 and 0 <= res['beats_random_tile'] <= 1 and 0 <= res['beats_random_cell'] <= 1
    # spray flips are counted only over cell 0, the one cell that is SPRAY on its rescored score
    one = E.faithfulness(pd.DataFrame(dels).assign(tile='t').query('cell == 0'), xai)
    assert out['grad_cam']['0.1']['spray_flip_rate'] == one['grad_cam']['0.1']['spray_flip_rate']
    assert E.faithfulness(pd.DataFrame(), xai) == {}


def test_decision_check():
    t = pd.DataFrame({'decision': [SPRAY, NO_SPRAY, DEFER, NO_SPRAY], 'w': [0.02, 0.0, 0.02, 0.0],
                      'crop': 0.1, 'fold': [0, 0, 1, 1], 'lam_lo': [0.01, 0.01, 0.02, 0.02],
                      'lam_hi': [0.005, 0.005, 0.006, 0.006]})
    reported = {'checks': {'C2_spray_reduction': {'value': 0.5}, 'C3_deferred': {'value': 0.25}},
                'info': {'missed_need_rate': 0.0, 'false_spray_rate': 0.0, 'lam_lo': [0.01, 0.02],
                         'lam_hi': [0.005, 0.006]}}
    assert E.decision_check(t, reported)['max_abs_diff'] == pytest.approx(0)
    reported['info']['lam_hi'] = [0.005, 0.007]
    assert E.decision_check(t, reported)['max_abs_diff'] == pytest.approx(0.001)
    assert 'reported' not in E.decision_check(t, None)


def test_run_end_to_end(tmp_path, monkeypatch):
    """The whole run on real test tiles cropped to one 200 px cell, with a random model: decisions, per-tile parts,
    resuming, merged outputs and the summary. About 2 minutes on CPU, so it runs only with WEEDS_SLOW_TESTS=1
    (run it before sending changed XAI code to Colab)."""
    import argparse
    import json
    import os
    import zipfile
    from pathlib import Path
    from src.weeds.data import read_tile, split_ids
    if os.environ.get('WEEDS_SLOW_TESTS') != '1':
        pytest.skip('slow end-to-end test: set WEEDS_SLOW_TESTS=1')
    zip_path = Path(os.environ.get('WEEDSGALORE_ZIP', 'data/weedsgalore/weedsgalore-dataset.zip'))
    if not zip_path.exists():
        pytest.skip(f'dataset zip not found: {zip_path}')
    data = tmp_path / 'data'
    data.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        for split, n in (('train', 2), ('test', None)):  # every test tile: the location folds need all of them
            ids = split_ids(zf, split)[:n]
            tiles = [read_tile(zf, t) for t in ids]
            np.savez_compressed(data / f'weedsgalore_{split}.npz', ids=np.array(ids),
                                x=np.stack([t[0][:, 200:400, 200:400] for t in tiles]),
                                y=np.stack([t[1][200:400, 200:400] for t in tiles]))
    torch.manual_seed(0)
    small = DeepLabV3Plus(5, 3, pretrained_backbone=False).eval()
    monkeypatch.setattr(E, 'load_checkpoint', lambda *a: small)
    out = tmp_path / 'xai'
    args = argparse.Namespace(data=str(data), ckpt='fake.pth', out=str(out), expect=None, steps=2, patch=20, draws=1,
                              ig_batch=2, batch=8, seed=0, amp=True, budget_hours=3.0, limit_tiles=2)
    E.run(args, torch.device('cpu'))
    decisions = pd.read_csv(out / 'decisions.csv')
    assert len(decisions) == 24  # 24 pre-herbicide test tiles, one 200 px cell each
    cells = pd.read_csv(out / 'xai_cells.csv')
    assert len(cells) == 2 and (cells['s_model'] - cells['s']).abs().max() < 1e-3
    with np.load(out / 'xai_maps.npz') as m:
        assert list(m['tile']) == list(cells['tile']) and m['integrated_gradients'].shape == (2, 50, 50)
    summary = json.loads((out / 'xai_summary.json').read_text())
    assert summary['cells'] == 2 and len(summary['soil_baseline']) == 5 and summary['decision_check'] is not None
    # a second run reuses the decisions and every finished tile, and only merges again
    before = {p.name: p.stat().st_mtime_ns for p in [*(out / 'parts').iterdir(), out / 'decisions.csv']}
    E.run(args, torch.device('cpu'))
    assert {p.name: p.stat().st_mtime_ns for p in [*(out / 'parts').iterdir(), out / 'decisions.csv']} == before
    # other settings in the same folder are refused
    args.steps = 4
    with pytest.raises(ValueError):
        E.run(args, torch.device('cpu'))
