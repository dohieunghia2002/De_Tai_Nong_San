"""Run the real training and evaluation code of the WeedsGalore pilot on tiny 64x64 crops of real tiles (CPU, a few
seconds). Skipped without PyTorch or without the dataset zip (set WEEDSGALORE_ZIP)."""
import argparse
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from src.weeds import train as T  # noqa: E402
from src.weeds.data import SEMANTIC_CLASSES, read_tile, split_ids, to_3class  # noqa: E402
from src.weeds.model import DeepLabV3Plus  # noqa: E402

ZIP = Path(os.environ.get('WEEDSGALORE_ZIP', 'data/weedsgalore/weedsgalore-dataset.zip'))
pytestmark = pytest.mark.skipif(not ZIP.exists(), reason=f'dataset zip not found: {ZIP}')
CROP = 64


@pytest.fixture(scope='module')
def tiny_data(tmp_path_factory):
    """First tiles of each split, centre-cropped to 64x64, saved in the same npz format as build_npz."""
    out = tmp_path_factory.mktemp('data')
    lo = (600 - CROP) // 2
    with zipfile.ZipFile(ZIP) as zf:
        for split, n in (('train', 4), ('val', 2), ('test', 2)):
            ids = split_ids(zf, split)[:n]
            tiles = [read_tile(zf, t) for t in ids]
            x = np.stack([t[0][:, lo:lo + CROP, lo:lo + CROP] for t in tiles])
            y = np.stack([t[1][lo:lo + CROP, lo:lo + CROP] for t in tiles])
            np.savez_compressed(out / f'weedsgalore_{split}.npz', ids=np.array(ids), x=x, y=y)
    return out


def make_args(data, out, **kw):
    a = argparse.Namespace(data=str(data), out=str(out), stage='train', epochs=2, patience=30, batch=2, ckpt_every=1,
                           lr=1e-3, lr_schedule='constant', workers=0, seed=0, budget_hours=3.5, max_iters=0,
                           limit_val=None, smoke=False, no_pretrained=True)
    vars(a).update(kw)
    return a


def test_model_output_shape():
    m = DeepLabV3Plus(5, 3, pretrained_backbone=False).eval()
    with torch.no_grad():
        assert m(torch.zeros(1, 5, 64, 64)).shape == (1, 3, 64, 64)


def test_confusion_and_iou():
    lab = torch.tensor([0, 0, 1, 1, 2, 2])
    pred = torch.tensor([0, 1, 1, 1, 2, 0])
    cm = T.confusion(pred, lab)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    r = T.iou_from_confusion(cm)
    # background 1 / (2 + 2 - 1), crop 2 / (3 + 2 - 2), weed 1 / (1 + 2 - 1)
    assert r['background'] == pytest.approx(100 / 3)
    assert r['crop'] == pytest.approx(200 / 3)
    assert r['weed'] == pytest.approx(50.0)
    assert r['miou'] == pytest.approx((100 / 3 + 200 / 3 + 50) / 3)


def test_evaluate_six_classes_rgb(tiny_data):
    # the authors' 6-class RGB setting: first three bands, semantic labels, every pixel counted once
    d = T.load_npz(tiny_data / 'weedsgalore_val.npz')
    model = DeepLabV3Plus(3, 6, pretrained_backbone=False)
    res = T.evaluate(model, d, torch.device('cpu'), classes=6, bands=3, amp=False)
    cm = np.array(res['confusion'])
    assert cm.shape == (6, 6) and cm.sum() == d['y'].size
    assert cm.sum(1).tolist() == np.bincount(d['y'].ravel(), minlength=6).tolist()
    assert set(res) == set(SEMANTIC_CLASSES) | {'miou', 'confusion'}
    with pytest.raises(ValueError):
        T.evaluate(model, d, torch.device('cpu'), keep_probs=True, classes=6, bands=3)


def test_augmentation_keeps_image_and_label_aligned(tiny_data, monkeypatch):
    # with the jitter switched off and band 0 set to the label, every rotation / flip must move both identically
    monkeypatch.setattr(T.np.random, 'randn', lambda *shape: np.zeros(shape))
    d = T.load_npz(tiny_data / 'weedsgalore_val.npz')
    y3 = to_3class(d['y'])
    d['x'] = d['x'].copy()
    d['x'][:, 0] = y3.astype(np.uint16) * 1000
    ds = T.TileDataset(d, augment=True)
    for i in range(40):
        img, lab = ds[i % len(ds)]
        assert torch.equal(torch.round(img[0] * 65535 / 1000).long(), lab)
    assert img.dtype == torch.float32 and lab.dtype == torch.int64


def test_train_resume_and_test(tiny_data, tmp_path):
    out = tmp_path / 'run'
    T.train(make_args(tiny_data, out), torch.device('cpu'))
    for f in ('best.pt', 'best.json', 'last.pt', 'history.json', 'train_done.json'):
        assert (out / f).exists(), f
    assert len(json.loads((out / 'history.json').read_text())) == 2
    done = json.loads((out / 'train_done.json').read_text())
    assert 'reached 2 epochs' in done['reason']
    assert (done['batch'], done['lr'], done['seed']) == (2, 1e-3, 0)  # the run records its own settings

    # a finished run is not trained again
    T.train(make_args(tiny_data, out), torch.device('cpu'))
    assert len(json.loads((out / 'history.json').read_text())) == 2

    # resume from last.pt for one more epoch
    (out / 'train_done.json').unlink()
    T.train(make_args(tiny_data, out, epochs=3), torch.device('cpu'))
    hist = json.loads((out / 'history.json').read_text())
    assert [h['epoch'] for h in hist] == [0, 1, 2]
    best = json.loads((out / 'best.json').read_text())['val_miou']
    assert best == pytest.approx(max(h['val_miou'] for h in hist))

    T.test(make_args(tiny_data, out, stage='test'), torch.device('cpu'))
    for split in ('val', 'test'):
        with np.load(out / f'probs_{split}.npz') as p:
            ids = list(p['ids'].astype(str))
            assert p['p_weed'].shape == (2, CROP, CROP) and p['p_weed'].dtype == np.float16
            assert np.all((p['p_crop'].astype(float) + p['p_weed'].astype(float)) <= 1.001)
        assert ids == list(T.load_npz(tiny_data / f'weedsgalore_{split}.npz')['ids'])
    seg = json.loads((out / 'segmentation_metrics.json').read_text())
    assert set(seg) >= {'val', 'test', 'paper_test_3class_msi', 'test_minus_paper_miou'}


def test_cosine_schedule_survives_resume(tiny_data, tmp_path):
    # stopped by the time budget after the first epoch, then resumed: the learning rates must still follow one cosine
    out = tmp_path / 'cosine'
    T.train(make_args(tiny_data, out, epochs=3, lr_schedule='cosine', budget_hours=1e-9), torch.device('cpu'))
    assert not (out / 'train_done.json').exists()
    assert len(json.loads((out / 'history.json').read_text())) == 1
    T.train(make_args(tiny_data, out, epochs=3, lr_schedule='cosine'), torch.device('cpu'))
    lrs = [h['lr'] for h in json.loads((out / 'history.json').read_text())]
    assert lrs == pytest.approx([1e-3 * (1 + math.cos(math.pi * k / 3)) / 2 for k in range(3)])
    done = json.loads((out / 'train_done.json').read_text())
    assert (done['lr_schedule'], done['epochs'], done['patience']) == ('cosine', 3, 30)
