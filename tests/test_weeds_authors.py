"""Loading the authors' released checkpoints into src.weeds.model and the evaluation outputs (CPU, a few seconds).
The real checkpoints (887 MB) are only used on Colab; here they are imitated with randomly initialised models saved
under the authors' (VainF) parameter names. The synthetic arrays below test code paths only, never results."""
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from src.weeds import authors as A  # noqa: E402
from src.weeds.analyze import load_cells  # noqa: E402
from src.weeds.data import CLASSES3, SEMANTIC_CLASSES  # noqa: E402
from src.weeds.model import DeepLabV3Plus  # noqa: E402

# ours -> theirs, the inverse of A.RENAMES
TO_AUTHORS = [('stem.0.', 'backbone.conv1.'), ('stem.1.', 'backbone.bn1.'), ('layer', 'backbone.layer'),
              ('low_proj.', 'classifier.project.'), ('aspp.branches.', 'classifier.aspp.convs.'),
              ('aspp.project.', 'classifier.aspp.project.'), ('classifier.', 'classifier.classifier.')]


def as_authors_state_dict(model):
    out = {}
    for k, v in model.state_dict().items():
        old, new = next((o, n) for o, n in TO_AUTHORS if k.startswith(o))
        out[new + k[len(old):]] = v
    return out


def test_rename_vainf_names():
    # names as produced by VainF's DeepLabV3Plus (backbone via IntermediateLayerGetter, DeepLabHeadV3Plus head)
    sd = {'backbone.conv1.weight': 1, 'backbone.bn1.running_var': 2, 'backbone.layer4.2.bn3.weight': 3,
          'classifier.project.1.num_batches_tracked': 4, 'classifier.aspp.convs.4.1.weight': 5,
          'classifier.aspp.project.0.weight': 6, 'module.classifier.classifier.3.bias': 7}
    assert A.rename_state_dict(sd) == {'stem.0.weight': 1, 'stem.1.running_var': 2, 'layer4.2.bn3.weight': 3,
                                       'low_proj.1.num_batches_tracked': 4, 'aspp.branches.4.1.weight': 5,
                                       'aspp.project.0.weight': 6, 'classifier.3.bias': 7}
    with pytest.raises(KeyError):
        A.rename_state_dict({'backbone.fc.weight': 0})


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    ref = DeepLabV3Plus(5, 3, pretrained_backbone=False).eval()
    sd = as_authors_state_dict(ref)
    torch.save(sd, tmp_path / 'ckpt.pth')
    model = A.load_checkpoint(tmp_path / 'ckpt.pth', 5, 3, torch.device('cpu'))
    x = torch.rand(1, 5, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(x), ref(x))
    # strict loading: a checkpoint with a missing tensor must fail, not silently keep random weights
    sd.pop('classifier.classifier.3.bias')
    torch.save(sd, tmp_path / 'broken.pth')
    with pytest.raises(RuntimeError):
        A.load_checkpoint(tmp_path / 'broken.pth', 5, 3, torch.device('cpu'))


def test_run_writes_analyze_compatible_outputs(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    data, ckpts, out = tmp_path / 'data', tmp_path / 'ckpts', tmp_path / 'out'
    data.mkdir()
    ckpts.mkdir()
    for split in ('val', 'test'):
        ids = np.array([f'2023-05-30_{split}{i}' for i in range(2)])
        np.savez_compressed(data / f'weedsgalore_{split}.npz', ids=ids,
                            x=rng.integers(5000, 60000, (2, 5, 64, 64), dtype=np.uint16),
                            y=rng.integers(0, 6, (2, 64, 64), dtype=np.uint8))
    names = {'dlv3p_msi_3': (5, 3), 'dlv3p_rgb_6': (3, 6)}
    monkeypatch.setattr(A, 'CKPTS', names)
    for n in names:
        (ckpts / f'{n}.pth').touch()

    def fake_load(path, bands, classes, dev):
        torch.manual_seed(0)
        return DeepLabV3Plus(bands, classes, pretrained_backbone=False).eval()
    monkeypatch.setattr(A, 'load_checkpoint', fake_load)

    summary = A.run(data, ckpts, out, torch.device('cpu'))
    assert set(summary) == set(names)
    for name, (bands, classes) in names.items():
        seg = json.loads((out / name / 'segmentation_metrics.json').read_text())
        for r in (seg['val'], seg['test'], seg['fp32']['val'], seg['fp32']['test']):
            cm = np.array(r['confusion'])
            assert cm.shape == (classes, classes) and cm.sum() == 2 * 64 * 64
            ious = [r[c] for c in (CLASSES3 if classes == 3 else SEMANTIC_CLASSES) if r[c] is not None]
            assert r['miou'] == pytest.approx(np.mean(ious))
        # on CPU float16 autocast is off, so both evaluations must agree exactly
        assert seg['test']['confusion'] == seg['fp32']['test']['confusion']
        assert seg['fp32_test_minus_paper_miou'] == pytest.approx(seg['fp32']['test']['miou'] - A.PAPER[name]['miou'])
        assert seg['paper_test'] == A.PAPER[name]
        assert (out / name / 'probs_test.npz').exists() == (classes == 3)
    with np.load(out / 'dlv3p_msi_3' / 'probs_test.npz') as p:
        assert p['p_weed'].shape == (2, 64, 64) and p['p_weed'].dtype == np.float16
        assert list(p['ids'].astype(str)) == ['2023-05-30_test0', '2023-05-30_test1']
    # the decision layer reads these outputs as it reads our own runs
    cells = load_cells(data, out / 'dlv3p_msi_3', 'test', 32)
    assert len(cells) == 2 * 4 and cells['s'].between(0, 1).all()
    assert json.loads((out / 'authors_vs_paper.json').read_text()).keys() == set(names)

    (ckpts / 'dlv3p_rgb_6.pth').unlink()
    with pytest.raises(FileNotFoundError):
        A.run(data, ckpts, out, torch.device('cpu'))
