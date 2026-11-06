"""The run list of the seed / schedule study and its summary (no training: result folders are faked)."""
import json

import pandas as pd
import pytest

pytest.importorskip('torch')

from src.weeds import runs as R  # noqa: E402
from src.weeds.analyze import PRIMARY  # noqa: E402
from src.weeds.train import parser  # noqa: E402


def test_runs_parse_as_training_arguments():
    names = [r['name'] for r in R.RUNS]
    assert len(set(names)) == len(names)
    for r in R.RUNS:
        a = parser().parse_args(['--data', 'd', '--out', 'o', *R.train_args(r)])
        assert (a.batch, a.seed, a.epochs, a.patience, a.lr_schedule) == (
            8, r['seed'], r['epochs'], r['patience'], r['lr_schedule'])


def test_schedule_labels():
    assert R.schedule({'epochs': 100, 'patience': 100, 'lr_schedule': 'constant'}) == '100 epochs fixed'
    assert R.schedule({'epochs': 100, 'patience': 100, 'lr_schedule': 'cosine'}) == '100 epochs, cosine lr'
    assert R.schedule({}) == 'early stopping (30)'  # run 2 predates these fields: parser defaults 200 / 30 / constant


def fake_run(d, seed, done, test_miou, deferred):
    d.mkdir()
    (d / 'train_done.json').write_text(json.dumps({'seed': seed, 'epochs_run': 100, **done}))
    (d / 'best.json').write_text(json.dumps({'epoch': 90, 'val_miou': 83.0}))
    (d / 'segmentation_metrics.json').write_text(json.dumps({'test': {'miou': test_miou, 'crop': 70.0, 'weed': 75.0}}))
    key = {k: PRIMARY[k] for k in ('cell_size', 'tau', 'scope', 'calibration')}
    pd.DataFrame([{**key, 'policy': p, 'spray_reduction': 0.5, 'deferred': deferred if p == PRIMARY['policy'] else 0.0,
                   'missed_need_rate': 0.01, 'false_spray_rate': 0.02} for p in (PRIMARY['policy'], 'plug_in')]
                 ).to_csv(d / 'decision_metrics.csv', index=False)


def test_summarize_and_by_schedule(tmp_path):
    e100 = {'epochs': 100, 'patience': 100, 'lr_schedule': 'constant'}
    fake_run(tmp_path / 'a', 0, e100, 80.0, 0.05)
    fake_run(tmp_path / 'b', 1, e100, 82.0, 0.06)
    fake_run(tmp_path / 'c', 0, {}, 79.0, 0.07)
    (tmp_path / 'unfinished').mkdir()
    df = R.summarize({'a': tmp_path / 'a', 'b': tmp_path / 'b', 'c': tmp_path / 'c', 'x': tmp_path / 'unfinished'})
    assert list(df['run']) == ['a', 'b', 'c']
    assert df.set_index('run').loc['b', 'deferred'] == 0.06  # the conformal_3way row, not plug_in
    s = R.by_schedule(df).set_index('schedule')
    assert s.loc['100 epochs fixed', 'n'] == 2 and s.loc['100 epochs fixed', 'mean'] == pytest.approx(81.0)
    assert s.loc['100 epochs fixed', 'std'] == pytest.approx(2 ** 0.5)
    assert s.loc['early stopping (30)', 'n'] == 1
