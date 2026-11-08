"""The run list of the seed / schedule study and its summary (no training: result folders are faked)."""
import json

import pandas as pd
import pytest

from src.weeds import runs as R
from src.weeds.analyze import PRIMARY


def test_runs_parse_as_training_arguments():
    train = pytest.importorskip('src.weeds.train')
    names = [r['name'] for r in R.RUNS]
    assert len(set(names)) == len(names)
    for r in R.RUNS:
        a = train.parser().parse_args(['--data', 'd', '--out', 'o', *R.train_args(r)])
        assert (a.batch, a.seed, a.epochs, a.patience, a.lr_schedule) == (
            8, r['seed'], r['epochs'], r['patience'], r['lr_schedule'])
    # run 2's schedule: what LEGACY says it used, and what the training defaults still are
    defaults = train.parser().parse_args(['--data', 'd', '--out', 'o'])
    assert (defaults.epochs, defaults.patience, defaults.lr_schedule) == tuple(R.LEGACY.values())


def test_schedule_labels():
    assert R.schedule({'epochs': 100, 'patience': 100, 'lr_schedule': 'constant'}) == '100 epochs fixed'
    assert R.schedule({'epochs': 100, 'patience': 100, 'lr_schedule': 'cosine'}) == '100 epochs, cosine lr'
    assert R.schedule({'batch': 8, 'seed': 0}) == 'early stopping (30)'  # run 2's train_done.json: LEGACY
    assert [R.schedule(r) for r in R.RUNS].count('100 epochs fixed') == 3


def write_model(d, seg_model, test_miou, deferred):
    d.mkdir(parents=True)
    (d / 'segmentation_metrics.json').write_text(json.dumps(
        {**seg_model, 'val': {'miou': 83.0}, 'test': {'miou': test_miou, 'crop': 70.0, 'weed': 75.0}}))
    key = {k: PRIMARY[k] for k in ('cell_size', 'tau', 'scope', 'calibration')}
    pd.DataFrame([{**key, 'policy': p, 'spray_reduction': 0.5, 'deferred': deferred if p == PRIMARY['policy'] else 0.0,
                   'missed_need_rate': 0.01, 'false_spray_rate': 0.02} for p in (PRIMARY['policy'], 'plug_in')]
                 ).to_csv(d / 'decision_metrics.csv', index=False)


def fake_run(d, seed, done, test_miou, deferred, last_miou=None):
    write_model(d, {}, test_miou, deferred)  # run 2's segmentation_metrics.json has no 'model' entry
    (d / 'train_done.json').write_text(json.dumps({'seed': seed, 'batch': 8, 'epochs_run': 100, **done}))
    (d / 'best.json').write_text(json.dumps({'epoch': 90, 'val_miou': 83.0}))
    if last_miou is not None:
        write_model(d / 'last', {'model': {'weights': 'last', 'epoch': 99}}, last_miou, deferred)


def test_summarize_and_by_schedule(tmp_path):
    e100 = {'epochs': 100, 'patience': 100, 'lr_schedule': 'constant'}
    fake_run(tmp_path / 'a', 0, e100, 80.0, 0.05, last_miou=81.0)
    fake_run(tmp_path / 'b', 1, e100, 82.0, 0.06, last_miou=83.0)
    fake_run(tmp_path / 'c', 0, {}, 79.0, 0.07)
    (tmp_path / 'unfinished').mkdir()
    df = R.summarize({'a': tmp_path / 'a', 'b': tmp_path / 'b', 'c': tmp_path / 'c', 'x': tmp_path / 'unfinished'})
    assert df.attrs['skipped'] == ['x']
    assert list(zip(df['run'], df['selection'], df['epoch'])) == [
        ('a', 'best val', 90), ('a', 'last epoch', 99), ('b', 'best val', 90), ('b', 'last epoch', 99),
        ('c', 'best val', 90)]
    assert df.set_index(['run', 'selection']).loc[('b', 'best val'), 'deferred'] == 0.06  # conformal_3way, not plug_in
    s = R.by_schedule(df).set_index(['schedule', 'batch', 'selection'])
    assert s.loc[('100 epochs fixed', 8, 'best val'), 'n'] == 2
    assert s.loc[('100 epochs fixed', 8, 'best val'), 'mean'] == pytest.approx(81.0)
    assert s.loc[('100 epochs fixed', 8, 'best val'), 'std'] == pytest.approx(2 ** 0.5)
    assert s.loc[('100 epochs fixed', 8, 'last epoch'), 'mean'] == pytest.approx(82.0)
    assert s.loc[('early stopping (30)', 8, 'best val'), 'n'] == 1


def test_summary_of_nothing():
    df = R.summarize({})
    assert df.empty and df.attrs['skipped'] == []
    assert R.by_schedule(df).empty
