"""Further training runs to explain the gap between our pilot run 2 (test mIoU 79.25, one seed) and the paper (82.90),
whose released checkpoint our evaluation reproduces exactly (src.weeds.authors). The authors' training code differs
from ours only in its schedule: 100 epochs without early stopping, checkpoints chosen on validation. So the runs
measure seed variance under both schedules, plus one variant with a decaying learning rate. All use batch 8 and the
5-band input. Each run is evaluated twice: with the model chosen on validation (best.pt), and with the final-epoch
model, because choosing the best of many noisy validation scores may pick a lucky epoch (run 2: validation 83.94,
test 79.25; the authors' released model: validation 79.53, test 82.90).
notebooks/colab_weeds_runs.ipynb trains them in the order below, as many as fit in a Colab session, evaluates each,
runs the decision layer, and resumes the rest in the next session.
"""
import json
from pathlib import Path

import pandas as pd

from src.weeds.analyze import PRIMARY, pick

BATCH = 8
RUNS = [  # in order of priority
    # the authors' schedule (github.com/GFZ/weedsgalore README: --epochs 100), constant learning rate
    {'name': 'e100_seed0', 'seed': 0, 'epochs': 100, 'patience': 100, 'lr_schedule': 'constant'},
    {'name': 'e100_seed1', 'seed': 1, 'epochs': 100, 'patience': 100, 'lr_schedule': 'constant'},
    {'name': 'e100_seed2', 'seed': 2, 'epochs': 100, 'patience': 100, 'lr_schedule': 'constant'},
    # 100 epochs with the learning rate decayed to zero: a steadier end of training (not in the authors' code)
    {'name': 'e100_cosine_seed0', 'seed': 0, 'epochs': 100, 'patience': 100, 'lr_schedule': 'cosine'},
    # our pilot schedule (run 2 = seed 0): early stopping after 30 epochs without improvement, at most 200
    {'name': 'es_seed1', 'seed': 1, 'epochs': 200, 'patience': 30, 'lr_schedule': 'constant'},
    {'name': 'es_seed2', 'seed': 2, 'epochs': 200, 'patience': 30, 'lr_schedule': 'constant'},
]
# what run 2 used: its train_done.json predates these fields (src.weeds.train defaults on 2026-11-04)
LEGACY = {'epochs': 200, 'patience': 30, 'lr_schedule': 'constant'}


def train_args(run):
    """Command-line arguments of src.weeds.train for one run (data, output and budget are added by the caller)."""
    return ['--batch', str(BATCH), '--seed', str(run['seed']), '--epochs', str(run['epochs']),
            '--patience', str(run['patience']), '--lr-schedule', run['lr_schedule']]


def fixed(run):
    """True when early stopping can never fire before the last epoch."""
    return run['patience'] >= run['epochs']


def schedule(done):
    """Schedule label from a train_done.json (or a RUNS entry)."""
    s = {**LEGACY, **{k: done[k] for k in LEGACY if k in done}}
    if s['lr_schedule'] != 'constant':
        return f"{s['epochs']} epochs, {s['lr_schedule']} lr"
    return f"{s['epochs']} epochs fixed" if fixed(s) else f"early stopping ({s['patience']})"


def decision_row(csv):
    r = pick(pd.read_csv(csv), policy=PRIMARY['policy'],
             **{k: PRIMARY[k] for k in ('cell_size', 'tau', 'scope', 'calibration')})
    return {k: r[k] for k in ('spray_reduction', 'deferred', 'missed_need_rate', 'false_spray_rate')}


def summarize(run_dirs):
    """One row per evaluated model of each finished run: 'best val' (best.pt) and, when <run>/last/ was evaluated,
    'last epoch'. Columns: schedule, batch, seed, epoch of the model, validation and test mIoU, and the decisions of
    conformal_3way in the primary setting. Runs without results are listed in df.attrs['skipped']."""
    rows, skipped = [], []
    for name, d in run_dirs.items():
        d = Path(d)
        need = [d / f for f in ('train_done.json', 'best.json', 'segmentation_metrics.json', 'decision_metrics.csv')]
        if not all(f.exists() for f in need):
            skipped.append(name)
            continue
        done, best = json.loads(need[0].read_text()), json.loads(need[1].read_text())
        base = {'run': name, 'schedule': schedule(done), 'batch': done['batch'], 'seed': done['seed'],
                'epochs_run': done['epochs_run']}
        models = [('best val', best['epoch'], d)]
        last = d / 'last'
        if (last / 'segmentation_metrics.json').exists() and (last / 'decision_metrics.csv').exists():
            models.append(('last epoch', json.loads((last / 'segmentation_metrics.json').read_text())['model']['epoch'],
                           last))
        for selection, epoch, m in models:
            seg = json.loads((m / 'segmentation_metrics.json').read_text())
            rows.append({**base, 'selection': selection, 'epoch': epoch, 'val_miou': seg['val']['miou'],
                         'test_miou': seg['test']['miou'], 'test_crop': seg['test']['crop'],
                         'test_weed': seg['test']['weed'], **decision_row(m / 'decision_metrics.csv')})
    df = pd.DataFrame(rows)
    df.attrs['skipped'] = skipped
    return df


def by_schedule(df):
    """Count, mean, sample standard deviation (NaN for a single run) and range of test mIoU per schedule, batch
    size and model selection."""
    keys = ['schedule', 'batch', 'selection']
    if df.empty:
        return pd.DataFrame(columns=keys + ['n', 'mean', 'std', 'min', 'max'])
    return df.groupby(keys)['test_miou'].agg(n='count', mean='mean', std='std', min='min', max='max').reset_index()
