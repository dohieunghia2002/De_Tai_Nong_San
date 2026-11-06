"""Further training runs to explain the gap between our pilot run 2 (test mIoU 79.25, one seed) and the paper (82.90),
whose released checkpoint our evaluation reproduces exactly (src.weeds.authors). The authors' training code differs
from ours only in its schedule: 100 epochs without early stopping, checkpoints chosen on validation. So the runs
measure seed variance under both schedules, plus one variant with a decaying learning rate. All use batch 8 and the
5-band input. notebooks/colab_weeds_runs.ipynb trains them in the order below, as many as fit in a Colab session,
evaluates each on test, runs the decision layer, and resumes the rest in the next session.
"""
import json
from pathlib import Path

import pandas as pd

from src.weeds.analyze import PRIMARY, pick
from src.weeds.train import parser

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


def train_args(run):
    """Command-line arguments of src.weeds.train for one run (data, output and budget are added by the caller)."""
    return ['--batch', str(BATCH), '--seed', str(run['seed']), '--epochs', str(run['epochs']),
            '--patience', str(run['patience']), '--lr-schedule', run['lr_schedule']]


def schedule(done):
    """Schedule label from a train_done.json; runs written before these fields existed used the parser defaults."""
    p = parser()
    epochs, patience = done.get('epochs', p.get_default('epochs')), done.get('patience', p.get_default('patience'))
    lr = done.get('lr_schedule', p.get_default('lr_schedule'))
    if lr != 'constant':
        return f'{epochs} epochs, {lr} lr'
    return f'{epochs} epochs fixed' if patience >= epochs else f'early stopping ({patience})'


def summarize(run_dirs):
    """One row per finished run (train_done, segmentation_metrics, decision_metrics present): schedule, seed, best
    epoch, validation and test mIoU, and the decisions of conformal_3way in the primary setting."""
    rows = []
    for name, d in run_dirs.items():
        d = Path(d)
        files = [d / f for f in ('train_done.json', 'best.json', 'segmentation_metrics.json', 'decision_metrics.csv')]
        if not all(f.exists() for f in files):
            continue
        done, best, seg = (json.loads(f.read_text()) for f in files[:3])
        r = pick(pd.read_csv(files[3]), policy=PRIMARY['policy'],
                 **{k: PRIMARY[k] for k in ('cell_size', 'tau', 'scope', 'calibration')})
        rows.append({'run': name, 'schedule': schedule(done), 'seed': done['seed'], 'epochs_run': done['epochs_run'],
                     'best_epoch': best['epoch'], 'val_miou': best['val_miou'], 'test_miou': seg['test']['miou'],
                     'test_crop': seg['test']['crop'], 'test_weed': seg['test']['weed'],
                     'spray_reduction': r['spray_reduction'], 'deferred': r['deferred'],
                     'missed_need_rate': r['missed_need_rate'], 'false_spray_rate': r['false_spray_rate']})
    return pd.DataFrame(rows)


def by_schedule(df):
    """Count, mean, sample standard deviation (NaN for a single run) and range of test mIoU per schedule."""
    return (df.groupby('schedule')['test_miou'].agg(n='count', mean='mean', std='std', min='min', max='max')
            .reset_index())
