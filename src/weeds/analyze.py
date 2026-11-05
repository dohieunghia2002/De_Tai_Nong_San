"""Pilot decision analysis and go / no-go check (numpy and pandas only; runs on Colab or locally).

  python -m src.weeds.analyze --data <dir with weedsgalore_<split>.npz> --out <dir with probs_*.npz, segmentation_metrics.json>

The validation split already chose the model (best epoch, early stopping), so it is not independent of the model.
Primary calibration is therefore 2-fold on the test split by field location: thresholds from 4 locations decide the
other 4, and vice versa. Calibration on the validation split is reported as a secondary variant.
The weed-cover thresholds tau are illustrative until replaced by sourced action thresholds.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.weeds.data import (DATES, EXPECTED_TEST_LOCATIONS, PAPER_TEST_3CLASS_MSI, PRE_HERBICIDE_DATES,
                            load_npz, location_blocks, to_3class)
from src.weeds.decision import cell_table, evaluate_crossfit, evaluate_policies

CELL_SIZES = (100, 200)                # 0.25 m and 0.5 m cells at 2.5 mm GSD
TAUS = (0.001, 0.005, 0.01, 0.02)      # illustrative weed-cover fractions that make a cell need spraying
ALPHA = BETA = 0.1
CAL_TEST_2FOLD = 'test 2-fold by location'
CAL_VAL = 'val (also used for model selection)'
PRIMARY = {'cell_size': 200, 'tau': 0.01, 'policy': 'conformal_3way', 'scope': 'pre-herbicide pooled',
           'calibration': CAL_TEST_2FOLD}
# Proposed pilot criteria (agreed before running; adjustable)
CRITERIA = {'miou_min': PAPER_TEST_3CLASS_MSI['miou'] - 5, 'spray_reduction_min': 0.30, 'deferred_max': 0.30}


def load_cells(data_dir, out_dir, split, cell):
    d = load_npz(Path(data_dir) / f'weedsgalore_{split}.npz')
    with np.load(Path(out_dir) / f'probs_{split}.npz') as p:
        ids, p_weed = p['ids'].astype(str), p['p_weed']
    n = len(ids)
    if not np.array_equal(ids, d['ids'][:n]):
        raise ValueError(f'{split}: probabilities and labels are not in the same tile order')
    return cell_table(ids, to_3class(d['y'][:n]), p_weed, cell)


def test_location_folds(test_ids):
    """Fold 0 / 1 per test tile, alternating over the 8 field locations (4 locations per fold)."""
    blocks = location_blocks(test_ids)
    if len(blocks) != EXPECTED_TEST_LOCATIONS:
        raise ValueError(f'expected {EXPECTED_TEST_LOCATIONS} test locations, found {len(blocks)}')
    return {t: i % 2 for i, b in enumerate(blocks) for t in b}


def run(data_dir, out_dir):
    rows = []
    scopes = [('pre-herbicide pooled', PRE_HERBICIDE_DATES)] + [(d, (d,)) for d in DATES]
    for cell in CELL_SIZES:
        val_all, test_all = load_cells(data_dir, out_dir, 'val', cell), load_cells(data_dir, out_dir, 'test', cell)
        fold_of = test_location_folds(list(dict.fromkeys(test_all['tile'])))
        for tau in TAUS:
            for scope, dates in scopes:
                t = test_all[test_all['date'].isin(dates)]
                if t.empty:
                    continue
                base = {'cell_size': cell, 'scope': scope}
                rows += [{**base, 'calibration': CAL_TEST_2FOLD, **r}
                         for r in evaluate_crossfit(t, t['tile'].map(fold_of).to_numpy(), tau, ALPHA, BETA)]
                v = val_all[val_all['date'].isin(dates)]
                if not v.empty:
                    rows += [{**base, 'calibration': CAL_VAL, **r} for r in evaluate_policies(v, t, tau, ALPHA, BETA)]
    return pd.DataFrame(rows)


def pick(df, **keys):
    m = np.ones(len(df), dtype=bool)
    for k, v in keys.items():
        m &= (df[k] == v).to_numpy()
    if m.sum() != 1:
        raise ValueError(f'expected exactly one row for {keys}, found {m.sum()}')
    return df[m].iloc[0]


def go_no_go(df, seg):
    key = {k: PRIMARY[k] for k in ('cell_size', 'tau', 'scope', 'calibration')}
    r = pick(df, policy=PRIMARY['policy'], **key)
    oracle = pick(df, policy='oracle', **key)
    miou = seg['test']['miou']
    checks = {
        'C1_test_miou': {'value': miou, 'required': f">= {CRITERIA['miou_min']:.2f}",
                         'pass': bool(miou >= CRITERIA['miou_min'])},
        'C2_spray_reduction': {'value': r['spray_reduction'], 'required': f">= {CRITERIA['spray_reduction_min']}",
                               'pass': bool(r['spray_reduction'] >= CRITERIA['spray_reduction_min'])},
        'C3_deferred': {'value': r['deferred'], 'required': f"<= {CRITERIA['deferred_max']}",
                        'pass': bool(r['deferred'] <= CRITERIA['deferred_max'])},
    }
    info = {
        'primary_setting': PRIMARY, 'alpha': ALPHA, 'beta': BETA,
        'missed_need_rate': r['missed_need_rate'], 'false_spray_rate': r['false_spray_rate'],
        'infestation': r['infestation'], 'oracle_spray_reduction': oracle['spray_reduction'],
        'c2_reachable': bool(oracle['spray_reduction'] >= CRITERIA['spray_reduction_min']),
        'n_cal_need': int(r['n_cal_need']), 'n_cal_no_need': int(r['n_cal_no_need']),
        'lam_lo': r['lam_lo'], 'lam_hi': r['lam_hi'],
    }
    return {'verdict': 'GO' if all(c['pass'] for c in checks.values()) else 'NO-GO', 'checks': checks, 'info': info}


def clean(o):
    """JSON-safe: NaN / inf -> None, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    df = run(args.data, out)
    df.to_csv(out / 'decision_metrics.csv', index=False)
    seg = json.loads((out / 'segmentation_metrics.json').read_text())
    verdict = go_no_go(df, seg)
    (out / 'go_no_go.json').write_text(json.dumps(clean(verdict), indent=1))

    cols = ['policy', 'infestation', 'sprayed', 'deferred', 'spray_reduction', 'missed_need_rate', 'false_spray_rate',
            'missed_weed_area', 'crop_area_sprayed']
    same = (df['cell_size'] == PRIMARY['cell_size']) & (df['tau'] == PRIMARY['tau'])
    pd.set_option('display.width', 200)
    for cal in (CAL_TEST_2FOLD, CAL_VAL):
        print(f"\nCell {PRIMARY['cell_size']} px, tau {PRIMARY['tau']}, {PRIMARY['scope']}, calibration: {cal}, "
              f'alpha = beta = {ALPHA}')
        print(df[same & (df['scope'] == PRIMARY['scope']) & (df['calibration'] == cal)][cols].round(3)
              .to_string(index=False))
    per_date = df[same & (df['policy'] == PRIMARY['policy']) & (df['scope'] != PRIMARY['scope'])]
    print('\nPer acquisition date (conformal_3way):')
    print(per_date[['scope', 'calibration', 'n_cal_need', 'infestation', 'deferred', 'spray_reduction',
                    'missed_need_rate', 'false_spray_rate']].round(3).to_string(index=False))
    print('\nGO / NO-GO:', verdict['verdict'])
    for name, c in verdict['checks'].items():
        print(f"  {name}: {c['value']:.3f} (required {c['required']}) -> {'PASS' if c['pass'] else 'FAIL'}")
    i = verdict['info']
    print(f"  validity: missed-need rate {i['missed_need_rate']:.3f} vs alpha {ALPHA}, "
          f"false-spray rate {i['false_spray_rate']:.3f} vs beta {BETA}")
    if not i['c2_reachable']:
        print(f"  note: C2 cannot be met at this tau even by the oracle (oracle reduction {i['oracle_spray_reduction']:.3f})")


if __name__ == '__main__':
    main()
