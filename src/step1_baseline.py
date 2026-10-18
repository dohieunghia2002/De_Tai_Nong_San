"""Step 1 - reproduce the PLS baseline of Anderson et al. 2020: train on Cal + Tuning, test on seasons 4 and 5.

Run from the repo root:  python -m src.step1_baseline
Writes results/step1_pls_baseline.json, results/step1_predictions.csv and the 'step1' entry of results/summary.json.
"""
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.data import ROOT, TARGET, TRAIN_SETS, EXPECTED_COUNTS, count_table, load_data, set_masks, wavelength_columns, wavelengths
from src.models_pls import fit_pls, predict, regression_metrics
from src.preprocess import PreprocessConfig, preprocess

RESULTS = ROOT / 'results'
N_COMPONENTS = 11
TOL = 0.005  # benchmarks are reported with 2 decimals

# Spectrum-level benchmarks in %FW. Our bias is mean(pred - true); the papers' sign convention is not stated.
BENCHMARKS = {
    'Val Ext': [
        {'source': 'Anderson et al. 2020, Table 6 (PLSR - all, 11 LV)', 'rmsep': 1.01, 'bias': -0.07, 'must_match': True},
    ],
    'Val Ext 2': [
        {'source': 'pilot_dss.py (same method)', 'rmsep': 1.83, 'bias': -1.48, 'must_match': True},
        {'source': 'Walsh et al. 2024, best PLS (different tuning)', 'rmsep': 1.87, 'bias': 1.56, 'must_match': False},
    ],
}


def compare(metrics, bench):
    out = dict(bench)
    out['rmsep_diff'] = metrics['rmsep'] - bench['rmsep']
    out['bias_diff'] = metrics['bias'] - bench['bias']
    out['rmsep_match'] = bool(abs(out['rmsep_diff']) <= TOL)
    return out


def main():
    df = load_data()  # raises if shape or per-set counts are off
    counts = count_table(df)
    cfg = PreprocessConfig()
    X, wl = preprocess(df[wavelength_columns(df)].to_numpy(), wavelengths(df), cfg)
    y = df[TARGET].to_numpy()
    masks = set_masks(df)
    train = np.isin(df['Set'], TRAIN_SETS)

    print('Counts (spectra / fruit) vs expected:')
    for s, (n_spec, n_fruit) in EXPECTED_COUNTS.items():
        print(f'  {s:<10} {counts.loc[s, "spectra"]:>6} / {counts.loc[s, "fruit"]:>5}   expected {n_spec:>6} / {n_fruit:>5}')
    print(f'Variables after trim: {X.shape[1]} ({wl[0]}-{wl[-1]} nm) | training spectra: {train.sum()}')

    model = fit_pls(X[train], y[train], N_COMPONENTS)
    test, preds = {}, []
    for s, benches in BENCHMARKS.items():
        m = masks[s]
        p = predict(model, X[m])
        metrics = regression_metrics(y[m], p)
        metrics['benchmarks'] = [compare(metrics, b) for b in benches]
        test[s] = metrics
        preds.append(pd.DataFrame({'row': np.flatnonzero(m), 'Set': s, 'fruit': df['fruit'].to_numpy()[m],
                                   'Pop': df['Pop'].to_numpy()[m], 'DM': y[m], 'pred': p}))
        print(f'\n[{s}] PLS {N_COMPONENTS} LV, {metrics["n"]} spectra: RMSEP {metrics["rmsep"]:.3f}, bias {metrics["bias"]:+.3f}, '
              f'SEP {metrics["sep"]:.3f}, R2 {metrics["r2"]:.3f}')
        for b in metrics['benchmarks']:
            status = ('KHỚP' if b['rmsep_match'] else 'LỆCH') if b['must_match'] else 'tham khảo'
            print(f'  vs {b["source"]}: RMSEP {b["rmsep"]:.2f} (diff {b["rmsep_diff"]:+.3f}), '
                  f'bias {b["bias"]:+.2f} (diff {b["bias_diff"]:+.3f}) -> {status}')

    all_match = all(b['rmsep_match'] for t in test.values() for b in t['benchmarks'] if b['must_match'])
    result = {
        'preprocess': asdict(cfg),
        'n_variables': int(X.shape[1]),
        'n_components': N_COMPONENTS,
        'train_sets': TRAIN_SETS,
        'n_train_spectra': int(train.sum()),
        'counts': {s: {'spectra': int(counts.loc[s, 'spectra']), 'fruit': int(counts.loc[s, 'fruit'])} for s in counts.index},
        'test': test,
        'all_required_benchmarks_match': all_match,
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'step1_pls_baseline.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.concat(preds).to_csv(RESULTS / 'step1_predictions.csv', index=False)
    summary_path = RESULTS / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    summary['step1'] = {
        'description': 'PLS 11 LV on Cal+Tuning, SG d2 w17 p2 on full spectrum, 684-990 nm; spectrum-level metrics',
        **{s: {k: t[k] for k in ('n', 'rmsep', 'bias', 'sep', 'r2')} for s, t in test.items()},
        'all_required_benchmarks_match': all_match,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nAll required benchmarks match: {all_match}. Results written to {RESULTS}')


if __name__ == '__main__':
    main()
