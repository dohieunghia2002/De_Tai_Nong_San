"""Step 3 analysis - RMSEP of the ANN / 1D-CNN runs against Walsh et al. 2024, then the Step 2 decision layer with the CNN.

Reads the predictions written by src.step3_train_nn. Light computation: runs locally or at the end of the Colab notebook.
  python -m src.step3_analysis [--runs-dir results/step3]
Writes results/step3_regression.csv (per run, seed and set), results/step3_decision_metrics.csv (per seed),
results/step3.json and the 'step3' entry of results/summary.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ROOT, TARGET, load_data, set_masks
from src.models_pls import regression_metrics
from src.step2_decision import evaluate_predictions

RESULTS = ROOT / 'results'
DECISION_RUN = 'cnn_AT_calonly'
EVAL_SETS = ['Tuning', 'Val Ext', 'Val Ext 2']
EQUIVALENT_TOL = 0.05  # about twice the seed-to-seed SD of RMSEP in Walsh 2024, Table 2 (0.02-0.03 %FW)

# Walsh 2024 models trained like ours: Set partition (KS population), file order, no extra outlier removal,
# standardised inputs. Walsh reports bias as mean(true - pred); converted here to our mean(pred - true).
BENCHMARKS = {
    'ann_S_cal': {'source': 'Walsh 2024, Table 5, model A026', 'rmsep': 0.96, 'bias': 0.01},
    'ann_S_all': {'source': 'Walsh 2024, Tables 1 and 3, model A002 (Anderson 2020 ANN: 0.89)', 'rmsep': 0.89, 'bias': -0.08},
    'cnn_TA_cal': {'source': 'Walsh 2024, Table 5, model C028', 'rmsep': 0.82, 'bias': 0.01},
}
REFERENCE_NOTE = ('Not comparable, reference only: Walsh 2024 best models used the extra outlier removal of Mishra & Passos '
                  '2021 and/or their random sample partition. CNN C134: season 4 RMSEP 0.77, season 5 RMSEP 1.18 '
                  '(bias -0.16 in our convention); ANN A010: 0.88 and 2.62; PLS P076: 1.00 and 1.87.')
DECISION_METRICS = ['half_width', 'truly_ready', 'accuracy_no_defer', 'false_ready_no_defer', 'interval_coverage',
                    'decided', 'accuracy_decided', 'false_ready_decided', 'missed_ready_decided']


def records(table):
    """DataFrame rows as dicts, with NaN (e.g. the SD of a single seed) as None so the JSON stays valid."""
    return table.astype(object).where(table.notna(), None).to_dict('records')


def load_runs(runs_dir, n_rows):
    """{run: {seed: per-spectrum predictions}} from <runs_dir>/preds/<run>_seed<k>.csv"""
    runs = {}
    for f in sorted((Path(runs_dir) / 'preds').glob('*_seed*.csv')):
        run, seed = f.stem.rsplit('_seed', 1)
        p = pd.read_csv(f)
        if len(p) != n_rows:
            raise ValueError(f'{f}: {len(p)} rows, expected {n_rows}')
        pred = np.full(n_rows, np.nan)
        pred[p['row'].to_numpy()] = p['pred'].to_numpy()
        runs.setdefault(run, {})[int(seed)] = pred
    return runs


def regression_table(df, runs):
    y = df[TARGET].to_numpy()
    masks = set_masks(df)
    return pd.DataFrame([{'run': run, 'seed': seed, 'set': s, **regression_metrics(y[masks[s]], pred[masks[s]])}
                         for run, seeds in runs.items() for seed, pred in sorted(seeds.items()) for s in EVAL_SETS])


def summarise_regression(reg):
    g = reg.groupby(['run', 'set'])
    out = g.agg(n_seeds=('seed', 'nunique'), rmsep_mean=('rmsep', 'mean'), rmsep_sd=('rmsep', 'std'),
                rmsep_min=('rmsep', 'min'), rmsep_max=('rmsep', 'max'), bias_mean=('bias', 'mean'), sep_mean=('sep', 'mean'))
    return out.reset_index()


def benchmark_comparison(summary):
    out = []
    for run, b in BENCHMARKS.items():
        row = summary[(summary['run'] == run) & (summary['set'] == 'Val Ext')]
        if row.empty:
            continue
        r = row.iloc[0]
        diff = float(r['rmsep_mean'] - b['rmsep'])
        out.append({'run': run, **b, 'rmsep_mean': float(r['rmsep_mean']), 'rmsep_sd': float(r['rmsep_sd']),
                    'bias_mean': float(r['bias_mean']), 'n_seeds': int(r['n_seeds']), 'rmsep_diff': diff,
                    'equivalent': bool(abs(diff) <= EQUIVALENT_TOL)})
    return out


def decision_table(df, seeds):
    rows = []
    for seed, pred in sorted(seeds.items()):
        res = evaluate_predictions(df, pred)
        rows += [{'seed': seed, **r} for r in res['rows']]
    return pd.DataFrame(rows)


def summarise_decision(dec):
    g = dec.groupby(['set', 'correction', 'T'], sort=False)[DECISION_METRICS]
    mean, sd = g.mean().add_suffix('_mean'), g.std().add_suffix('_sd')
    return pd.concat([mean, sd], axis=1).reset_index().assign(n_seeds=dec['seed'].nunique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', type=Path, default=RESULTS / 'step3')
    args = ap.parse_args()

    df = load_data()
    runs = load_runs(args.runs_dir, len(df))
    if not runs:
        raise SystemExit(f'no predictions found in {args.runs_dir / "preds"}')
    reg = regression_table(df, runs)
    summary = summarise_regression(reg)

    print('RMSEP and bias (pred - true) in %FW, mean (SD) over seeds:')
    for _, r in summary.iterrows():
        print(f'  {r["run"]:<15} {r["set"]:<10} seeds {r["n_seeds"]} | RMSEP {r["rmsep_mean"]:.3f} ({r["rmsep_sd"]:.3f}), '
              f'range {r["rmsep_min"]:.3f}-{r["rmsep_max"]:.3f} | bias {r["bias_mean"]:+.3f}')
    comparison = benchmark_comparison(summary)
    print(f'\nSeason 4 vs comparable Walsh 2024 models (equivalent if |diff| <= {EQUIVALENT_TOL}):')
    for c in comparison:
        status = 'TƯƠNG ĐƯƠNG' if c['equivalent'] else 'KHÁC'
        print(f'  {c["run"]:<15} {c["rmsep_mean"]:.3f} vs {c["rmsep"]:.2f} ({c["source"]}), diff {c["rmsep_diff"]:+.3f} -> {status}')
    print(f'  {REFERENCE_NOTE}')

    decision = None
    if DECISION_RUN in runs:
        dec = decision_table(df, runs[DECISION_RUN])
        decision = summarise_decision(dec)
        cols = ['interval_coverage', 'decided', 'accuracy_decided', 'false_ready_decided']
        print(f'\nDecision layer with {DECISION_RUN}, T = 15, mean (SD) over {dec["seed"].nunique()} seeds, %:')
        for _, r in decision[decision['T'] == 15.0].iterrows():
            vals = ' | '.join(f'{c} {100 * r[c + "_mean"]:.1f} ({100 * r[c + "_sd"]:.1f})' for c in cols)
            print(f'  {r["set"]:<10} {r["correction"]:<5} q {r["half_width_mean"]:.3f} | {vals}')
        dec.to_csv(RESULTS / 'step3_decision_metrics.csv', index=False)

    reg.to_csv(RESULTS / 'step3_regression.csv', index=False)
    detail = {'runs': {run: sorted(seeds) for run, seeds in runs.items()}, 'equivalent_tolerance': EQUIVALENT_TOL,
              'regression_summary': records(summary), 'benchmark_comparison': comparison,
              'reference_note': REFERENCE_NOTE, 'decision_run': DECISION_RUN,
              'decision_summary': None if decision is None else records(decision)}
    (RESULTS / 'step3.json').write_text(json.dumps(detail, indent=2), encoding='utf-8')
    summary_path = RESULTS / 'summary.json'
    all_summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    all_summary['step3'] = {
        'description': 'ANN and 1D-CNN (Walsh 2024 settings) on the fixed Set partition; spectrum-level RMSEP over seeds; '
                       f'decision layer of Step 2 with {DECISION_RUN} (Tuning set unused in training)',
        **{k: detail[k] for k in ('runs', 'regression_summary', 'benchmark_comparison', 'decision_summary')},
    }
    summary_path.write_text(json.dumps(all_summary, indent=2), encoding='utf-8')
    print(f'\nResults written to {RESULTS}')


if __name__ == '__main__':
    main()
