"""Step 2 - decision layer with PLS: split-conformal deferral and per-population bias correction, T = 14/15/16 %FW.

PLS is trained on Cal and the 90% conformal half-width is calibrated on Tuning fruit, so Tuning stays out of training
here (Step 1 trains on Cal + Tuning). Decisions and metrics are per fruit (mean prediction of its duplicate spectra).
Run from the repo root:  .venv/Scripts/python -m src.step2_decision
Writes results/step2_decision_metrics.csv, results/step2_decision.json and the 'step2' entry of results/summary.json.
"""
import json

import numpy as np
import pandas as pd

from src.data import ROOT, TARGET, fruit_table, load_data, set_masks, wavelength_columns, wavelengths
from src.decision import decision_metrics
from src.models_pls import fit_pls, predict
from src.preprocess import preprocess
from src.uncertainty import SpectralOOD, conformal_quantile, repeated_bias_correct

RESULTS = ROOT / 'results'
N_COMPONENTS = 11
ALPHA = 0.1  # 90% interval
# Illustrative thresholds. AMIA recommends a 15 %FW minimum for most Australian cultivars (cited in Anderson et al. 2020).
THRESHOLDS = (14.0, 15.0, 16.0)
KS = (5, 10)  # reference fruit measured destructively per population
N_REP = 30
SEED = 0
TEST_SETS = ['Val Ext', 'Val Ext 2']
REP_SD_METRICS = ['interval_coverage', 'decided', 'accuracy_decided', 'false_ready_decided']

# pilot_dss.py output at T = 15 (same seed and draw order), formatted as the pilot prints it.
PILOT_FORMATS = [('accuracy_no_defer', '.1%'), ('false_ready_no_defer', '.1%'), ('interval_coverage', '.0%'),
                 ('decided', '.0%'), ('accuracy_decided', '.1%'), ('false_ready_decided', '.1%')]
PILOT_T15 = {
    ('Val Ext', 'none'): '93.0% 3.9% 76% 72% 99.6% 0.2%',
    ('Val Ext', 'k=5'): '94.9% 2.4% 91% 71% 99.7% 0.2%',
    ('Val Ext', 'k=10'): '95.1% 2.1% 93% 71% 99.8% 0.1%',
    ('Val Ext 2', 'none'): '61.3% 0.0% 19% 46% 71.6% 0.0%',
    ('Val Ext 2', 'k=5'): '82.8% 14.3% 85% 53% 95.1% 4.9%',
    ('Val Ext 2', 'k=10'): '83.8% 13.1% 86% 51% 95.9% 4.1%',
}
PILOT_Q = '1.26'
PILOT_OOD_Q = {'Val Ext': '21%', 'Val Ext 2': '24%'}
# Our q is the standard split-conformal rank statistic (1.2555); the pilot interpolates with np.quantile (1.2558).
# Exactly one fruit lies in between, so it is decided here and deferred in the pilot.
PILOT_KNOWN_DIFF = {
    'Val Ext 2, none, T=15': ('61.3% 0.0% 19% 47% 72.0% 0.0%',
                              'one fruit (pop 129, true DMC 17.2, |p - 15| = 1.2557) lies between our q = 1.2555 '
                              '(standard rank) and the pilot q = 1.2558 (interpolated): decided ready here (correct), '
                              'deferred in the pilot'),
}

SHORT = {'truly_ready': 'ready', 'accuracy_no_defer': 'acc', 'false_ready_no_defer': 'false_rdy', 'interval_coverage': 'cov90',
         'decided': 'decided', 'accuracy_decided': 'acc_dec', 'false_ready_decided': 'false_rdy_dec'}


def evaluate(df, quantile=conformal_quantile):
    X, _ = preprocess(df[wavelength_columns(df)].to_numpy(), wavelengths(df))
    y = df[TARGET].to_numpy()
    masks = set_masks(df)
    model = fit_pls(X[masks['Cal']], y[masks['Cal']], N_COMPONENTS)
    fruit = fruit_table(df, predict(model, X))
    tuning = fruit[fruit['Set'] == 'Tuning']
    q = quantile(tuning['p'] - tuning['y'], ALPHA)
    ood = SpectralOOD(model, X[masks['Cal']])

    rng = np.random.default_rng(SEED)
    rows, ood_rates, pop_bias = [], {}, {}
    for s in TEST_SETS:
        ood_rates[s] = {name: float(f.mean()) for name, f in ood.flags(X[masks[s]]).items()}
        sub = fruit[fruit['Set'] == s]
        pop_bias[s] = {int(pop): float((g['p'] - g['y']).mean()) for pop, g in sub.groupby('Pop')}
        tables = {'none': sub}
        for k in KS:  # same draw order as pilot_dss.py
            tables[f'k={k}'] = repeated_bias_correct(sub, k, N_REP, rng)
        for corr, t in tables.items():
            repeated = 'rep' in t
            for T in THRESHOLDS:
                row = {'set': s, 'correction': corr, 'T': T, 'half_width': q, 'n_rep': N_REP if repeated else 1,
                       **decision_metrics(t['y'], t['p'], q, T)}
                if repeated:  # pooled over repetitions as in the pilot; SD across repetitions alongside
                    per_rep = pd.DataFrame([decision_metrics(g['y'], g['p'], q, T) for _, g in t.groupby('rep')])
                    row.update({f'{m}_sd': float(per_rep[m].std(ddof=1)) for m in REP_SD_METRICS})
                rows.append(row)
    return {'half_width': q, 'n_tuning_fruit': int(len(tuning)), 'ood_spectra_flagged': ood_rates,
            'population_bias': pop_bias, 'rows': rows}


def pilot_comparison(res, known=PILOT_KNOWN_DIFF):
    """Compare with the pilot output. A mismatch counts as explained only if our value is exactly the documented one."""
    out = [{'item': 'half-width q', 'ours': f'{res["half_width"]:.2f}', 'pilot': PILOT_Q}]
    for s, v in PILOT_OOD_Q.items():
        out.append({'item': f'{s}, spectra flagged by Q-residual', 'ours': f'{res["ood_spectra_flagged"][s]["q_residual"]:.0%}', 'pilot': v})
    for (s, corr), pilot in PILOT_T15.items():
        r = next(r for r in res['rows'] if r['set'] == s and r['correction'] == corr and r['T'] == 15.0)
        out.append({'item': f'{s}, {corr}, T=15', 'ours': ' '.join(f'{r[m]:{fmt}}' for m, fmt in PILOT_FORMATS), 'pilot': pilot})
    for c in out:
        c['match'] = c['ours'] == c['pilot']
        if not c['match'] and c['item'] in known and known[c['item']][0] == c['ours']:
            c['explained'] = known[c['item']][1]
    return out


def pilot_reproduced(comparison):
    return all(c['match'] or 'explained' in c for c in comparison)


def main():
    res = evaluate(load_data())
    rows = pd.DataFrame(res['rows'])
    print(f'90% split-conformal half-width q = {res["half_width"]:.3f} %FW '
          f'({res["n_tuning_fruit"]} Tuning fruit, PLS {N_COMPONENTS} LV trained on Cal)')
    for s in TEST_SETS:
        o, b = res['ood_spectra_flagged'][s], res['population_bias'][s]
        print(f'\n[{s}] {rows.loc[rows.set == s, "n_fruit"].iloc[0]} fruit, {len(b)} populations | population bias (p - y) '
              f'{min(b.values()):+.2f} to {max(b.values()):+.2f} | spectra flagged OOD: Q {o["q_residual"]:.0%}, T2 {o["hotelling_t2"]:.0%}')
        t = rows[rows.set == s].set_index(['T', 'correction'])[list(SHORT)].rename(columns=SHORT)
        print((t * 100).round(1).to_string())

    comparison = pilot_comparison(res)
    all_match = pilot_reproduced(comparison)
    print('\nComparison with pilot_dss.py (T = 15):')
    for c in comparison:
        status = 'KHỚP' if c['match'] else ('LỆCH, đã giải thích' if 'explained' in c else 'LỆCH')
        print(f'  {c["item"]:<38} ours {c["ours"]:<34} pilot {c["pilot"]:<34} {status}')
        if 'explained' in c:
            print(f'    -> {c["explained"]}')

    config = {'n_components': N_COMPONENTS, 'train_set': 'Cal', 'calibration_set': 'Tuning', 'alpha': ALPHA,
              'thresholds': list(THRESHOLDS), 'k_reference_fruit': list(KS), 'n_rep': N_REP, 'seed': SEED,
              'ood_limit': '95th percentile of Cal spectra'}
    RESULTS.mkdir(exist_ok=True)
    rows.to_csv(RESULTS / 'step2_decision_metrics.csv', index=False)
    detail = {'config': config, **{k: v for k, v in res.items() if k != 'rows'},
              'pilot_comparison': comparison, 'pilot_reproduced': all_match}
    (RESULTS / 'step2_decision.json').write_text(json.dumps(detail, indent=2), encoding='utf-8')
    summary_path = RESULTS / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    summary['step2'] = {
        'description': 'Fruit-level decisions: ready / not ready / defer with a 90% split-conformal interval, '
                       'optional per-population bias correction with k destructive reference fruit',
        'config': config,
        'half_width': res['half_width'],
        'ood_spectra_flagged': res['ood_spectra_flagged'],
        'metrics': res['rows'],
        'pilot_reproduced': all_match,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nPilot reproduced (exact or documented difference): {all_match}. Results written to {RESULTS}')


if __name__ == '__main__':
    main()
