"""Step 3 - train the ANN and 1D-CNN of Walsh et al. 2024 on the fixed Set partition. Runs on Colab (GPU).

Every run saves the predictions for all 12,011 spectra. Finished runs are skipped, so the script can be restarted
after a Colab disconnect. Seeds form the outer loop, so a partial run still has every configuration for the first seeds.

  python -m src.step3_train_nn --out <dir> [--seeds 0 1 2 3 4] [--runs cnn_AT_calonly ...] [--smoke]
"""
import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import ROOT, TARGET, load_data, set_masks, wavelength_columns, wavelengths
from src.models_nn import predict, train
from src.models_pls import regression_metrics
from src.preprocess import augment, preprocess

DEFAULT_OUT = ROOT / 'results' / 'step3'
SEEDS = (0, 1, 2, 3, 4)

# Pretreatments: S = SG 2nd derivative, 17 points, full spectrum then trim (Anderson 2020, 103 variables);
# TA / AT = six pretreatments concatenated, trimmed before / after pretreating (618 variables). Inputs are always
# standardised per variable (StandardScaler), as in every best model of Walsh 2024.
# Walsh protocol: scaler fit on Cal + Tuning; train on Cal with Tuning monitoring early stopping and learning rate,
# or train on Cal + Tuning with the training loss monitored. Sample order is the file order (Walsh's "chronological").
RUNS = {
    'ann_S_cal': dict(model='ann', pretreatment='S', train='cal', monitor='tuning', scaler_fit='cal+tuning'),
    'ann_S_all': dict(model='ann', pretreatment='S', train='cal+tuning', monitor='loss', scaler_fit='cal+tuning'),
    'cnn_TA_cal': dict(model='cnn', pretreatment='TA', train='cal', monitor='tuning', scaler_fit='cal+tuning'),
    'cnn_AT_cal': dict(model='cnn', pretreatment='AT', train='cal', monitor='tuning', scaler_fit='cal+tuning'),
    'cnn_AT_all': dict(model='cnn', pretreatment='AT', train='cal+tuning', monitor='loss', scaler_fit='cal+tuning'),
    # Decision layer: the Tuning set is not used at all (scaler, early stopping), so it can calibrate the conformal interval.
    'cnn_AT_calonly': dict(model='cnn', pretreatment='AT', train='cal', monitor='loss', scaler_fit='cal'),
}


def features(raw, wl, pretreatment):
    if pretreatment == 'S':
        return preprocess(raw, wl)[0]
    return augment(raw, wl, trim_first=(pretreatment == 'TA'))


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def environment():
    import tensorflow as tf
    return {'python': platform.python_version(), 'tensorflow': tf.__version__, 'keras': getattr(tf.keras, '__version__', None),
            'numpy': np.__version__, 'pandas': pd.__version__, 'gpu': [d.name for d in tf.config.list_physical_devices('GPU')],
            'git_commit': git_commit()}


def run_one(name, cfg, seed, X, df, masks, out, env, smoke=False):
    pred_path = out / 'preds' / f'{name}_seed{seed}.csv'
    if pred_path.exists():
        print(f'{name} seed {seed}: already done, skipped')
        return
    y = df[TARGET].to_numpy().astype('float32').reshape(-1, 1)  # same shape as the network output
    cal, tun = masks['Cal'], masks['Tuning']
    train_rows = np.flatnonzero(cal | tun if cfg['train'] == 'cal+tuning' else cal)
    scaler = StandardScaler().fit(X[cal | tun] if cfg['scaler_fit'] == 'cal+tuning' else X[cal])
    Xs = scaler.transform(X).astype('float32')
    if smoke:
        train_rows = train_rows[:1000]
    val = (Xs[tun], y[tun]) if cfg['monitor'] == 'tuning' else (None, None)
    model, meta = train(cfg['model'], Xs[train_rows], y[train_rows], *val, seed=seed, max_epochs=2 if smoke else None)
    pred = predict(model, Xs)

    quick = {s: regression_metrics(y[masks[s]].ravel(), pred[masks[s]]) for s in ('Tuning', 'Val Ext', 'Val Ext 2')}
    meta.update({'run': name, 'seed': seed, 'config': cfg, 'n_train': int(train_rows.size), 'n_features': int(X.shape[1]),
                 'smoke': smoke, 'environment': env, 'quick_metrics': quick})
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    (out / 'runs' / f'{name}_seed{seed}.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    # predictions last, through a temporary file: an existing CSV marks a finished run
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pred_path.with_suffix('.tmp')
    pd.DataFrame({'row': np.arange(len(pred)), 'pred': pred}).to_csv(tmp, index=False, float_format='%.5f')
    tmp.replace(pred_path)
    print(f'{name} seed {seed}: {meta["epochs"]} epochs (best {meta["best_epoch"]}), {meta["train_seconds"]:.0f} s | '
          f'RMSEP Tuning {quick["Tuning"]["rmsep"]:.3f}, Val Ext {quick["Val Ext"]["rmsep"]:.3f}, '
          f'Val Ext 2 {quick["Val Ext 2"]["rmsep"]:.3f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--runs', nargs='+', default=list(RUNS), choices=list(RUNS))
    ap.add_argument('--smoke', action='store_true', help='2 epochs on 1,000 spectra, seed 0 only: checks the code path')
    args = ap.parse_args()

    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    env = environment()
    print(json.dumps(env))

    df = load_data()
    masks = set_masks(df)
    raw, wl = df[wavelength_columns(df)].to_numpy(), wavelengths(df)
    X = {p: features(raw, wl, p) for p in sorted({RUNS[r]['pretreatment'] for r in args.runs})}
    print({p: x.shape for p, x in X.items()})
    for seed in ([0] if args.smoke else args.seeds):
        for name in args.runs:
            run_one(name, RUNS[name], seed, X[RUNS[name]['pretreatment']], df, masks, args.out, env, args.smoke)


if __name__ == '__main__':
    main()
