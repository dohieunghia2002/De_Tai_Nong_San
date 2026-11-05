"""Step 3 code that runs without TensorFlow: pretreatments, run configurations and the analysis of saved predictions.
The analysis is exercised with PLS predictions standing in for network outputs."""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_filter

from src.data import TARGET, set_masks, wavelength_columns, wavelengths
from src.models_pls import fit_pls, predict, regression_metrics
from src.preprocess import augment, preprocess, snv
from src.step2_decision import evaluate
from src.step3_analysis import decision_table, load_runs, regression_table, summarise_regression
from src.step3_train_nn import RUNS, features


def spectra(df, n=None):
    return df[wavelength_columns(df)].to_numpy()[:n], wavelengths(df)


def test_augment_after_trim_blocks(df):
    raw, wl = spectra(df, 20)
    keep = (wl >= 684) & (wl <= 990)
    A = augment(raw, wl, trim_first=False)
    assert A.shape == (20, 618)
    blocks = np.split(A, 6, axis=1)
    np.testing.assert_allclose(blocks[0], raw[:, keep])
    np.testing.assert_allclose(blocks[1], snv(raw)[:, keep])
    np.testing.assert_allclose(blocks[3], savgol_filter(raw, 13, 2, deriv=2, axis=1)[:, keep])
    np.testing.assert_allclose(blocks[5], savgol_filter(snv(raw), 13, 2, deriv=2, axis=1)[:, keep])


def test_augment_trim_first_blocks(df):
    raw, wl = spectra(df, 20)
    t = raw[:, (wl >= 684) & (wl <= 990)]
    blocks = np.split(augment(raw, wl, trim_first=True), 6, axis=1)
    np.testing.assert_allclose(blocks[1].mean(axis=1), 0, atol=1e-12)  # SNV over the 103 trimmed variables
    np.testing.assert_allclose(blocks[2], savgol_filter(t, 13, 2, deriv=1, axis=1))
    np.testing.assert_allclose(blocks[4], savgol_filter(snv(t), 13, 2, deriv=1, axis=1))


def test_run_configurations(df):
    raw, wl = spectra(df, 5)
    n_features = {'S': 103, 'TA': 618, 'AT': 618}
    for name, cfg in RUNS.items():
        assert cfg['model'] in ('ann', 'cnn') and cfg['train'] in ('cal', 'cal+tuning')
        assert features(raw, wl, cfg['pretreatment']).shape == (5, n_features[cfg['pretreatment']]), name
    # the decision-layer run must not touch the Tuning set, which calibrates the conformal interval
    assert RUNS['cnn_AT_calonly'] == dict(model='cnn', pretreatment='AT', train='cal', monitor='loss', scaler_fit='cal')


@pytest.fixture(scope='module')
def pls_cal_pred(df):
    X, _ = preprocess(*spectra(df))
    cal = set_masks(df)['Cal']
    y = df[TARGET].to_numpy()
    return predict(fit_pls(X[cal], y[cal], 11), X)


def test_analysis_reads_saved_predictions(df, pls_cal_pred, tmp_path):
    (tmp_path / 'preds').mkdir()
    for seed in (0, 1):
        pd.DataFrame({'row': np.arange(len(df)), 'pred': pls_cal_pred}).to_csv(
            tmp_path / 'preds' / f'pls_seed{seed}.csv', index=False, float_format='%.5f')
    runs = load_runs(tmp_path, len(df))
    assert sorted(runs['pls']) == [0, 1]
    summary = summarise_regression(regression_table(df, runs)).set_index('set')
    m = set_masks(df)['Val Ext']
    expected = regression_metrics(df[TARGET].to_numpy()[m], pls_cal_pred[m])['rmsep']
    assert summary.loc['Val Ext', 'rmsep_mean'] == pytest.approx(expected, abs=1e-4)  # CSV keeps 5 decimals
    assert summary.loc['Val Ext', 'rmsep_sd'] == 0


def test_decision_table_matches_step2(df, pls_cal_pred):
    ours = decision_table(df, {0: pls_cal_pred}).drop(columns='seed')
    step2 = pd.DataFrame(evaluate(df)['rows'])
    pd.testing.assert_frame_equal(ours, step2)
