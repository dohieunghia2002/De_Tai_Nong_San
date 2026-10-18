import numpy as np
import pandas as pd
import pytest

from src.data import TARGET, set_masks, wavelength_columns, wavelengths
from src.models_pls import fit_pls
from src.preprocess import preprocess
from src.uncertainty import SpectralOOD, bias_correct, conformal_quantile


def test_conformal_quantile_rank():
    # n = 19, alpha = 0.1 -> the ceil(20 * 0.9) = 18th smallest |residual|
    assert conformal_quantile(np.arange(1, 20), 0.1) == 18
    # absolute values are used: |[-3, 1, 2]| sorted = [1, 2, 3], ceil(4 * 0.5) = 2nd smallest
    assert conformal_quantile([-3, 1, 2], 0.5) == 2
    # too few calibration points for the requested level
    assert conformal_quantile(np.arange(5), 0.1) == np.inf


def test_bias_correct_removes_constant_shift():
    y = np.linspace(12, 20, 20)
    fruit = pd.DataFrame({'Pop': [1] * 10 + [2] * 10, 'y': y, 'p': y + np.r_[np.full(10, 2.0), np.full(10, -1.0)]})
    out = bias_correct(fruit, 3, np.random.default_rng(0))
    assert len(out) == 20 - 2 * 3
    np.testing.assert_allclose(out['p'], out['y'])
    assert sorted(out['bias_est'].unique()) == pytest.approx([-1.0, 2.0])


def test_ood_limits_flag_five_percent_of_training(df):
    X, _ = preprocess(df[wavelength_columns(df)].to_numpy(), wavelengths(df))
    cal = set_masks(df)['Cal']
    model = fit_pls(X[cal], df[TARGET].to_numpy()[cal], 11)
    flags = SpectralOOD(model, X[cal], pct=95).flags(X[cal])
    for f in flags.values():
        assert f.mean() == pytest.approx(0.05, abs=0.001)
