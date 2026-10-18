import numpy as np
import pytest

from src.data import TARGET, TRAIN_SETS, wavelength_columns, wavelengths
from src.models_pls import fit_pls, predict, regression_metrics
from src.preprocess import preprocess


def test_metrics_hand_computed():
    # errors (pred - true) = 1, 0, 2
    m = regression_metrics([10, 12, 14], [11, 12, 16])
    assert m['n'] == 3
    assert m['bias'] == pytest.approx(1.0)
    assert m['rmsep'] == pytest.approx(np.sqrt(5 / 3))
    assert m['sep'] == pytest.approx(1.0)
    assert m['r2'] == pytest.approx(1 - 5 / 8)


def test_season4_rmsep_matches_anderson_2020(df):
    X, _ = preprocess(df[wavelength_columns(df)].to_numpy(), wavelengths(df))
    y = df[TARGET].to_numpy()
    train = df['Set'].isin(TRAIN_SETS).to_numpy()
    test = (df['Set'] == 'Val Ext').to_numpy()
    model = fit_pls(X[train], y[train], 11)
    m = regression_metrics(y[test], predict(model, X[test]))
    assert m['rmsep'] == pytest.approx(1.01, abs=0.005)
