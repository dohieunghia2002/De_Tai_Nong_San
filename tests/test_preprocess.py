import numpy as np
from scipy.signal import savgol_filter

from src.data import wavelength_columns, wavelengths
from src.preprocess import PreprocessConfig, preprocess


def spectra(df, n=None):
    return df[wavelength_columns(df)].to_numpy()[:n], wavelengths(df)


def test_default_window(df):
    raw, wl = spectra(df)
    X, wl_kept = preprocess(raw, wl)
    assert X.shape == (len(df), 103)
    assert wl_kept[0] == 684 and wl_kept[-1] == 990


def test_derivative_taken_before_trim(df):
    raw, wl = spectra(df, 50)
    keep = (wl >= 684) & (wl <= 990)
    X, _ = preprocess(raw, wl)
    np.testing.assert_allclose(X, savgol_filter(raw, 17, 2, deriv=2, axis=1)[:, keep])
    # trimming first would distort the window edges
    assert not np.allclose(X[:, :8], savgol_filter(raw[:, keep], 17, 2, deriv=2, axis=1)[:, :8])


def test_alternative_settings(df):
    raw, wl = spectra(df, 50)
    X, _ = preprocess(raw, wl, PreprocessConfig(deriv=1, window=13, snv=True))
    assert X.shape == (50, 103) and np.isfinite(X).all()
