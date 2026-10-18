"""M1 - spectral pre-treatment: optional SNV, Savitzky-Golay derivative on the full spectrum, then trim to a window."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class PreprocessConfig:
    # Defaults follow Anderson et al. 2020: SG second derivative, 2nd-order polynomial, 17 points, 684-990 nm.
    deriv: int = 2
    window: int = 17
    polyorder: int = 2
    snv: bool = False
    wl_min: int = 684
    wl_max: int = 990


def snv(X, ref_mask=None):
    """Standard normal variate per spectrum, with mean and SD taken over the columns in ref_mask (all by default)."""
    R = X if ref_mask is None else X[:, ref_mask]
    return (X - R.mean(axis=1, keepdims=True)) / R.std(axis=1, keepdims=True)


def preprocess(X, wl, cfg=PreprocessConfig()):
    """Return (X_trimmed, wl_trimmed). The SG filter runs on the full spectrum before trimming, so the window edges
    are not distorted by the filter boundary."""
    X = np.asarray(X, dtype=float)
    wl = np.asarray(wl)
    keep = (wl >= cfg.wl_min) & (wl <= cfg.wl_max)
    if cfg.snv:
        # SNV statistics over the model window only: the file spans 285-1200 nm, wider than the F750's 300-1100 nm.
        X = snv(X, keep)
    X = savgol_filter(X, cfg.window, cfg.polyorder, deriv=cfg.deriv, axis=1)
    return X[:, keep], wl[keep]
