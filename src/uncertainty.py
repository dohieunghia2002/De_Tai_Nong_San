"""M3 - uncertainty and season shift: split-conformal interval, per-population bias correction, spectral OOD scores."""
import numpy as np
import pandas as pd


def conformal_quantile(residuals, alpha=0.1):
    """Split-conformal half-width q: the ceil((n+1)(1-alpha))-th smallest |residual| of the calibration fruit.
    [p - q, p + q] then covers the true value with probability >= 1 - alpha for exchangeable new fruit."""
    s = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    k = int(np.ceil((s.size + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= s.size else np.inf


def bias_correct(fruit, k, rng):
    """Per population, k random fruit are measured destructively and their mean error (p - y) is subtracted from the
    predictions of the other fruit. The k reference fruit are dropped from the returned table."""
    parts = []
    for _, g in fruit.groupby('Pop'):
        idx = rng.choice(len(g), size=min(k, len(g)), replace=False)
        bias = (g['p'].to_numpy()[idx] - g['y'].to_numpy()[idx]).mean()
        rest = g.drop(g.index[idx])
        parts.append(rest.assign(p=rest['p'] - bias, bias_est=bias))
    return pd.concat(parts)


def repeated_bias_correct(fruit, k, n_rep, rng):
    return pd.concat([bias_correct(fruit, k, rng).assign(rep=r) for r in range(n_rep)])


class SpectralOOD:
    """Q-residual (squared reconstruction error outside the PLS score space) and Hotelling T2 (distance inside it),
    with limits at a percentile of the training spectra."""

    def __init__(self, pls, X_train, pct=95):
        self.pls = pls
        self.score_var = pls.transform(X_train).var(axis=0, ddof=1)
        self.q_limit = np.percentile(self.q_residual(X_train), pct)
        self.t2_limit = np.percentile(self.hotelling_t2(X_train), pct)

    def q_residual(self, X):
        return ((X - self.pls.inverse_transform(self.pls.transform(X))) ** 2).sum(axis=1)

    def hotelling_t2(self, X):
        return (self.pls.transform(X) ** 2 / self.score_var).sum(axis=1)

    def flags(self, X):
        return {'q_residual': self.q_residual(X) > self.q_limit, 'hotelling_t2': self.hotelling_t2(X) > self.t2_limit}
