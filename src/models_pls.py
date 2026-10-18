"""M2 baseline - PLS regression of DMC, and spectrum-level regression metrics."""
import numpy as np
from sklearn.cross_decomposition import PLSRegression


def fit_pls(X, y, n_components=11):
    # scale=False: mean centring only, as in Anderson et al. 2020.
    return PLSRegression(n_components=n_components, scale=False).fit(X, y)


def predict(model, X):
    return model.predict(X).ravel()


def regression_metrics(y_true, y_pred):
    """bias = mean(pred - true); SEP = SD of the errors (bias-corrected RMSEP);
    r2 = 1 - SSE/SST; r2_pearson = squared correlation between prediction and reference."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e = y_pred - y_true
    bias = e.mean()
    return {
        'n': int(e.size),
        'rmsep': float(np.sqrt(np.mean(e ** 2))),
        'bias': float(bias),
        'sep': float(np.sqrt(np.sum((e - bias) ** 2) / (e.size - 1))),
        'r2': float(1 - np.sum(e ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        'r2_pearson': float(np.corrcoef(y_true, y_pred)[0, 1] ** 2),
    }
