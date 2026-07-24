"""metrics.py -- MSE/RMSE/MAE, BOTH NMSE conventions (G8-5), MASE, capacity C(tau_B).

Every normalised error is labelled by convention (errata G8-5):
  - ``nmse_variance``: MSE / Var(y)  -- the mean-predictor scores exactly 1.000.
  - ``nmse_fn``      : sum (y - yhat)^2 / sum y^2  (Fujii-Nakajima Eq. A1).
Also R^2, Pearson r, MASE (scaled by the in-sample naive one-step MAE), anomaly
correlation, linear memory/processing capacity C(tau_B) = squared correlation
between target and prediction, and the effective rank of a feature matrix
(Roy-Vetterli spectral entropy). Implemented in P2.
"""
from __future__ import annotations

import numpy as np


def _flat(y):
    return np.asarray(y, dtype=float).ravel()


def mse(y, yhat):
    y, yhat = _flat(y), _flat(yhat)
    return float(np.mean((y - yhat) ** 2))


def rmse(y, yhat):
    return float(np.sqrt(mse(y, yhat)))


def mae(y, yhat):
    y, yhat = _flat(y), _flat(yhat)
    return float(np.mean(np.abs(y - yhat)))


def nmse_variance(y, yhat):
    """MSE / Var(y). Mean predictor -> 1.000 (anchor)."""
    y, yhat = _flat(y), _flat(yhat)
    var = np.var(y)
    if var == 0:
        raise ValueError("nmse_variance undefined for constant target")
    return mse(y, yhat) / float(var)


def nmse_fn(y, yhat):
    """FN Eq. A1: sum (y - yhat)^2 / sum y^2."""
    y, yhat = _flat(y), _flat(yhat)
    denom = float(np.sum(y ** 2))
    if denom == 0:
        raise ValueError("nmse_fn undefined for all-zero target")
    return float(np.sum((y - yhat) ** 2)) / denom


def r2(y, yhat):
    y, yhat = _flat(y), _flat(yhat)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def pearson(y, yhat):
    y, yhat = _flat(y), _flat(yhat)
    if np.std(y) == 0 or np.std(yhat) == 0:
        return float("nan")
    return float(np.corrcoef(y, yhat)[0, 1])


def mase(y, yhat, y_train):
    """MAE scaled by the in-sample naive one-step MAE of ``y_train`` (seasonal=1)."""
    y, yhat, y_train = _flat(y), _flat(yhat), _flat(y_train)
    scale = np.mean(np.abs(np.diff(y_train)))
    if scale == 0:
        raise ValueError("MASE undefined: zero naive-forecast scale")
    return mae(y, yhat) / float(scale)


def anomaly_correlation(y, yhat, clim=0.0):
    """Anomaly correlation coefficient about a climatology ``clim`` (scalar/array)."""
    y, yhat = _flat(y), _flat(yhat)
    clim = np.asarray(clim, dtype=float)
    ya, yha = y - clim, yhat - clim
    denom = np.sqrt(np.sum(ya ** 2) * np.sum(yha ** 2))
    if denom == 0:
        return float("nan")
    return float(np.sum(ya * yha) / denom)


def capacity(target, pred):
    """Linear reservoir capacity C = squared correlation between target and pred."""
    r = pearson(target, pred)
    return float(r ** 2) if np.isfinite(r) else 0.0


def effective_rank(X, tol=1e-12):
    """Roy-Vetterli effective rank: exp(entropy of the normalised singular values)."""
    s = np.linalg.svd(np.asarray(X, dtype=float), compute_uv=False)
    s = s[s > tol]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def nmse_mean_predictor(y):
    """Anchor: NMSE(variance) of the constant mean predictor == 1.000."""
    y = _flat(y)
    return nmse_variance(y, np.full_like(y, np.mean(y)))


def persistence_skill(mse_model, mse_persistence):
    """Skill score 1 - MSE_model / MSE_persistence (>0 means beating persistence)."""
    if mse_persistence == 0:
        raise ValueError("persistence skill undefined: zero persistence MSE")
    return float(1.0 - mse_model / mse_persistence)
