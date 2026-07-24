"""accumulated_error.py -- CNRMSE_T(h) etc. with train-only sigma.

Autonomous-error accumulators (spec s16). ``sigma_train`` is computed on TRAIN
data only and normalises every "N" (normalised) quantity, so a value of 1.0 means
"as wrong as the training standard deviation". For a rollout from origin T with
predictions yhat[1..H] against truth y[1..H]:

  instantaneous  NE(h)     = |yhat_h - y_h| / sigma_train
  cumulative     CRMSE(h)  = sqrt(mean_{k<=h} (yhat_k - y_k)^2)
                 CNRMSE(h) = CRMSE(h) / sigma_train
                 CNMAE(h)  = (mean_{k<=h} |yhat_k - y_k|) / sigma_train

The cumulative forms are RUNNING averages -- they can dip after a spike, which is
why the prediction-horizon prefix rule (prediction_horizon.py) is load-bearing.
Implemented in P4.
"""
from __future__ import annotations

import numpy as np


def sigma_train(y_train):
    s = float(np.std(np.asarray(y_train, dtype=float)))
    if s == 0:
        raise ValueError("sigma_train is zero (constant training series)")
    return s


def instantaneous_ne(y_true, y_pred, sigma):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return np.abs(y_pred - y_true) / sigma


def crmse(y_true, y_pred):
    """Cumulative RMSE over the prefix 1..h, as an array indexed by h-1."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    se = (y_pred - y_true) ** 2
    return np.sqrt(np.cumsum(se) / np.arange(1, len(se) + 1))


def cnrmse(y_true, y_pred, sigma):
    return crmse(y_true, y_pred) / sigma


def cnmae(y_true, y_pred, sigma):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ae = np.abs(y_pred - y_true)
    return (np.cumsum(ae) / np.arange(1, len(ae) + 1)) / sigma
