"""trivial_baselines.py -- climatological mean, persistence, seasonal persistence.

The trivial forecasters that define the skill baselines (they are part of the
metric definitions, phase file P2). All are fit on TRAIN data only:

  - climatological_mean : constant = mean of the training span.
  - persistence         : y_hat_{t} = y_{t-h}  (last observed value, lead h).
  - seasonal_persistence: y_hat_{t} = y_{t-period}  (period=12 for monthly data).

Each ``*_forecast`` returns ``(idx, y_true, y_pred)`` over the forecastable index
range so metrics can be computed on any sub-span (e.g. the untouched test span).
Implemented in P2. (New module: the phase file mandates these here; heavier
statistical baselines -- ridge-AR, ARIMA -- arrive in P7.)
"""
from __future__ import annotations

import numpy as np


def climatological_mean(y, n_train):
    """Constant train-mean predictor value (fit on the first ``n_train`` points)."""
    y = np.asarray(y, dtype=float)
    return float(np.mean(y[:n_train]))


def mean_forecast(y, n_train):
    """``(idx, y_true, y_pred)`` predicting the train mean at every index.

    The predictor is the train-only climatological mean; the caller slices the
    returned arrays to the span of interest (e.g. the untouched test span).
    """
    y = np.asarray(y, dtype=float)
    mu = climatological_mean(y, n_train)
    idx = np.arange(len(y))
    return idx, y, np.full(len(y), mu)


def persistence_forecast(y, horizon=1):
    """Persistence: predict y_{t-horizon}. Valid for t >= horizon."""
    y = np.asarray(y, dtype=float)
    h = int(horizon)
    idx = np.arange(h, len(y))
    return idx, y[h:], y[:len(y) - h]


def seasonal_persistence_forecast(y, period=12):
    """Seasonal persistence: predict y_{t-period}. Valid for t >= period."""
    y = np.asarray(y, dtype=float)
    p = int(period)
    idx = np.arange(p, len(y))
    return idx, y[p:], y[:len(y) - p]
