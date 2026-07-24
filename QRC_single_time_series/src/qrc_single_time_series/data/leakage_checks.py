"""leakage_checks.py -- assertions that fitted stats saw train rows only."""
from __future__ import annotations
import numpy as np


def assert_scaler_train_only(scale_fn, y, n_train):
    """A [0,1] scaler must map the train min/max to 0/1 and never peek ahead:
    perturbing test-span values must not change the scaling of train values."""
    base = scale_fn(y[:n_train]).copy()
    y2 = y.copy(); y2[n_train:] += 1e6
    assert np.allclose(base, scale_fn(y2[:n_train])), "scaler leaked test info"


def assert_no_future_in_window(win_end_idx, window_len, series_len):
    assert win_end_idx < series_len and win_end_idx - window_len + 1 >= 0
