"""preprocessing.py -- leak-free transforms (train-statistics only).

Every fitted statistic (monthly climatology for the anomaly, and the [0,1]
encoder scaler) is computed on TRAIN rows only and then applied to the whole
series. Detrending / deseasonalisation / differencing are explicit ablation
arms (spec s14); the anomaly convention (subtract a training-years-only monthly
climatology) is the project default per the playbook rule for climate targets.

The [0,1] scaling is required by the paper-faithful input encoding
|psi(s)> = sqrt(1-s)|0> + sqrt(s)|1>, which needs s in [0,1]. Test-span values
outside the training range are clipped, and the count of clipped values is
reported by the caller (never hidden).
"""

from __future__ import annotations

import numpy as np


def monthly_climatology(month: np.ndarray, y: np.ndarray, n_train: int) -> np.ndarray:
    """Per-calendar-month mean over the first n_train points only."""
    clim = np.zeros(12)
    m_tr, y_tr = month[:n_train], y[:n_train]
    for j in range(12):
        sel = m_tr == j + 1
        clim[j] = y_tr[sel].mean() if sel.any() else 0.0
    return clim


def anomaly(month: np.ndarray, y: np.ndarray, n_train: int):
    """Full-series anomaly against the train-only monthly climatology."""
    clim = monthly_climatology(month, y, n_train)
    return y - clim[month - 1], clim


def fit_scaler(y: np.ndarray, n_train: int):
    """Affine map to [0,1] fit on train rows; clips outside. Returns
    (scale_fn, (lo, hi))."""
    lo, hi = float(np.min(y[:n_train])), float(np.max(y[:n_train]))
    span = hi - lo if hi > lo else 1.0

    def scale(x):
        return np.clip((x - lo) / span, 0.0, 1.0)

    return scale, (lo, hi)


def clip_report(u_unclipped: np.ndarray) -> dict:
    """Telemetry for encoder-domain clipping (spec s16.4)."""
    below = u_unclipped < 0.0
    above = u_unclipped > 1.0
    n = len(u_unclipped)
    return {
        "n": int(n),
        "n_clipped": int(below.sum() + above.sum()),
        "frac_clipped": float((below.sum() + above.sum()) / n) if n else 0.0,
        "max_below": float((-u_unclipped[below]).max()) if below.any() else 0.0,
        "max_above": float((u_unclipped[above] - 1.0).max()) if above.any() else 0.0,
    }
