"""metrics.py -- scoring, with the reference lines that make a score readable.

A bare RMSE means nothing without something to read it against. Every metric
here is either variance-normalised (so 1.0 has a fixed meaning) or expressed as
skill against an explicit baseline, because "NRMSE = 0.43" is only good news
once you know persistence scored 0.71 on the same split.
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Variance-normalised MSE. 1.0 == the mean predictor; >1 is worse than it."""
    var = float(np.var(y_true))
    if var < 1e-15:
        return float("nan")
    return float(np.mean((y_true - y_pred) ** 2) / var)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Square root of :func:`nmse`; 1.0 is still the mean-predictor line."""
    v = nmse(y_true, y_pred)
    return float(np.sqrt(v)) if np.isfinite(v) else float("nan")


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    v = nmse(y_true, y_pred)
    return float(1.0 - v) if np.isfinite(v) else float("nan")


def skill(score_model: float, score_reference: float) -> float:
    """Fractional improvement over a reference score (lower-is-better metrics).

    ``1 - model/reference``: positive means the model beat the reference,
    0 means it merely matched it, negative means it lost. This is the number
    that belongs next to any claim, and the one a demo is most tempted to omit
    when it is negative.
    """
    if not np.isfinite(score_reference) or abs(score_reference) < 1e-15:
        return float("nan")
    return float(1.0 - score_model / score_reference)


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray
                      ) -> float:
    """Fraction of truths inside the predicted band -- the calibration check.

    A 90% band that covers 62% of outcomes is not a tight forecast, it is a
    broken one, and only this number distinguishes the two.
    """
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    """Mean band width. Coverage is trivially satisfiable by widening; report both."""
    return float(np.mean(hi - lo))


def summarise(y_true: np.ndarray, y_pred: np.ndarray,
              lo: np.ndarray | None = None, hi: np.ndarray | None = None
              ) -> dict:
    """The standard metric block for one predictor on one test split."""
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nmse": nmse(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    if lo is not None and hi is not None:
        out["coverage"] = interval_coverage(y_true, lo, hi)
        out["interval_width"] = interval_width(lo, hi)
    return out
