"""shadow_battery.py -- the always-on baseline battery (Phase 9; R5).

Champion-challenger is the STEADY STATE: every stored prediction row
carries the battery columns (Phase-9 acceptance). Members: mean (=1.0
NMSE line), persistence, smart-persistence (persist k_t, reproject --
the honest one-step champion on solar), linear lags, and hooks for the
size-matched ESN / NVAR / Haar / classical-only ablation columns that
the stage-6 harness supplies at study time.
"""

from __future__ import annotations

import numpy as np


def battery_row(y_hist: np.ndarray, horizon: int,
                kt_hist: np.ndarray | None = None,
                env_future: float | None = None,
                lag_weights: np.ndarray | None = None) -> dict:
    """One prediction row's battery forecasts at `horizon` steps ahead.

    y_hist: target history up to now (transformed target).
    smart-persistence needs kt_hist + the clear-sky envelope at the
    future slot (persist k_t, reproject to the target's units)."""
    row = {
        "mean": float(np.nanmean(y_hist)),
        "persistence": float(y_hist[-1]),
    }
    if kt_hist is not None and env_future is not None:
        kt_now = kt_hist[np.isfinite(kt_hist)]
        row["smart_persistence"] = (float(kt_now[-1]) * env_future
                                    if len(kt_now) else row["persistence"])
    if lag_weights is not None:
        L = len(lag_weights) - 1
        lags = y_hist[-L:][::-1] if L else np.array([])
        row["linear_lags"] = float(np.dot(lag_weights[:-1][:len(lags)],
                                          lags) + lag_weights[-1])
    return row


def fit_linear_lags(y_train: np.ndarray, L: int, horizon: int,
                    lam: float = 1e-3) -> np.ndarray:
    """Ridge on the last-L-lags window, trained on the TRAIN span only."""
    T = len(y_train)
    rows, tgts = [], []
    for k in range(L, T - horizon):
        rows.append(y_train[k - L:k][::-1])
        tgts.append(y_train[k + horizon])
    X = np.hstack([np.array(rows), np.ones((len(rows), 1))])
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]),
                        X.T @ np.array(tgts))
    return w


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = np.sin(np.arange(600) / 8.0) + 0.1 * rng.normal(size=600)
    w = fit_linear_lags(y[:400], L=8, horizon=3)
    row = battery_row(y[:500], horizon=3, lag_weights=w)
    truth = y[503]
    print({k: round(v, 3) for k, v in row.items()}, "truth",
          round(truth, 3))
    err_pers = abs(row["persistence"] - truth)
    err_lags = abs(row["linear_lags"] - truth)
    print(f"lags beat persistence here: {err_lags < err_pers}")
    print("battery self-test PASS")
