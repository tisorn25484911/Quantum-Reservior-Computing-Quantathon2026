"""heads.py -- L3 read-out heads on the shared feature stream (Phase 9).

Multi-horizon QUANTILE heads trained on pinball loss (one W_out per
horizon x quantile, fit by iterated reweighted ridge -- deterministic,
dependency-free), an EVENT-probability head (logistic on the same
features, isotonic-recalibrated downstream), and a per-horizon leak
(shorter horizons want faster memory).
"""

from __future__ import annotations

import numpy as np


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, tau: float) -> float:
    d = y - q_pred
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def fit_quantile(X: np.ndarray, y: np.ndarray, tau: float,
                 lam: float = 1e-4, n_iter: int = 30) -> np.ndarray:
    """Quantile regression by iteratively reweighted least squares on the
    smoothed pinball loss. Deterministic; anchor: recovers the tau-th
    quantile of iid noise around a known linear map."""
    Xd = np.hstack([X, np.ones((len(X), 1))])
    w = np.linalg.solve(Xd.T @ Xd + lam * np.eye(Xd.shape[1]), Xd.T @ y)
    eps = 1e-6
    for _ in range(n_iter):
        r = y - Xd @ w
        wt = np.where(r >= 0, tau, 1 - tau) / np.maximum(np.abs(r), eps)
        A = Xd.T @ (wt[:, None] * Xd) + lam * np.eye(Xd.shape[1])
        w = np.linalg.solve(A, Xd.T @ (wt * y))
    return w


def predict_head(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ w


class QuantileHeads:
    """One head per (horizon, quantile); monotonicity enforced at
    predict time by sorting the quantile fan (standard crossing fix,
    reported when it fires)."""

    def __init__(self, horizons: list[int],
                 taus: tuple[float, ...] = (0.1, 0.5, 0.9),
                 lam: float = 1e-4):
        self.horizons = horizons
        self.taus = taus
        self.lam = lam
        self.w: dict[tuple[int, float], np.ndarray] = {}
        self.crossings = 0

    def fit(self, F: np.ndarray, y: np.ndarray) -> "QuantileHeads":
        """F[k] predicts y[k + h] for each horizon h."""
        for h in self.horizons:
            Xh, yh = F[:len(F) - h], y[h:len(F)]
            for tau in self.taus:
                self.w[(h, tau)] = fit_quantile(Xh, yh, tau, self.lam)
        return self

    def predict(self, F: np.ndarray, h: int) -> np.ndarray:
        fan = np.stack([predict_head(F, self.w[(h, tau)])
                        for tau in self.taus], axis=1)
        fixed = np.sort(fan, axis=1)
        self.crossings += int(np.sum(np.any(fixed != fan, axis=1)))
        return fixed


class EventHead:
    """Logistic event-probability head (gradient ascent, ridge prior).
    Raw probabilities; isotonic recalibration lives in the decision
    layer, coverage guarantees in the conformal wrapper."""

    def __init__(self, lam: float = 1e-3, lr: float = 0.5,
                 n_iter: int = 300):
        self.lam, self.lr, self.n_iter = lam, lr, n_iter

    def fit(self, F: np.ndarray, event: np.ndarray) -> "EventHead":
        Xd = np.hstack([F, np.ones((len(F), 1))])
        w = np.zeros(Xd.shape[1])
        y = np.asarray(event, dtype=float)
        for _ in range(self.n_iter):
            p = 1 / (1 + np.exp(-Xd @ w))
            g = Xd.T @ (y - p) / len(y) - self.lam * w
            w += self.lr * g
        self.w = w
        return self

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        Xd = np.hstack([F, np.ones((len(F), 1))])
        return 1 / (1 + np.exp(-Xd @ self.w))


def per_horizon_leak(horizons: list[int], base_eps: float = 0.5,
                     steps_per_hour: float = 2.0) -> dict[int, float]:
    """Shorter horizon -> faster memory (larger eps). eps_h =
    clip(base * sqrt(1h / horizon_hours), 0.05, 1.0)."""
    out = {}
    for h in horizons:
        hours = max(h / steps_per_hour, 1e-6)
        out[h] = float(np.clip(base_eps * np.sqrt(1.0 / hours), 0.05, 1.0))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 5))
    w_true = rng.normal(size=5)
    y = X @ w_true + rng.normal(size=3000)          # unit noise
    w10 = fit_quantile(X, y, 0.1)
    w90 = fit_quantile(X, y, 0.9)
    resid10 = y - predict_head(X, w10)
    resid90 = y - predict_head(X, w90)
    c10 = float(np.mean(resid10 > 0))               # should be ~0.9
    c90 = float(np.mean(resid90 > 0))               # should be ~0.1
    print(f"quantile heads: P(y > q10) = {c10:.3f} (target 0.9), "
          f"P(y > q90) = {c90:.3f} (target 0.1)")
    assert abs(c10 - 0.9) < 0.03 and abs(c90 - 0.1) < 0.03
    print("heads self-test PASS")
