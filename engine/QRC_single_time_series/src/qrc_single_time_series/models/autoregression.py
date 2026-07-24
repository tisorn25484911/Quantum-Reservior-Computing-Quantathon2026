"""autoregression.py -- ridge-AR on lags, with a warm/step rollout adapter.

Linear autoregression y_hat_t = b + sum_{j=1..p} w_j y_{t-j}, fit by the shared
economy-SVD ridge readout (``models.readout``) on TRAIN rows only (G3/G4). The
same object exposes ``warm``/``step`` so it feeds its own predictions back through
the ONE autonomous engine (spec s24, G5) -- no separate recursive path.

Lag/target convention (G3): the row of lags available at step k pairs with the
next value y_{k+1}. The most-recent lag is column 0, so ``step`` prepends the
fed-back value. Implemented in P7.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import readout as R


def _design(y, p):
    """Rows [y_{t-1}, ..., y_{t-p}] (most-recent first) and targets y_t, t>=p."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    X = np.empty((n - p, p))
    for j in range(1, p + 1):
        X[:, j - 1] = y[p - j: n - j]
    return X, y[p:]


@dataclass
class RidgeAR:
    """Ridge autoregression of order ``p`` with a closed-loop adapter."""

    p: int
    W: np.ndarray                 # (p,) lag coefficients, column 0 == most recent
    b: float                      # intercept
    lam: float

    # -- teacher-forced one-step ---------------------------------------------
    def predict_onestep(self, y):
        """One-step predictions y_hat_t for every t>=p over the given series."""
        X, _ = _design(y, self.p)
        return X @ self.W + self.b

    # -- autonomous rollout adapter (spec s16/s24) ---------------------------
    def warm(self, history):
        """State = the p-1 values before the last (the last arrives as first u)."""
        h = np.asarray(history, dtype=float)
        if len(h) < self.p:
            h = np.concatenate([np.full(self.p - len(h), h[0]), h])
        return list(h[-self.p:-1][::-1])                 # most-recent first

    def step(self, state, u):
        full = np.array([u, *state])[: self.p]
        if full.size < self.p:                           # pad short warmups
            full = np.concatenate([full, np.full(self.p - full.size, full[-1])])
        yhat = float(full @ self.W + self.b)
        return list(full[: self.p - 1]), yhat


def fit(y, n_train, p=6, lam="gcv", washout=0):
    """Fit RidgeAR on the first ``n_train`` points (target index in [washout,n_train))."""
    X, target = _design(y, p)
    lo = max(washout - p, 0)
    hi = max(n_train - p, lo + 1)
    ro = R.fit(X[lo:hi], target[lo:hi], lam=lam)
    return RidgeAR(p=p, W=ro.W[:, 0].copy(), b=float(ro.b[0]), lam=float(ro.lam[0]))
