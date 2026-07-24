"""nvar.py -- Nonlinear Vector AutoRegression (Gauthier et al. 2021), 1-D form.

Delay-embedded linear features (k lags, stride s) augmented with unique degree-2
polynomial products, then a ridge readout (fit on TRAIN rows only, G3/G4). NVAR is
the "next-generation reservoir computing" control: a fixed nonlinear feature map +
linear readout with NO trained recurrent state, so any QRC edge over it is a
statement about the quantum feature map, not about recurrence per se.

Feature row at step k (available information up to y_k):
    linear  : [y_k, y_{k-s}, ..., y_{k-(k_lag-1)s}]
    nonlinear: unique products f_i * f_j (i<=j) of the linear part
    + constant.
Pairs with target y_{k+1} (G3). Same object exposes ``warm``/``step`` for the one
autonomous engine. Implemented in P7.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import readout as R


def _lin_row(buf):
    """Linear delay vector from a most-recent-first buffer (already strided)."""
    return np.asarray(buf, dtype=float)


def _poly(lin):
    """Constant + linear + unique degree-2 products of ``lin``."""
    k = len(lin)
    quad = [lin[i] * lin[j] for i in range(k) for j in range(i, k)]
    return np.concatenate([[1.0], lin, quad])


def _design(y, k_lag, stride):
    """Feature rows (with target y_{t+1}); needs history span (k_lag-1)*stride."""
    y = np.asarray(y, dtype=float)
    span = (k_lag - 1) * stride
    rows, targets = [], []
    for t in range(span, len(y) - 1):
        lin = np.array([y[t - j * stride] for j in range(k_lag)])   # most-recent first
        rows.append(_poly(lin))
        targets.append(y[t + 1])
    return np.array(rows), np.array(targets), span


@dataclass
class NVAR:
    k_lag: int
    stride: int
    span: int
    W: np.ndarray                 # (F,) readout weights (includes constant col)
    b: float
    lam: float

    def _row(self, buf):
        return _poly(_lin_row(buf))

    def predict_onestep(self, y):
        X, _, _ = _design(y, self.k_lag, self.stride)
        return X @ self.W + self.b

    # -- autonomous adapter --------------------------------------------------
    def warm(self, history):
        """Buffer of the last span+1 raw values (last one arrives as first u)."""
        h = np.asarray(history, dtype=float)
        need = self.span + 1
        if len(h) < need:
            h = np.concatenate([np.full(need - len(h), h[0]), h])
        return list(h[-need:-1])                          # oldest..newest (excl last)

    def step(self, state, u):
        hist = np.array([*state, u])                      # oldest..newest incl u
        buf = np.array([hist[-1 - j * self.stride] for j in range(self.k_lag)])
        yhat = float(self._row(buf) @ self.W + self.b)
        return list(hist[-self.span:]) if self.span else [], yhat


def fit(y, n_train, k_lag=3, stride=1, lam="gcv", washout=0):
    X, target, span = _design(y, k_lag, stride)
    lo = max(washout - span - 1, 0)
    hi = max(n_train - span - 1, lo + 1)
    ro = R.fit(X[lo:hi], target[lo:hi], lam=lam)
    return NVAR(k_lag=k_lag, stride=stride, span=span,
                W=ro.W[:, 0].copy(), b=float(ro.b[0]), lam=float(ro.lam[0]))
