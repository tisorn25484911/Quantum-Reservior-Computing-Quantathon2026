"""readout.py -- ridge/pinv linear readout via one economy SVD (ridge path + GCV).

Linear readout y_hat = X W + b. Fitted by centering X and y on TRAIN statistics
(no bias column in the solve; the intercept is recovered as b = y_bar - x_bar W),
then one economy SVD of the centered feature matrix gives the whole ridge path

    W(lambda) = V diag(s_j / (s_j^2 + lambda)) U^T y_c,   dof(lambda) = sum s_j^2/(s_j^2+lambda)

for free across lambda. lambda = 0 is the paper-faithful pseudo-inverse mode
(the retained rank and ||W|| are logged). A GCV-via-SVD selector chooses lambda
per target for thin / regime-switching validation tails. Multi-target Y is solved
in the one decomposition. Implemented in P2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Readout:
    W: np.ndarray                 # (F, L)
    b: np.ndarray                 # (L,)
    lam: np.ndarray               # (L,) lambda used per target
    rank: int                     # retained SVD rank (s > rcond*s_max)
    w_norm: np.ndarray            # (L,) ||W[:, l]||_2, logged for the pinv mode
    dof: np.ndarray               # (L,) effective degrees of freedom
    _1d: bool = field(default=False, repr=False)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = X @ self.W + self.b
        return out[:, 0] if self._1d else out


def _svd(Xc, rcond=1e-10):
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    tol = rcond * (s[0] if s.size else 0.0)
    rank = int(np.sum(s > tol))
    return U[:, :rank], s[:rank], Vt[:rank], rank


def _gcv_lambda(U, s, yc, grid):
    """Pick lambda minimising GCV(lambda) for a single centered target ``yc``."""
    T = len(yc)
    Uty = U.T @ yc
    y_norm2 = float(yc @ yc)
    proj2 = float(Uty @ Uty)          # ||U^T yc||^2
    resid_null = y_norm2 - proj2      # component outside col(U): lambda-independent
    best_lam, best_gcv = grid[0], np.inf
    for lam in grid:
        filt = s ** 2 / (s ** 2 + lam)          # (r,)
        resid = resid_null + float(((1.0 - filt) ** 2) @ (Uty ** 2))
        dof = float(np.sum(filt))
        denom = (1.0 - dof / T) ** 2
        gcv = (resid / T) / denom if denom > 1e-15 else np.inf
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, lam
    return best_lam


def fit(X, Y, lam="gcv", rcond=1e-10, gcv_grid=None):
    """Fit a centered linear readout.

    ``lam`` : "gcv" (per-target GCV), 0.0 (pinv / paper-faithful), or a float ridge.
    Returns a ``Readout``. ``Y`` may be 1-D (single target) or (T, L).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    is_1d = Y.ndim == 1
    if is_1d:
        Y = Y[:, None]
    xbar = X.mean(axis=0)
    ybar = Y.mean(axis=0)
    Xc = X - xbar
    Yc = Y - ybar

    U, s, Vt, rank = _svd(Xc, rcond)
    if gcv_grid is None:
        smax2 = (s[0] ** 2) if s.size else 1.0
        gcv_grid = np.concatenate([[0.0], np.logspace(-10, 2, 60) * smax2])

    L = Y.shape[1]
    F = X.shape[1]
    W = np.zeros((F, L))
    lams = np.zeros(L)
    dofs = np.zeros(L)
    for l in range(L):
        yc = Yc[:, l]
        if lam == "gcv":
            lam_l = _gcv_lambda(U, s, yc, gcv_grid)
        else:
            lam_l = float(lam)
        d = s / (s ** 2 + lam_l)
        W[:, l] = Vt.T @ (d * (U.T @ yc))
        lams[l] = lam_l
        dofs[l] = float(np.sum(s ** 2 / (s ** 2 + lam_l)))
    b = ybar - xbar @ W
    return Readout(W=W, b=b, lam=lams, rank=rank,
                   w_norm=np.linalg.norm(W, axis=0), dof=dofs, _1d=is_1d)
