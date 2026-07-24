"""mitigation.py -- shot emulation + the three-arm noise-vs-regulariser adjudication.

Spec s27 sharpened: if hardware-like noise improves test error only by CONDITIONING
the readout, an explicit classical regulariser at matched effective dof must recover
at least the same gain WITHOUT noise. This module provides the three arms and the
one shot-emulation formula used consistently across ALL models (spec s10, gotchas):

    z in [-1,1], p=(1+z)/2, sigma(z)=2 sqrt(p(1-p)/S), z' = clip(z + xi sigma, -1, 1)

applied to the reservoir feature x' = (1+z)/2. The three arms:
  A  noise      -- readout fit on finite-shot (noisy) TRAIN features.
  B  ridge      -- readout fit on EXACT features with ridge lambda (matched dof).
  C  truncation -- readout fit on EXACT features truncated to the matched SVD rank
                   (the Ahmed-Tennie-Magri filtering arm).
A "noise helps" claim is admitted only if arm A beats arm B AND arm C on held-out
test error, across seeds. Implemented in P10.
"""
from __future__ import annotations

import numpy as np

from ..models import readout as R


def emulate_shots(features, shots, rng):
    """Apply the finite-shot emulation formula to a reservoir feature matrix.

    The bias column (constant 1.0, detected as zero-variance) is left untouched.
    """
    X = np.asarray(features, dtype=float).copy()
    z = 2.0 * X - 1.0
    p = 0.5 * (1.0 + z)
    sig = 2.0 * np.sqrt(np.clip(p * (1 - p), 0, None) / shots)
    zc = np.clip(z + rng.standard_normal(z.shape) * sig, -1.0, 1.0)
    Xn = 0.5 * (1.0 + zc)
    const = X.std(axis=0) < 1e-12
    Xn[:, const] = X[:, const]                       # keep the bias column exact
    return Xn


def svd_truncate_fit(X, y, rank):
    """Least-squares readout on the top-``rank`` SVD directions of centred X."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    xbar, ybar = X.mean(0), y.mean()
    Xc, yc = X - xbar, y - ybar
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = int(min(rank, len(s)))
    d = np.zeros_like(s)
    d[:r] = 1.0 / s[:r]
    W = Vt.T @ (d * (U.T @ yc))
    b = ybar - xbar @ W
    return W, b


def _nmse(y, pred):
    return float(np.var(y - pred) / np.var(y))


def three_arm_adjudication(X_exact_tr, y_tr, X_exact_te, y_te, shots,
                           ridge_lam, trunc_rank, seeds=range(8)):
    """Held-out NMSE for arms A (noise), B (matched ridge), C (SVD truncation).

    Arm A is averaged over ``seeds`` shot draws (the test features are ALWAYS the
    exact ones -- we ask whether training on noise conditions the readout better than
    an explicit regulariser). Returns the three NMSEs and the verdict.
    """
    # B: matched ridge on exact features
    roB = R.fit(X_exact_tr, y_tr, lam=ridge_lam)
    nmse_B = _nmse(y_te, roB.predict(X_exact_te))
    # C: SVD truncation on exact features
    Wc, bc = svd_truncate_fit(X_exact_tr, y_tr, trunc_rank)
    nmse_C = _nmse(y_te, X_exact_te @ Wc + bc)
    # A: train on finite-shot features, several seeds
    nmses_A = []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        Xn = emulate_shots(X_exact_tr, shots, rng)
        roA = R.fit(Xn, y_tr, lam=0.0)
        nmses_A.append(_nmse(y_te, roA.predict(X_exact_te)))
    nmse_A = float(np.mean(nmses_A))
    noise_helps = bool(nmse_A < nmse_B and nmse_A < nmse_C)
    return {
        "arm_A_noise_nmse": nmse_A,
        "arm_A_noise_std": float(np.std(nmses_A)),
        "arm_B_ridge_nmse": nmse_B,
        "arm_C_truncation_nmse": nmse_C,
        "shots": int(shots), "ridge_lam": float(ridge_lam), "trunc_rank": int(trunc_rank),
        "noise_helps_beyond_matched_regulariser": noise_helps,
    }
