"""windows.py -- align(features_by_k, y, horizon): the ONLY feature/target pairing (G3).

Single choke point for pairing a feature row with its target so an off-by-one can
only ever happen in one place (G3). Next-step convention (Hamhoum Eq. 7, FN NARMA):
the feature row assembled from information available at step ``k`` pairs with the
target ``y_{k+horizon}``. ``horizon=1`` is one-step-ahead forecasting; ``horizon=0``
pairs the row with the target at the same step (used by the STM tau_B=0 exactness
test, where the debug post-injection sample carries <Z_inj> = 1 - 2 s_k exactly).
For memory/STM tasks with delay tau_B, pass ``horizon = -tau_B``. Implemented in P2.
"""
from __future__ import annotations

import numpy as np


def align(features_by_k, y, horizon=1):
    """Pair feature rows with targets under the next-step convention (G3).

    Parameters
    ----------
    features_by_k : array (K, F) -- feature row per step k (k = 0..K-1).
    y : array (K,) or (K, L) -- target series indexed by the SAME k.
    horizon : int -- pair row k with target at k + horizon (may be negative).

    Returns ``(X, Y)`` with matching first dimension, dropping the boundary rows
    that have no valid partner.
    """
    X = np.asarray(features_by_k)
    Y = np.asarray(y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"feature/target length mismatch: {X.shape[0]} vs {Y.shape[0]}")
    K = X.shape[0]
    if horizon >= 0:
        Xa, Ya = X[: K - horizon], Y[horizon:]
    else:
        d = -horizon
        Xa, Ya = X[d:], Y[: K - d]
    if Xa.shape[0] == 0:
        raise ValueError(f"horizon={horizon} leaves no aligned rows (K={K})")
    return Xa, Ya
