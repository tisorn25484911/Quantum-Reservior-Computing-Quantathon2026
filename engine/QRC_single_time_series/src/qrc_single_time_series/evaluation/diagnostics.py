"""diagnostics.py -- feature-space suite: eff. rank, condition, SVD, cosine (spec s27).

How finite sampling and noise deform the reservoir feature matrix, relative to the
exact (noiseless) features. All quantities are decision-adjacent diagnostics, but
the spec's own caveat holds: open-loop feature RMSE UNDERSTATES closed-loop damage
because the recursion amplifies it -- so the horizon curves (P8/P10 scripts), not
these numbers, are the decision output. Reported per spec s27:

  bias / variance / RMSE vs exact, feature-wise correlation, covariance-matrix
  distance, cosine similarity, singular-value spectrum, explained variance,
  effective rank (Rank_eff = 1/HHI, Hamhoum Eq. 17, on STANDARDISED features),
  condition number, and readout-coefficient stability across seeds. Implemented in P10.
"""
from __future__ import annotations

import numpy as np


def _standardise(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def effective_rank_hhi(X):
    """Rank_eff = 1 / sum(p_i^2), p_i = normalised singular-value energy (Hamhoum Eq. 17)."""
    s = np.linalg.svd(_standardise(X), compute_uv=False)
    e = s ** 2
    tot = e.sum()
    if tot == 0:
        return 0.0
    p = e / tot
    return float(1.0 / np.sum(p ** 2))


def condition_number(X):
    s = np.linalg.svd(_standardise(X), compute_uv=False)
    s = s[s > 1e-15]
    return float(s[0] / s[-1]) if s.size else np.inf


def explained_variance(X, k=None):
    """Cumulative explained-variance ratio of the top-k singular directions."""
    s = np.linalg.svd(_standardise(X), compute_uv=False)
    e = s ** 2
    ev = np.cumsum(e) / e.sum() if e.sum() > 0 else e
    return ev.tolist() if k is None else float(ev[min(k, len(ev)) - 1])


def feature_error(X_exact, X_noisy):
    """Per-entry bias / variance / RMSE and mean feature-wise correlation vs exact."""
    Xe = np.asarray(X_exact, float)
    Xn = np.asarray(X_noisy, float)
    d = Xn - Xe
    corrs = []
    for j in range(Xe.shape[1]):
        if Xe[:, j].std() > 1e-12 and Xn[:, j].std() > 1e-12:
            corrs.append(np.corrcoef(Xe[:, j], Xn[:, j])[0, 1])
    return {
        "bias_mean": float(d.mean()),
        "variance_mean": float(d.var()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "feature_corr_mean": float(np.mean(corrs)) if corrs else None,
    }


def cosine_similarity_rows(X_exact, X_noisy):
    """Mean row-wise cosine similarity between exact and noisy feature rows."""
    Xe = np.asarray(X_exact, float)
    Xn = np.asarray(X_noisy, float)
    num = np.sum(Xe * Xn, axis=1)
    den = np.linalg.norm(Xe, axis=1) * np.linalg.norm(Xn, axis=1)
    ok = den > 1e-12
    return float(np.mean(num[ok] / den[ok])) if ok.any() else None


def covariance_distance(X_exact, X_noisy):
    """Frobenius distance between the two feature covariance matrices."""
    Ce = np.cov(np.asarray(X_exact, float), rowvar=False)
    Cn = np.cov(np.asarray(X_noisy, float), rowvar=False)
    return float(np.linalg.norm(Ce - Cn))


def readout_coefficient_stability(fit_fn, feature_fn, seeds):
    """Coefficient-of-variation of the readout weights across independent seeds.

    ``feature_fn(seed) -> (X, y)``; ``fit_fn(X, y) -> weight_vector``. Returns the
    mean |std/mean| of the stacked weights (lower == more stable readout).
    """
    Ws = []
    for s in seeds:
        X, y = feature_fn(s)
        Ws.append(np.ravel(fit_fn(X, y)))
    W = np.stack(Ws)
    mu = np.abs(W.mean(axis=0))
    sd = W.std(axis=0)
    ok = mu > 1e-9
    return float(np.mean(sd[ok] / mu[ok])) if ok.any() else None


def feature_space_report(X_exact, X_noisy):
    """All spec-s27 diagnostics comparing noisy features to the exact ceiling."""
    rep = feature_error(X_exact, X_noisy)
    rep.update({
        "cosine_rows": cosine_similarity_rows(X_exact, X_noisy),
        "covariance_frobenius": covariance_distance(X_exact, X_noisy),
        "eff_rank_exact": effective_rank_hhi(X_exact),
        "eff_rank_noisy": effective_rank_hhi(X_noisy),
        "condition_exact": condition_number(X_exact),
        "condition_noisy": condition_number(X_noisy),
    })
    return rep
