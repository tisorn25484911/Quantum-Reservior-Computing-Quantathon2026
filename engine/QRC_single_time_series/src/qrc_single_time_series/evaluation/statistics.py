"""statistics.py -- block bootstrap + Diebold-Mariano (Harvey correction).

Uncertainty over forecast ORIGINS (spec s16/s23). The short observational records
mean origins overlap, so CIs use a stationary (Politis-Romano) moving-block
bootstrap rather than an iid one. Forecast-comparison significance uses the
Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.
Implemented in P4.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def stationary_bootstrap_ci(values, block_length, n_resamples=2000, alpha=0.10,
                            statistic=np.mean, seed=0):
    """Stationary-bootstrap CI for a statistic of per-origin ``values``.

    Blocks have geometric length with mean ``block_length`` (Politis-Romano).
    Returns ``(point, lo, hi)`` for a central (1-alpha) interval.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("empty values")
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block_length, 1)
    boot = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(n)         # start a new block
            else:
                i = (i + 1) % n             # continue the block (wrap)
        boot[b] = statistic(x[idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(statistic(x)), float(lo), float(hi)


def skill_bootstrap_ci(se_model, se_base, block_length, n_resamples=2000,
                       alpha=0.10, seed=0):
    """Block-bootstrap point + (1-alpha) CI of the persistence skill score.

    skill = 1 - sum(se_model)/sum(se_base) over paired per-time squared errors, with
    the numerator/denominator resampled TOGETHER (a paired stationary block bootstrap)
    so the ratio's origin-overlap dependence is respected. This is the Gate-1
    statistic on the untouched test span (spec s25.7). Returns ``(point, lo, hi)``.
    """
    se_model = np.asarray(se_model, dtype=float)
    se_base = np.asarray(se_base, dtype=float)
    n = len(se_model)
    if n == 0:
        raise ValueError("empty error arrays")
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block_length, 1)

    def skill(idx):
        b = se_base[idx].sum()
        return 1.0 - se_model[idx].sum() / b if b > 0 else 0.0

    point = skill(np.arange(n))
    boot = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        boot[k] = skill(idx)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def diebold_mariano(loss_a, loss_b, h=1):
    """Diebold-Mariano statistic with Harvey small-sample correction.

    ``loss_a``/``loss_b`` are per-observation losses of two forecasts at horizon
    ``h``. Returns ``(DM_stat, p_value)`` (two-sided, Student-t with n-1 dof).
    Positive DM means A has the LARGER loss (B is better).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if n < 2:
        raise ValueError("need >= 2 observations")
    dbar = d.mean()
    # autocovariances up to lag h-1 (h-step forecasts have MA(h-1) errors)
    gamma0 = np.var(d, ddof=0)
    var = gamma0
    for k in range(1, h):
        if k < n:
            gk = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
            var += 2.0 * gk
    var = var / n
    if var <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var)
    # Harvey-Leybourne-Newbold correction
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    p = 2.0 * (1.0 - sps.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)
