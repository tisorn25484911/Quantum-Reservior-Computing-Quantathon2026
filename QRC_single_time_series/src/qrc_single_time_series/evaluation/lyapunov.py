"""lyapunov.py -- Rosenstein scalar estimator (P5) + Benettin/Jacobian machinery (P6).

Phase 5 needs the ported scalar largest-Lyapunov estimator (Rosenstein et al.
1993): delay-embed a scalar series, track the mean log divergence of
nearest-neighbour pairs, and read the largest exponent off the slope of the linear
region. Validated on the logistic map (r=3.9, lambda ~ 0.494/iterate) before it is
pointed at any generated series. The learned-map Jacobian / Benettin QR spectrum
lands in P6. Implemented (Rosenstein part) in P5.
"""
from __future__ import annotations

import numpy as np


def delay_embed(x, m, lag):
    """Time-delay embedding of a scalar series into R^m with delay ``lag``."""
    x = np.asarray(x, dtype=float)
    M = len(x) - (m - 1) * lag
    if M <= 1:
        raise ValueError("series too short for the requested embedding")
    return np.stack([x[i * lag: i * lag + M] for i in range(m)], axis=1)


def rosenstein(x, m=4, lag=6, mean_period=12, max_t=40, fit_range=None):
    """Largest Lyapunov exponent (per sample) via Rosenstein's method.

    ``mean_period`` sets the Theiler window that excludes temporally-close points
    from the nearest-neighbour search. Returns ``(lambda_hat, divergence_curve)``;
    the slope is fit over ``fit_range`` (defaults to the first third of ``max_t``).
    """
    emb = delay_embed(x, m, lag)
    M = len(emb)
    theiler = mean_period
    nn = np.empty(M, dtype=int)
    for i in range(M):
        d = np.linalg.norm(emb - emb[i], axis=1)
        d[max(0, i - theiler): i + theiler + 1] = np.inf
        nn[i] = int(np.argmin(d))

    div = np.full(max_t, np.nan)
    for k in range(max_t):
        vals = []
        for i in range(M):
            j = nn[i]
            if i + k < M and j + k < M:
                dd = np.linalg.norm(emb[i + k] - emb[j + k])
                if dd > 0:
                    vals.append(np.log(dd))
        if vals:
            div[k] = np.mean(vals)
    if fit_range is None:
        fit_range = (1, max(3, max_t // 3))
    lo, hi = fit_range
    ks = np.arange(lo, hi)
    slope = float(np.polyfit(ks, div[lo:hi], 1)[0])
    return slope, div


def logistic_series(r=3.9, n=4000, x0=0.4, burn=500):
    """Logistic-map series x_{t+1} = r x_t (1-x_t) for estimator validation."""
    x = np.empty(n + burn)
    x[0] = x0
    for i in range(1, n + burn):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[burn:]


# --- learned-map Lyapunov machinery (P6) -----------------------------------
def jacobian(fmap, x, eps=1e-6):
    """Central finite-difference Jacobian of a vector map ``fmap`` at ``x``.

    ``fmap`` maps R^d -> R^d. Step is ``eps`` scaled per coordinate. Returns (d,d).
    """
    x = np.asarray(x, dtype=float)
    d = len(x)
    f0 = np.asarray(fmap(x), dtype=float)
    J = np.empty((len(f0), d))
    for j in range(d):
        h = eps * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (np.asarray(fmap(xp)) - np.asarray(fmap(xm))) / (2 * h)
    return J


def benettin_spectrum(step_map, x0, n_steps=200, warmup=50, eps=1e-6, jac_fn=None,
                      k=None):
    """Lyapunov spectrum (per step) of an autonomous map via Benettin QR product.

    ``step_map``: R^d -> R^d one-step map. ``jac_fn`` optionally supplies an exact
    Jacobian; otherwise a central finite difference is used. Returns exponents in
    descending order. Positive largest exponent => chaotic.
    """
    x = np.asarray(x0, dtype=float).copy()
    d = len(x)
    k = d if k is None else k
    Q = np.eye(d)[:, :k]
    acc = np.zeros(k)
    count = 0
    for i in range(n_steps):
        J = jac_fn(x) if jac_fn is not None else jacobian(step_map, x, eps)
        Z = J @ Q
        Q, Rm = np.linalg.qr(Z)
        sign = np.sign(np.diag(Rm))
        sign[sign == 0] = 1.0
        Q = Q * sign
        diagR = np.abs(np.diag(Rm))
        if i >= warmup:
            acc += np.log(np.where(diagR > 0, diagR, 1e-300))
            count += 1
        x = np.asarray(step_map(x), dtype=float)
    return np.sort(acc / max(count, 1))[::-1]


def benettin_largest(step_map, x0, **kw):
    """Largest Lyapunov exponent (per step) via ``benettin_spectrum``."""
    return float(benettin_spectrum(step_map, x0, **kw)[0])


def delay_map_from_prediction(predict_fn, clip=None):
    """Build the delay map G(window) = [window[1:], predict_fn(window)].

    ``predict_fn`` maps a length-t_w window to the next value (the learned F). The
    returned G advances the autonomous recursion in delay coordinates. ``clip`` is
    intentionally optional and OFF by default: Lyapunov runs must not differentiate
    through a hard clip (spec 20.3) -- operate on verified-interior trajectories.
    """
    def G(window):
        w = np.asarray(window, dtype=float)
        y = float(predict_fn(w))
        if clip is not None:
            y = float(np.clip(y, clip[0], clip[1]))
        return np.concatenate([w[1:], [y]])
    return G


def benettin_largest_interior(step_map, x0, bounds=(0.0, 1.0), margin=1e-3,
                              n_steps=120, warmup=20, eps=1e-5):
    """Largest exponent of a learned map on a VERIFIED-INTERIOR trajectory.

    The learned autonomous map may leave its domain (where a hard clip would make
    it non-smooth). Following spec 20.3 the estimate is accumulated only while every
    coordinate stays inside ``(lo+margin, hi-margin)``; the run stops at the first
    boundary contact. Returns ``(lambda_hat, n_interior_steps)`` -- lambda is NaN if
    too few interior steps were collected.
    """
    lo, hi = bounds
    x = np.asarray(x0, dtype=float).copy()
    d = len(x)
    q = np.ones(d) / np.sqrt(d)
    acc, count = 0.0, 0
    for i in range(n_steps):
        if np.any(x <= lo + margin) or np.any(x >= hi - margin):
            break
        J = jacobian(step_map, x, eps)
        v = J @ q
        nrm = np.linalg.norm(v)
        if nrm == 0:
            break
        q = v / nrm
        if i >= warmup:
            acc += np.log(nrm)
            count += 1
        x = np.asarray(step_map(x), dtype=float)
    lam = acc / count if count >= 10 else float("nan")
    return lam, count
