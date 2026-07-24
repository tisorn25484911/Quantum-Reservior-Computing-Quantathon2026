"""mackey_glass.py -- Fujii-Nakajima Mackey-Glass generator (tau_MG in {16,17}).

The Mackey-Glass delay differential equation

    dx/dt = alpha * x(t-tau_MG) / (1 + x(t-tau_MG)^beta) - gamma * x(t)

integrated by FN's exact discrete recipe: forward Euler with step sigma=0.1,
recorded every ``subsample=10`` Euler steps (so the sample interval is 1.0), a
long washout, then a linear rescale to [0,1] for the paper-faithful encoder. With
alpha=0.2, beta=10, gamma=0.1 the system is a stable limit cycle at tau_MG=16 and
chaotic at tau_MG=17 (H1). Implemented in P5.
"""
from __future__ import annotations

import numpy as np


def _euler_trajectory(tau_MG, n_euler, dt, alpha, beta, gamma, x_init, rng=None,
                      perturb=None):
    """Raw Euler trajectory (one value per Euler step), constant initial history."""
    delay = int(round(tau_MG / dt))
    x = np.empty(n_euler + 1)
    # constant history for t in [-tau_MG, 0]
    x[0] = x_init
    hist = np.full(delay + 1, x_init)
    buf = list(hist)
    val = x_init
    traj = [val]
    if perturb is not None:
        val = val + perturb
        buf[-1] = val
        traj[0] = val
    for t in range(n_euler):
        xd = buf[-delay]
        val = val + dt * (alpha * xd / (1.0 + xd ** beta) - gamma * val)
        buf.append(val)
        traj.append(val)
    return np.asarray(traj)


def generate(tau_MG=17, n=6000, washout=2000, dt=0.1, subsample=10,
             alpha=0.2, beta=10.0, gamma=0.1, x_init=0.5, scale=True):
    """Generate a Mackey-Glass series (FN recipe).

    Returns ``dict`` with ``series`` (scaled to [0,1] if ``scale``), ``raw`` (before
    scaling), ``sample_dt`` and the scaling ``bounds``. The number of returned
    samples is ``n`` (post-washout).
    """
    total_samples = washout + n
    n_euler = total_samples * subsample
    raw_full = _euler_trajectory(tau_MG, n_euler, dt, alpha, beta, gamma, x_init)
    sampled = raw_full[::subsample][:total_samples]
    raw = sampled[washout:]
    if scale:
        lo, hi = float(raw.min()), float(raw.max())
        span = hi - lo if hi > lo else 1.0
        series = (raw - lo) / span
        bounds = (lo, hi)
    else:
        series, bounds = raw.copy(), None
    return {"series": series, "raw": raw, "sample_dt": dt * subsample,
            "bounds": bounds, "tau_MG": tau_MG}


def lyapunov_perturbation_pair(tau_MG, n=4000, washout=2000, dt=0.1, subsample=10,
                               alpha=0.2, beta=10.0, gamma=0.1, x_init=0.5,
                               d0=1e-8, renorm_every=10):
    """FN perturbation-pair estimate of the largest Lyapunov exponent (per sample).

    The state of a delay system is the whole history segment of length ``delay``,
    so the perturbation and its renormalisation act on that FULL window, not just
    the current value. Two trajectories a distance ``d0`` apart (in the delay
    window) are advanced; every ``renorm_every`` Euler steps the separation is
    measured over the window, the mean log growth accumulated, and the perturbed
    window rescaled back to ``d0``. Positive => chaotic. Units: per sample interval.
    """
    delay = int(round(tau_MG / dt))
    steps = (washout + n) * subsample

    bufA = [x_init] * (delay + 1)
    bufB = [x_init] * (delay + 1)
    bufB[-1] += d0
    vA, vB = bufA[-1], bufB[-1]
    logsum, count = 0.0, 0
    warm = washout * subsample
    for step in range(steps):
        xdA, xdB = bufA[-delay], bufB[-delay]
        vA = vA + dt * (alpha * xdA / (1 + xdA ** beta) - gamma * vA)
        vB = vB + dt * (alpha * xdB / (1 + xdB ** beta) - gamma * vB)
        bufA.append(vA)
        bufB.append(vB)
        if step >= warm and step % renorm_every == 0:
            wA = np.asarray(bufA[-delay:])
            wB = np.asarray(bufB[-delay:])
            diff = wB - wA
            d = np.linalg.norm(diff)
            if d > 0:
                logsum += np.log(d / d0)
                count += 1
                wB = wA + diff * (d0 / d)          # rescale the FULL window
                bufB[-delay:] = list(wB)
                vB = bufB[-1]
    per_euler = logsum / (count * renorm_every) if count else 0.0
    return float(per_euler * subsample)            # per sample interval
