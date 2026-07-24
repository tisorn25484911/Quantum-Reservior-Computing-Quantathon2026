"""lorenz63.py -- Lorenz-63 integrator and the beta = 8/3 vs printed 3/8 audit (G8-1).

Hamhoum et al. print the Lorenz parameters as (sigma, rho, beta) = (10, 28, 3/8),
but the standard chaotic Lorenz uses beta = 8/3 and their quoted LLE ~ 0.9 is the
8/3 figure. Per G8-1 we integrate BOTH, compute the largest Lyapunov exponent
ourselves (Benettin QR on the exact Jacobian), and document which reproduces the
~0.906 value -- never trusting the printed constant OR this repo's memory.

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z

A scalar component (default x) is exposed for the Rosenstein cross-check and for
the reduced-to-univariate autonomous forecasting task. Implemented in P8.
"""
from __future__ import annotations

import numpy as np

from .integration import integrate, benettin_largest_ode

BETA_STANDARD = 8.0 / 3.0
BETA_PRINTED = 3.0 / 8.0


def field(sigma=10.0, rho=28.0, beta=BETA_STANDARD):
    """Return the Lorenz vector field f(x) as a closure."""
    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    return f


def jacobian(sigma=10.0, rho=28.0, beta=BETA_STANDARD):
    """Return the exact Jacobian J(x) as a closure."""
    def jac(s):
        x, y, z = s
        return np.array([[-sigma, sigma, 0.0],
                         [rho - z, -1.0, -x],
                         [y, x, -beta]])
    return jac


def generate(n=6000, dt=0.01, washout=2000, sigma=10.0, rho=28.0,
             beta=BETA_STANDARD, x0=(1.0, 1.0, 1.0), component=0, scale=True):
    """Integrate Lorenz-63; return dict with full trajectory + a scalar ``series``.

    ``series`` is the ``component`` coordinate (0=x); scaled to [0,1] on its own
    range if ``scale`` (for the [0,1] QRC encoder). ``dt`` is the integration step
    and the sample interval (no subsampling), so Lyapunov times convert as
    LT = 1/(lambda_max) in the SAME time unit.
    """
    traj = integrate(field(sigma, rho, beta), np.asarray(x0, float), dt, n + washout,
                     washout)
    raw = traj[:, component]
    if scale:
        lo, hi = float(raw.min()), float(raw.max())
        span = hi - lo if hi > lo else 1.0
        series = (raw - lo) / span
        bounds = (lo, hi)
    else:
        series, bounds = raw.copy(), None
    return {"series": series, "trajectory": traj, "sample_dt": dt,
            "bounds": bounds, "beta": beta}


def lyapunov_largest(sigma=10.0, rho=28.0, beta=BETA_STANDARD, dt=0.01,
                     n_steps=20000, washout=2000, x0=(1.0, 1.0, 1.0)):
    """Largest Lyapunov exponent (per unit time) via Benettin QR on the flow."""
    return benettin_largest_ode(field(sigma, rho, beta), jacobian(sigma, rho, beta),
                                np.asarray(x0, float), dt, n_steps, washout=washout)


def beta_audit(dt=0.01, n_steps=20000, washout=2000):
    """G8-1: lambda_max for beta=8/3 and beta=3/8; which reproduces ~0.906?"""
    out = {}
    for name, beta in (("beta_8_3", BETA_STANDARD), ("beta_3_8", BETA_PRINTED)):
        lam = lyapunov_largest(beta=beta, dt=dt, n_steps=n_steps, washout=washout)
        out[name] = {"beta": beta, "lambda_max": lam,
                     "lyapunov_time": (1.0 / lam) if lam > 0 else None}
    out["reproduces_0.906"] = ("beta_8_3" if abs(out["beta_8_3"]["lambda_max"] - 0.906)
                               < abs(out["beta_3_8"]["lambda_max"] - 0.906)
                               else "beta_3_8")
    return out
