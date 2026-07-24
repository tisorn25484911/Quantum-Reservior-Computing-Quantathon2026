"""enso_ode.py -- Vallis (1986/1988) box model = Hamhoum 'ENSO' ODE; audit (G8-2/3).

Hamhoum's "ENSO" system is NOT the observational enso.csv (spec item 21) -- it is a
GENERATED trajectory of the Vallis conceptual box model. Per G8-2/3 we (a) never
transfer Hamhoum's quoted LLE 0.05-0.1 (that figure is from an observational
coastal-temperature study, not this ODE), and (b) compute lambda ourselves.

State s = [u, Te, Tw] (zonal current, east/west SST). Dimensional box model:

    du/dt  = B (Te - Tw)/(2 dx) - C (u - u*)
    dTe/dt = u (Tw - Tbar)/(2 dx) - A (Te - T*)
    dTw/dt = u (Tbar - Te)/(2 dx) - A (Tw - T*)

with the parameters printed in Hamhoum Eq. 13: B=940, dx=7.5, C=3, u*=-14.2,
Tbar=16, A=1, T*=28. The relaxation rate A=1 sets the time unit; "steps" are
therefore reported alongside Lyapunov times (LT = 1/lambda_max in the same unit),
never converted to calendar months (the ODE fixes no calendar).

AUDIT NOTE (G8-3): the primary Vallis 1986/1988 texts are scanned PDFs that could
not be machine-read in-session; the sign convention above is the form reproduced in
the secondary Vallis-chaos literature and is VERIFIED to reproduce the claimed
chaotic behaviour numerically here (lambda_max > 0). Documented, not silent.
Implemented in P8.
"""
from __future__ import annotations

import numpy as np

from .integration import integrate, benettin_largest_ode

# Hamhoum Eq. 13 parameters (audited numerically, see module docstring).
PARAMS = dict(B=940.0, dx=7.5, C=3.0, u_star=-14.2, Tbar=16.0, A=1.0, T_star=28.0)


def field(B=940.0, dx=7.5, C=3.0, u_star=-14.2, Tbar=16.0, A=1.0, T_star=28.0):
    """Vallis box-model vector field f([u, Te, Tw])."""
    two_dx = 2.0 * dx

    def f(s):
        u, Te, Tw = s
        du = B * (Te - Tw) / two_dx - C * (u - u_star)
        dTe = u * (Tw - Tbar) / two_dx - A * (Te - T_star)
        dTw = u * (Tbar - Te) / two_dx - A * (Tw - T_star)
        return np.array([du, dTe, dTw])
    return f


def jacobian(B=940.0, dx=7.5, C=3.0, u_star=-14.2, Tbar=16.0, A=1.0, T_star=28.0):
    """Exact Jacobian J([u, Te, Tw])."""
    two_dx = 2.0 * dx

    def jac(s):
        u, Te, Tw = s
        return np.array([
            [-C, B / two_dx, -B / two_dx],
            [(Tw - Tbar) / two_dx, -A, u / two_dx],
            [(Tbar - Te) / two_dx, -u / two_dx, -A],
        ])
    return jac


def generate(n=6000, dt=0.005, washout=4000, params=None, x0=(0.0, 25.0, 27.0),
             component=1, scale=True):
    """Integrate the Vallis box model; return a scalar ``series`` (default Te).

    ``component`` 0=u, 1=Te, 2=Tw. ``series`` is scaled to [0,1] on its own range
    for the QRC encoder. ``sample_dt`` is ``dt`` (no subsampling), so Lyapunov times
    convert consistently.
    """
    p = dict(PARAMS if params is None else params)
    traj = integrate(field(**p), np.asarray(x0, float), dt, n + washout, washout)
    raw = traj[:, component]
    if scale:
        lo, hi = float(raw.min()), float(raw.max())
        span = hi - lo if hi > lo else 1.0
        series = (raw - lo) / span
        bounds = (lo, hi)
    else:
        series, bounds = raw.copy(), None
    return {"series": series, "trajectory": traj, "sample_dt": dt,
            "bounds": bounds, "params": p}


def lyapunov_largest(params=None, dt=0.005, n_steps=40000, washout=4000,
                     x0=(0.0, 25.0, 27.0)):
    """Largest Lyapunov exponent (per unit time) via Benettin QR on the flow."""
    p = dict(PARAMS if params is None else params)
    return benettin_largest_ode(field(**p), jacobian(**p), np.asarray(x0, float),
                                dt, n_steps, washout=washout)


def audit(dt=0.005, n_steps=40000, washout=4000):
    """G8-2/3: compute lambda_max ourselves; do NOT reuse the 0.05-0.1 figure."""
    lam = lyapunov_largest(dt=dt, n_steps=n_steps, washout=washout)
    return {"params": PARAMS, "lambda_max": lam,
            "lyapunov_time": (1.0 / lam) if lam > 0 else None,
            "chaotic": bool(lam > 1e-3),
            "note": "computed in-repo; Hamhoum's observational 0.05-0.1 NOT reused"}
