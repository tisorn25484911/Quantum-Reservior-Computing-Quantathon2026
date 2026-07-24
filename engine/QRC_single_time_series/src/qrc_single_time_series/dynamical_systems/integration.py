"""integration.py -- shared RK4 ODE integration + continuous Benettin spectrum.

The controlled-system campaign (P8) needs (a) a deterministic fixed-step
integrator and (b) a Lyapunov-spectrum estimator for continuous flows so the
Lorenz beta-audit (G8-1) and the Vallis-ENSO exponent (G8-2/3) are computed
IN-REPO, never transferred from a paper. The Benettin method integrates the
trajectory together with ``k`` tangent vectors under the variational (Jacobian)
flow, periodically QR-reorthonormalises them, and accumulates the log of the R
diagonal; lambda_i = (1/T) sum log R_ii (per unit time). Implemented in P8.
"""
from __future__ import annotations

import numpy as np


def rk4_step(f, x, dt, t=0.0):
    """One classical RK4 step of dx/dt = f(x[, t])."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(f, x0, dt, n_steps, washout=0):
    """Integrate dx/dt=f(x) with RK4; return the trajectory AFTER ``washout`` steps.

    Returns array of shape ``(n_steps - washout + 1, d)`` (includes the state at the
    washout boundary as row 0).
    """
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(washout):
        x = rk4_step(f, x, dt)
    out = [x.copy()]
    for _ in range(n_steps - washout):
        x = rk4_step(f, x, dt)
        out.append(x.copy())
    return np.array(out)


def benettin_ode(f, jac, x0, dt, n_steps, k=None, washout=0, renorm_every=1):
    """Lyapunov spectrum (per unit time) of a continuous flow via the QR method.

    ``f(x)`` is the vector field, ``jac(x)`` its (d,d) Jacobian. ``k`` tangent
    vectors are co-integrated with an RK4-consistent linearisation (the Jacobian is
    applied to the tangent frame at each substage). Returns exponents descending.
    """
    x = np.asarray(x0, dtype=float).copy()
    d = len(x)
    k = d if k is None else k
    Q = np.eye(d)[:, :k]
    acc = np.zeros(k)
    count = 0

    def tangent_rk4(x, Q, dt):
        # RK4 on the coupled (state, tangent) system: dQ/dt = J(x) Q
        def state(x):
            return f(x)

        k1x, k1q = f(x), jac(x) @ Q
        k2x, k2q = f(x + 0.5 * dt * k1x), jac(x + 0.5 * dt * k1x) @ (Q + 0.5 * dt * k1q)
        k3x, k3q = f(x + 0.5 * dt * k2x), jac(x + 0.5 * dt * k2x) @ (Q + 0.5 * dt * k2q)
        k4x, k4q = f(x + dt * k3x), jac(x + dt * k3x) @ (Q + dt * k3q)
        xn = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        Qn = Q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
        return xn, Qn

    for i in range(n_steps):
        x, Q = tangent_rk4(x, Q, dt)
        if (i + 1) % renorm_every == 0:
            Q, Rm = np.linalg.qr(Q)
            sign = np.sign(np.diag(Rm))
            sign[sign == 0] = 1.0
            Q = Q * sign
            if i >= washout:
                acc += np.log(np.abs(np.diag(Rm)))
                count += 1
    T = count * renorm_every * dt
    return np.sort(acc / T)[::-1] if T > 0 else np.full(k, np.nan)


def benettin_largest_ode(f, jac, x0, dt, n_steps, **kw):
    """Largest Lyapunov exponent (per unit time) of a continuous flow."""
    return float(benettin_ode(f, jac, x0, dt, n_steps, k=1, **kw)[0])
