"""prediction_horizon.py -- H_error/H_skill/H_reliable/H_effective + survival.

Autonomous valid-horizon metrics (spec s17). All horizons are counted in steps.

H_error(NE, eps, K): the number of leading steps before a SUSTAINED failure -- the
first run of K consecutive breaches NE > eps. The prefix rule is load-bearing: once
a sustained failure begins the trajectory has failed for good; a later dip back
under eps does NOT re-validate it (the first sustained run wins). If no sustained
run occurs, the horizon is the full length.

H_skill: leading steps for which the model's cumulative error beats a baseline's
(CNRMSE_model < CNRMSE_base), same sustained convention.
H_reliable: leading steps before the first failure-mode flag (failure_modes).
H_effective = min(H_error, H_skill, H_reliable) -- but every component is returned,
never only the minimum (spec 17.6). Survival S(h) and H_product(p) aggregate
H_effective across origins. Implemented in P4.
"""
from __future__ import annotations

import numpy as np


def _first_sustained_breach(mask, K):
    """Index (0-based) of the first element of the first run of K consecutive True.

    Returns len(mask) if no such run exists.
    """
    mask = np.asarray(mask, dtype=bool)
    run = 0
    for i, b in enumerate(mask):
        run = run + 1 if b else 0
        if run >= K:
            return i - K + 1
    return len(mask)


def h_error(ne, eps, K=3):
    """Leading valid steps before a sustained (K-in-a-row) breach of NE > eps."""
    ne = np.asarray(ne, dtype=float)
    return int(_first_sustained_breach(ne > eps, K))


def h_skill(cnrmse_model, cnrmse_base, K=3):
    """Leading steps where the model sustainedly beats the baseline's CNRMSE."""
    a, b = np.asarray(cnrmse_model, float), np.asarray(cnrmse_base, float)
    return int(_first_sustained_breach(a >= b, K))


def h_reliable(first_failure_step, n_steps):
    """Leading steps before the first failure-mode flag (None -> full length)."""
    return int(n_steps if first_failure_step is None else first_failure_step)


def h_effective(h_err, h_skl, h_rel):
    """min of the components, with all components returned (spec 17.6)."""
    he = int(min(h_err, h_skl, h_rel))
    return {"H_effective": he, "H_error": int(h_err),
            "H_skill": int(h_skl), "H_reliable": int(h_rel)}


def survival_curve(h_effectives, max_h):
    """S(h) = fraction of origins whose H_effective >= h, for h = 1..max_h."""
    he = np.asarray(h_effectives, dtype=int)
    hs = np.arange(1, max_h + 1)
    return hs, np.array([(he >= h).mean() for h in hs])


def h_product(h_effectives, max_h, p):
    """Largest horizon h with survival S(h) >= p (0 if none)."""
    hs, s = survival_curve(h_effectives, max_h)
    ok = hs[s >= p]
    return int(ok.max()) if ok.size else 0
