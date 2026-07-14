"""regime_router.py -- clear-sky classification + changepoint gate
(Phase 10). Regime-conditional error reporting doubles as the router's
evaluation; Track A/B routing is the same idea one level up.
"""

from __future__ import annotations

import numpy as np

REGIMES = ("clear", "convective", "overcast", "night")


def classify_regime(kt_window: np.ndarray) -> str:
    """Rule-based clear-sky classification on a recent k_t window.
    NaN-dominated window = night; high mean + low var = clear; low mean
    = overcast; else convective (the ramp regime)."""
    kt = np.asarray(kt_window, dtype=float)
    finite = kt[np.isfinite(kt)]
    if len(finite) < max(3, len(kt) // 4):
        return "night"
    m, s = float(np.mean(finite)), float(np.std(finite))
    if m > 0.8 and s < 0.08:
        return "clear"
    if m < 0.35:
        return "overcast"
    return "convective"


def changepoint_score(x: np.ndarray, w: int = 12) -> float:
    """CUSUM-style two-window mean shift score (in pooled-sigma units)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 * w:
        return 0.0
    a, b = x[-2 * w:-w], x[-w:]
    s = np.sqrt(0.5 * (a.var() + b.var())) + 1e-9
    return float(abs(b.mean() - a.mean()) / s)


def regime_conditional_errors(errors: np.ndarray,
                              regimes: list[str]) -> dict:
    out = {}
    e = np.asarray(errors, dtype=float)
    r = np.asarray(regimes)
    for reg in REGIMES:
        sel = r == reg
        out[reg] = {"n": int(sel.sum()),
                    "mae": float(np.nanmean(np.abs(e[sel])))
                    if sel.any() else float("nan")}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    assert classify_regime(np.full(12, np.nan)) == "night"
    assert classify_regime(0.95 + 0.01 * rng.normal(size=12)) == "clear"
    assert classify_regime(np.full(12, 0.2)) == "overcast"
    assert classify_regime(0.6 + 0.25 * rng.normal(size=12)) == "convective"
    x = np.r_[np.zeros(24), np.ones(12)]
    assert changepoint_score(x) > 3.0
    print("regime router self-test PASS")
