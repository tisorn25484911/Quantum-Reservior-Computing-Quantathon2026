"""chaos.py -- predictability analysis: how far ahead a series can be forecast.

Wraps ``DataBase_Analysis.lyapunov`` and ``.fourier`` for the web layer. The
question it answers is the one that has to be settled *before* any forecast
result is interpretable:

    Given this series, what is the horizon beyond which no model -- quantum,
    classical, or otherwise -- can predict, because the dynamics themselves
    destroy the information?

That ceiling is the Lyapunov time ``T_lambda = 1 / lambda_1``, converted to
steps as ``H_max = T_lambda / dt``. A forecast that keeps skill well past it is
evidence of leakage, not of a good model; one that dies far short of it is
underperforming the physics rather than hitting a wall.

**The estimate is not always trustworthy, and the page has to say which.**
``lyapunov.py`` distinguishes two routes: an *ensemble* estimate from
independent realizations (available only for the simulated chaotic tier, and
comparable against a published lambda_1) and a single-trajectory *Rosenstein*
estimate for everything else. Rosenstein on a strongly periodic or seasonal
record is inflated -- it reads the seasonal cycle as divergence -- so it is
reported as an upper bound, never as a measurement.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .catalog import load_series, describe

import lyapunov as _lyap          # noqa: E402  (sys.path wired in config)
import fourier as _fourier        # noqa: E402


def _quality(est, meta: dict) -> tuple[str, list[str]]:
    """Verdict on how much the number can be leaned on, plus the caveats."""
    notes: list[str] = []

    if est.method == "ensemble":
        verdict = "reliable"
        notes.append(
            f"Estimated from {est.n_realizations} independent realizations, "
            "which is the trustworthy route: separate trajectories diverge "
            "from genuinely independent initial conditions.")
        if est.truth is not None and est.error is not None:
            notes.append(
                f"Published lambda_1 = {est.truth:.4g}; this estimate is off "
                f"by {est.error * 100:+.1f}% -- a direct accuracy check the "
                "other tiers cannot offer.")
    else:
        verdict = "upper bound"
        notes.append(
            "Single-trajectory Rosenstein estimate. It cannot separate true "
            "exponential divergence from the series simply moving through its "
            "cycle, so read it as an upper bound on lambda_1 -- i.e. a LOWER "
            "bound on predictability.")

    rho = meta.get("lag1_autocorr")
    if rho is not None and np.isfinite(rho) and rho > 0.9 and \
            est.method != "ensemble":
        notes.append(
            f"This record is strongly autocorrelated (lag-1 rho = {rho:.2f}) "
            "and likely seasonal, the case where Rosenstein is known to "
            "inflate. Treat H_max as a scale, not a measurement, and set the "
            "working horizon from measured forecast-skill decay instead.")
        verdict = "unreliable (seasonal)"

    if not np.isfinite(est.lyap) or est.lyap <= 0:
        verdict = "no positive exponent"
        notes.append(
            "No positive lambda_1 was found: there is no exponential error "
            "growth, so the dynamics impose no predictability horizon. Any "
            "forecast limit here comes from noise, not chaos.")
    return verdict, notes


@lru_cache(maxsize=32)
def _analyse_cached(key: str, max_points: int) -> dict:
    s = load_series(key)
    meta = describe(key)
    est = _lyap.estimate(s)
    spec = _fourier.spectrum(s)
    peaks = _fourier.dominant_periods(spec.f, spec.amp, k=4)

    H_max = (est.lyap_time / s.dt
             if np.isfinite(est.lyap_time) and s.dt else float("inf"))
    verdict, notes = _quality(est, meta)

    return {
        "key": key, "dataset": meta,
        "lambda1": float(est.lyap),
        "lyapunov_time": float(est.lyap_time),
        "H_max_steps": float(H_max),
        "method": est.method,
        "reliable": bool(est.reliable),
        "verdict": verdict,
        "notes": notes,
        "embedding": (None if est.embedding is None
                      else {"m": int(est.embedding[0]),
                            "tau": int(est.embedding[1])}),
        "truth": (None if est.truth is None else float(est.truth)),
        "error": (None if est.error is None else float(est.error)),
        "n_realizations": int(est.n_realizations),
        "record_length_in_T_lambda": (
            float(len(s.x) * s.dt / est.lyap_time)
            if np.isfinite(est.lyap_time) and est.lyap_time > 0 else None),
        "dominant_periods": [
            {"period": float(p["period"]), "amp": float(p["amp"]),
             "f": float(p["f"])} for p in peaks],
        # Curves for the figure; lists so the result stays JSON-safe.
        "t": np.asarray(est.t, float).tolist(),
        "curve": np.asarray(est.curve, float).tolist(),
        "window": [int(est.window[0]), int(est.window[1])],
        "spectrum_f": np.asarray(spec.f, float).tolist(),
        "spectrum_amp": np.asarray(spec.amp, float).tolist(),
    }


def analyse(key: str, max_points: int = 4000) -> dict:
    """Predictability analysis for one dataset. Cached -- the fit is ~1 s."""
    return _analyse_cached(key, int(max_points))


def horizon_budget(key: str, horizon_steps: int) -> dict:
    """Express a forecast horizon in Lyapunov units.

    ``h / H_max`` is the honest way to compare horizons across series with
    different clocks: one unit is one e-folding of an initial error, so 0.5
    means "half an e-folding ahead" whether the step is a month or a minute.
    """
    a = analyse(key)
    hmax = a["H_max_steps"]
    frac = (horizon_steps / hmax) if np.isfinite(hmax) and hmax > 0 else None
    return {"horizon_steps": int(horizon_steps), "H_max_steps": hmax,
            "fraction_of_lyapunov_time": frac, "verdict": a["verdict"],
            "reliable": a["reliable"]}
