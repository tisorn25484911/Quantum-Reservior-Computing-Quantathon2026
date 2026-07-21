"""chaos.py -- predictability analysis, delegated to DataBase_Analysis.

This module is a **thin adapter, not an implementation**. The analysis already
exists in ``Quantathon_stack/DataBase_Analysis`` and is the version the project's
figures and reports were produced with, so the web layer calls it rather than
re-deriving anything:

    analysis.analyse_one(series)   the canonical single-dataset pass
    visualizer.figure(...)         the canonical 4-panel figure
    provenance.DOCS                per-dataset source and gotchas
    compare.collect / figure       the cross-dataset comparison

Delegating matters here for a measured reason. An earlier version of this file
called ``fourier.dominant_periods`` directly, without the ``fmin = 2 / (N*dt)``
guard ``analyse_one`` applies before picking peaks. That guard drops the lowest
frequency bins, where residual near-DC leakage otherwise outranks every real
spectral line. Checked across all 17 datasets, omitting it changes the top three
periods on **five** of them:

    lorenz63, rikitake, tao, nyc, cuxhaven

and the spurious entries are the giveaway -- ``nyc`` reports a 57540-day
(~158 year) "dominant period" on a record nowhere near that long. ``analyse_one``
also carries hand-tuned per-dataset axis limits (``FMAX``, ``MAX_UNITS``).
Re-implementing silently lost all of it.

The question this page answers: given a series, what is the horizon beyond which
*no* model can predict, because the dynamics destroy the information? That is the
Lyapunov time ``T_lambda = 1/lambda_1``, i.e. ``H_max = T_lambda / dt`` steps.
Skill surviving well past it indicates leakage; skill dying far short of it means
the model, not the physics, is the limit.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .catalog import load_series, describe
from ..config import RESULTS_DIR

import analysis as _analysis      # noqa: E402  (sys.path wired in config)
import fourier as _fourier        # noqa: E402
import provenance as _provenance  # noqa: E402
import visualizer as _viz         # noqa: E402

FIGURE_CACHE = RESULTS_DIR / "chaos_figures"
FIGURE_CACHE.mkdir(parents=True, exist_ok=True)


def _jsonable(v):
    """Map non-finite floats to None so the payload is valid JSON.

    ``lyapunov_time`` is legitimately ``inf`` when lambda_1 is non-positive --
    that is the module's way of saying "no exponential error growth, so no
    predictability horizon". JSON has no encoding for it, and FastAPI's strict
    serialiser rejects it outright, so it becomes ``null`` here. The meaning is
    not lost: ``verdict`` says "no positive exponent" and ``positive`` carries
    the flag.
    """
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


def _quality(est, meta: dict) -> tuple[str, list[str]]:
    """How much the estimate can be leaned on, and why."""
    notes: list[str] = []

    if est.method == "ensemble":
        verdict = "reliable"
        notes.append(
            f"Estimated from {est.n_realizations} independent realizations -- "
            "the trustworthy route, since separate trajectories diverge from "
            "genuinely independent initial conditions.")
        if est.truth is not None and est.error is not None:
            notes.append(
                f"Published lambda_1 = {est.truth:.4g}; this estimate is off by "
                f"{est.error * 100:+.1f}%. A direct accuracy check the other "
                "tiers cannot offer.")
    else:
        verdict = "upper bound"
        notes.append(
            "Single-trajectory Rosenstein estimate: it cannot separate true "
            "exponential divergence from the series moving through its own "
            "cycle. Read it as an upper bound on lambda_1, i.e. a LOWER bound "
            "on predictability. The project's own cross-check puts these rows "
            "at ~93% mean error where a published truth exists.")

    rho = meta.get("lag1_autocorr")
    if (rho is not None and np.isfinite(rho) and rho > 0.9
            and est.method != "ensemble"):
        notes.append(
            f"Strongly autocorrelated (lag-1 rho = {rho:.2f}) and likely "
            "seasonal -- exactly where Rosenstein inflates. Treat H_max as a "
            "scale, not a measurement, and set working horizons from measured "
            "forecast-skill decay instead.")
        verdict = "unreliable (seasonal)"

    if not np.isfinite(est.lyap) or est.lyap <= 0:
        verdict = "no positive exponent"
        notes.append(
            "No positive lambda_1: no exponential error growth, so the dynamics "
            "impose no predictability horizon. Any forecast limit here comes "
            "from noise, not chaos.")
    return verdict, notes


def provenance_doc(key: str) -> dict | None:
    """Human documentation for a dataset: source, observable, and gotchas."""
    doc = _provenance.DOCS.get(key)
    if doc is None:
        return None
    return {"title": doc.title, "source": doc.source,
            "observable": doc.observable, "meaning": doc.meaning,
            "watch_out": list(doc.watch_out)}


@lru_cache(maxsize=32)
def _raw(key: str):
    """The canonical pass, cached once and shared by every consumer here.

    ``save=False`` keeps it from writing into the analysis folder's ``figures/``
    as a side effect of someone loading a web page.
    """
    return _analysis.analyse_one(load_series(key), save=False)


def _analyse_cached(key: str) -> dict:
    s = load_series(key)
    meta = describe(key)
    r = _raw(key)
    est = r["_est"]

    H_max = (est.lyap_time / s.dt
             if np.isfinite(est.lyap_time) and s.dt else float("inf"))
    verdict, notes = _quality(est, meta)

    return {
        "key": key, "dataset": meta,
        "lambda1": _jsonable(r["lyap"]),
        "lyapunov_time": _jsonable(r["lyap_time"]),
        "H_max_steps": _jsonable(H_max),
        # inf/NaN become null above; `positive` keeps the distinction between
        # "no chaos detected" and "estimate unavailable".
        "positive": bool(np.isfinite(r["lyap"]) and r["lyap"] > 0),
        "method": r["lyap_method"],
        "reliable": bool(est.reliable),
        "verdict": verdict, "notes": notes,
        "embedding": (None if r["embedding"] is None
                      else {"m": int(r["embedding"][0]),
                            "tau": int(r["embedding"][1])}),
        "truth": (None if r["lyap_truth"] is None else float(r["lyap_truth"])),
        "error": (None if r["lyap_error"] is None else float(r["lyap_error"])),
        "n_realizations": int(r["n_realizations"]),
        "span": _jsonable(r["span"]),
        "e_foldings_in_record": _jsonable(r["e_foldings_in_record"]),
        # Peaks come from analyse_one, so they carry its fmin guard and match
        # what `python analysis.py --dataset <key>` prints.
        "dominant_periods": [
            {"period": float(p["period"]), "amp": float(p["amp"]),
             "f": float(p["f"])} for p in r["dominant_periods"]],
        "provenance": provenance_doc(key),
    }


def analyse(key: str) -> dict:
    """Predictability analysis for one dataset. Cached -- the fit is ~1-2 s."""
    return _analyse_cached(key)


def figure_path(key: str):
    """Render (once) and return the canonical 4-panel analysis figure.

    Produced by ``DataBase_Analysis.visualizer.figure`` with the same per-dataset
    ``FMAX`` / ``MAX_UNITS`` limits ``analysis.py`` uses, so the page and the CLI
    emit the same picture. Cached under ``webapp/results/chaos_figures/``.
    """
    path = FIGURE_CACHE / f"{key}_analysis.png"
    if path.exists():
        return path

    s = load_series(key)
    r = _raw(key)
    spec = _fourier.spectrum(s)
    # Same fmin guard analyse_one applies, so the figure's marked peaks agree
    # with the numbers in the table beside it.
    fmin = 2.0 / (len(s.x) * s.dt)
    peaks = _fourier.dominant_periods(spec.f, spec.amp, k=5, fmin=fmin)
    _viz.figure(s, spec, r["_est"], peaks=peaks,
                fmax=_analysis.FMAX.get(key),
                max_units=_analysis.MAX_UNITS.get(key),
                outdir=FIGURE_CACHE)
    return path if path.exists() else None


def horizon_budget(key: str, horizon_steps: int) -> dict:
    """Express a forecast horizon in Lyapunov units.

    ``h / H_max`` is the honest way to compare horizons across series on
    different clocks: one unit is one e-folding of an initial error, whether the
    step is a month or a minute.
    """
    a = analyse(key)
    hmax = a["H_max_steps"]
    frac = (horizon_steps / hmax) if np.isfinite(hmax) and hmax > 0 else None
    return {"horizon_steps": int(horizon_steps), "H_max_steps": hmax,
            "fraction_of_lyapunov_time": frac, "verdict": a["verdict"],
            "reliable": a["reliable"]}


# ----------------------------------------------------------------------
# Cross-dataset comparison
# ----------------------------------------------------------------------
def compare_all(tier: str | None = None, progress=None) -> dict:
    """Run the cross-dataset comparison from ``DataBase_Analysis.compare``.

    Expensive -- it is a full Lyapunov pass over every dataset -- so callers
    should submit it as a background job rather than blocking a request.
    """
    import compare as _compare

    say = progress or (lambda frac, msg: None)
    say(0.05, "running the full pass over every dataset")
    rows = _compare.collect(tier=tier)

    say(0.75, "comparison figure")
    figs = {}
    try:
        figs["comparison"] = str(_compare.figure(rows, outdir=FIGURE_CACHE))
    except Exception as exc:                        # noqa: BLE001
        figs["comparison_error"] = str(exc)
    try:
        figs["spectra"] = str(
            _compare.spectra_figure(rows, tier="real", outdir=FIGURE_CACHE))
    except Exception as exc:                        # noqa: BLE001
        figs["spectra_error"] = str(exc)

    say(0.95, "packing")
    packed = []
    for r in rows:
        packed.append({
            k: _jsonable(v) for k, v in r.items()
            if k not in ("_est", "dominant_periods", "figure", "embedding")
        })
    say(1.0, "done")
    return {"tier": tier, "n": len(packed), "rows": packed, "figures": figs}
