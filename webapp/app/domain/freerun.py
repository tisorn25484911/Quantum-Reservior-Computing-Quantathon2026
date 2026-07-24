"""freerun.py -- the honest "predict the future" evaluation.

Every other forecasting view in this app re-anchors to real data as it goes:
the /forecast page predicts one step from real history and then *gets the real
next value* before predicting again. That answers "how good is the next step",
not "can it predict the future it has never seen".

This module runs the strict test instead -- a **free-running (closed-loop)**
forecast:

    train on the TRAIN span only  ->  start at the train/test boundary
    ->  recurse across the ENTIRE test span, feeding the model its OWN output
        back into the reservoir at every step, never seeing a test value
    ->  only then compare the single self-generated trajectory to the test set

It is deliberately unflattering. Because a ridge read-out fed back through a
damped reservoir is a contraction, a long free-run tends to decay toward a
fixed point, so the honest competitors are put on the same axis:

* **persistence** -- repeat the last training value (a flat line; the true
  floor when no new data arrives),
* **seasonal average (climatology)** -- replay the per-phase training mean; on
  a seasonal series this is a strong baseline the model has to beat to matter,
* **size-matched ESN** -- the same free-run with the classical reservoir.

Nothing here claims the model wins. It reports what the free-run actually does
against those baselines, and leads with the unflattering reading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .catalog import load_series
from ..config import SEED

# forecast.py wires Hamiltonian_QRC onto sys.path; its qrc_core is the same
# module object the rest of the app imports (Python caches by name).
from forecast import (climatology_forecast, fit_readout,  # noqa: E402
                      make_reservoir, nmse, persistence_forecast, rollout)


@dataclass
class FreeRunConfig:
    """One free-running forecast study."""

    dataset: str = "got_sst"
    kind: str = "ising"              # "ising" | "xxz_hx" | "esn"
    n_qubits: int = 5
    dt: float = 2.0
    virtual_nodes: int = 10          # paper regime (STM saturates ~V=10)
    seed: int = SEED
    washout: int = 100
    train_frac: float = 0.5          # the rest is free-run BLIND
    max_points: int = 2000
    # Closed-loop robustness (Fujii & Nakajima 2017, MG task): fit the read-out
    # on noise-perturbed features so the self-fed rollout corrects back toward
    # the trajectory instead of collapsing to a fixed point. Available as a knob,
    # but OFF by default: on the got_sst blind free-run it did not help and added
    # seed variance (verified 5-seed), so shipping it on would be an overclaim.
    train_noise: float = 0.0

    def validate(self) -> None:
        if self.kind not in ("ising", "xxz_hx", "esn"):
            raise ValueError(f"unknown reservoir kind {self.kind!r}")
        if not 1 <= self.n_qubits <= 8:
            raise ValueError("n_qubits must be between 1 and 8 (cost ~4^n)")
        if not 1 <= self.virtual_nodes <= 50:
            raise ValueError("virtual_nodes must be between 1 and 50")
        if not 0.1 <= self.train_frac <= 0.9:
            raise ValueError("train_frac must lie in [0.1, 0.9]")
        if not 0.0 <= self.train_noise <= 1.0:
            raise ValueError("train_noise must lie in [0.0, 1.0]")
        if self.max_points < 300:
            raise ValueError("max_points must be at least 300")


def _seasonal_period(dt: float, time_unit: str) -> int | None:
    """Samples per natural cycle, or None if there is no obvious seasonality.

    Used only to decide whether a climatology baseline is meaningful. On the
    anomaly series (nino34) the cycle has already been removed upstream, so its
    climatology degenerates to the mean -- which is exactly what we want the
    plot to reveal, so it is still computed where a period exists.
    """
    tu = (time_unit or "").lower()
    if tu.startswith("day") and abs(dt - 1.0) < 1e-9:
        return 365
    if tu.startswith("yr") and abs(dt - 1.0 / 12.0) < 1e-6:
        return 12
    if tu.startswith("h") and abs(dt - 1.0) < 1e-9:
        return 24
    return None


def _free_run(x, kind, cfg, origin, H):
    """Train read-out on TRAIN only, then free-run H steps from `origin`.

    Returns the H-step self-fed trajectory (original units) and the clip count.
    The reservoir is primed by replaying only x[:origin+1] (train history);
    from step 1 on it consumes its own output.
    """
    res = make_reservoir(kind, n_qubits=cfg.n_qubits, dt=cfg.dt,
                         virtual_nodes=cfg.virtual_nodes, seed=cfg.seed)
    ro = fit_readout(x, res, washout=cfg.washout, train_frac=cfg.train_frac,
                     train_noise=cfg.train_noise,
                     rng=np.random.default_rng(cfg.seed))
    ex = rollout(x, res, ro, origin, H)
    return ex.yhat, int(ex.n_clipped), ro


def run_freerun(cfg: FreeRunConfig, progress=None) -> dict:
    """Execute one free-running forecast study; JSON-serialisable result."""
    cfg.validate()
    say = progress or (lambda frac, msg: None)

    say(0.03, "loading series")
    s = load_series(cfg.dataset)
    x = np.asarray(s.x, float)
    finite = np.isfinite(x)
    if not finite.all():                        # keep the axis uniform
        idx = np.arange(len(x))
        x = np.interp(idx, idx[finite], x[finite])
    if len(x) > cfg.max_points:
        x = x[-cfg.max_points:]

    # The split boundary comes from the same fit_readout the model uses, so the
    # "train only" promise is structural: origin is the last training index and
    # the free-run spans every test index after it.
    say(0.12, f"training {cfg.kind} read-out on train span only")
    # Build the readout once to discover the split, then free-run from it.
    res0 = make_reservoir(cfg.kind, n_qubits=cfg.n_qubits, dt=cfg.dt,
                          virtual_nodes=cfg.virtual_nodes, seed=cfg.seed)
    ro = fit_readout(x, res0, washout=cfg.washout, train_frac=cfg.train_frac,
                     train_noise=cfg.train_noise,
                     rng=np.random.default_rng(cfg.seed))
    origin = int(ro.train_idx[-1])
    H = len(ro.test_idx)
    if H < 10:
        raise ValueError(
            f"free-run span is only {H} steps; lower train_frac or the washout")

    say(0.2, f"free-running {cfg.kind} across {H} blind test steps")
    ex = rollout(x, res0, ro, origin, H)
    truth = x[origin + 1: origin + 1 + H]
    m = int(min(len(ex.yhat), len(truth)))
    truth = truth[:m]
    qrc = ex.yhat[:m]
    qrc_clip = int(ex.n_clipped)

    # ---- baselines on the identical blind span ---------------------------
    say(0.55, "persistence + seasonal-average baselines")
    persist = persistence_forecast(x, origin, m)
    period = _seasonal_period(float(s.dt), s.time_unit)
    clim = (climatology_forecast(x, origin, m, period, origin + 1)
            if period else None)

    say(0.7, "size-matched ESN free-run (classical control)")
    esn_yhat, esn_clip, _ = _free_run(x, "esn", cfg, origin, m)
    esn = esn_yhat[:m]

    # ---- scores ----------------------------------------------------------
    say(0.9, "scoring against test")
    def score(pred):
        return None if pred is None else float(nmse(truth, np.asarray(pred)[:m]))

    scores = {"qrc": score(qrc), "esn": score(esn),
              "persistence": score(persist),
              "climatology": (score(clim) if clim is not None else None)}
    verdict = _verdict(scores, cfg, period)

    say(1.0, "done")
    ctx0 = max(0, origin - 3 * m)
    return {
        "config": asdict(cfg),
        "dataset": _meta(s, len(x)),
        "period": period,
        "origin": origin,
        "n_free": m,
        "free_years": float(m * s.dt),
        "clipped": {"qrc": qrc_clip, "esn": int(esn_clip)},
        "scores": scores,
        "verdict": verdict,
        "series": {
            "observed_full": x.tolist(),
            "context_start": int(ctx0),
            "truth": truth.tolist(),
            "qrc": np.asarray(qrc)[:m].tolist(),
            "esn": np.asarray(esn)[:m].tolist(),
            "persistence": np.asarray(persist)[:m].tolist(),
            "climatology": (np.asarray(clim)[:m].tolist()
                            if clim is not None else None),
        },
        "zones": {"washout": int(cfg.washout), "train_end": origin + 1,
                  "test_end": origin + 1 + m, "T": int(len(x))},
    }


def _meta(s, n: int) -> dict:
    from ..config import TIER_LABELS
    return {"key": s.key, "name": s.name, "tier": s.tier,
            "tier_label": TIER_LABELS.get(s.tier, s.tier),
            "dt": float(s.dt), "time_unit": s.time_unit or "sample",
            "unit": s.unit or "", "n": int(n)}


def _verdict(scores: dict, cfg: FreeRunConfig, period: int | None) -> dict:
    """Plain-language reading, written so the unflattering case is the default."""
    q = scores["qrc"]
    notes: list[str] = []

    beat_mean = q is not None and q < 1.0
    beat_persist = q is not None and scores["persistence"] is not None \
        and q < scores["persistence"]
    clim = scores["climatology"]
    beat_clim = (clim is None) or (q is not None and q < clim)

    if not beat_mean:
        headline = ("Free-run carries no skill: over the blind test span it is "
                    "no better than predicting the mean")
        notes.append(
            "A long closed-loop rollout of a mean read-out contracts toward a "
            "fixed point, so it flattens out and stops tracking. This is the "
            "expected behaviour, not a bug -- and it is why 'predict the "
            "distant future' is not a claim this method supports.")
    elif clim is not None and not beat_clim:
        headline = ("Free-run has some skill, but the trivial seasonal-average "
                    "baseline beats it")
        notes.append(
            f"Seasonal average NMSE {clim:.2f} vs free-run {q:.2f}: on a "
            "seasonal series, just replaying the per-phase mean is the better "
            "forecast. The model has to beat this line to be worth anything "
            "here, and on this run it does not.")
    elif not beat_persist:
        headline = "Free-run barely beats repeating the last observed value"
    else:
        headline = "Free-run beats the naive baselines over the whole blind span"

    e = scores["esn"]
    if cfg.kind != "esn" and q is not None and e is not None:
        ratio = q / e if e > 0 else float("inf")
        if ratio <= 0.97:
            notes.append(f"vs size-matched ESN free-run: NMSE ratio {ratio:.2f} "
                         "(QRC lower). At one seed/one split this is parity, "
                         "not a demonstrated advantage.")
        elif ratio >= 1.03:
            notes.append(f"vs size-matched ESN free-run: NMSE ratio {ratio:.2f} "
                         "-- the classical control free-runs at least as well. "
                         "Reported, not hidden.")
        else:
            notes.append(f"vs size-matched ESN free-run: within 3% "
                         f"(ratio {ratio:.2f}) -- parity.")
    if period is None:
        notes.append("No seasonal-average baseline: this series has no obvious "
                     "cycle at its sampling rate, so only persistence and the "
                     "mean-predictor line apply.")

    return {"headline": headline,
            "beat_mean": bool(beat_mean),
            "beat_persistence": bool(beat_persist),
            "beat_climatology": bool(beat_clim if clim is not None else False),
            "has_climatology": clim is not None,
            "notes": notes}
