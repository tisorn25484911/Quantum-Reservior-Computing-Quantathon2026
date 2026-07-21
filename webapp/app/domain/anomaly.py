"""anomaly.py -- the Forecast-then-Detect branch, as far as it is actually built.

Wraps ``Anomaly_Forecast/forecast.py`` (plan.md Steps 1-2) for the web layer.
The premise of that plan is to run anomaly detectors on *forecast* trajectories
rather than on observed data, so an anomaly is flagged before it happens:

    observed -> recursive QRC rollout -> treat yhat as data -> detectors
             -> P(anomaly at t+h) with lead time up to H

**Only the forecast half exists.** plan.md Steps 3-8 -- stochastic ensembles,
the three scorers, EVT thresholds, event debouncing, injected ground truth --
are not implemented, so this module exposes the multi-step rollout and its
skill decay and nothing more. It deliberately does not present a detection
probability, because a probability computed from a machinery that has not been
calibrated would be the single most misleading number the app could show.

What it *does* answer is the question that gates everything downstream: how far
ahead the rollout keeps any skill at all, and whether it beats the floors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import metrics
from .catalog import series_values
from .chaos import analyse as chaos_analyse
from ..config import SEED

# `forecast` puts Hamiltonian_QRC on sys.path itself; its `qrc_core` is
# byte-identical to the stage-1 reference the rest of the app imports, and
# Python caches by module name, so exactly one `qrc_core` is ever loaded.
from forecast import (fit_readout, make_reservoir,  # noqa: E402
                      rollout, skill_vs_horizon)

# plan.md §7: the reservoir must beat these, not merely have low error.
MATERIAL = 0.05          # relative NMSE margin resolvable at one seed


@dataclass
class AnomalyConfig:
    """Configuration for one Forecast-then-Detect rollout study."""

    dataset: str = "nino34"
    horizon: int = 12                # H, steps ahead
    kind: str = "xxz_hx"             # "ising" | "xxz_hx" | "esn"
    n_qubits: int = 5
    dt: float = 2.0
    virtual_nodes: int = 4
    seed: int = SEED
    washout: int = 100
    train_frac: float = 0.7
    max_points: int = 2000

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1 step")
        if self.kind not in ("ising", "xxz_hx", "esn"):
            raise ValueError(f"unknown reservoir kind {self.kind!r}")
        if not 1 <= self.n_qubits <= 8:
            raise ValueError("n_qubits must be between 1 and 8 (cost ~4^n)")
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError("train_frac must lie in (0, 1)")


def _wins(curve, ref, margin: float = 0.0) -> list[int]:
    """Horizons where ``curve`` beats ``ref`` by ``margin`` (1-indexed).

    A set, not a crossing point: these curves are not monotone in their
    difference, so "the horizon where it stops winning" is not well defined
    and quoting one discards half the picture.
    """
    return [h for h, (a, b) in enumerate(zip(curve, ref), start=1)
            if a < b * (1.0 - margin)]


def run_rollout_study(cfg: AnomalyConfig, progress=None) -> dict:
    """Recursive-rollout skill decay for one series, against both floors.

    Runs the configured reservoir *and* a size-matched ESN control over the
    identical held-out origins, then reports the horizons at which the
    reservoir has real skill (NMSE < 1) and materially beats each floor.
    """
    cfg.validate()
    say = progress or (lambda frac, msg: None)

    say(0.03, "loading series")
    x, meta = series_values(cfg.dataset, max_points=cfg.max_points)
    H = cfg.horizon

    say(0.10, f"driving {cfg.kind} reservoir")
    res = make_reservoir(cfg.kind, n_qubits=cfg.n_qubits, dt=cfg.dt,
                         virtual_nodes=cfg.virtual_nodes, seed=cfg.seed)
    ro = fit_readout(x, res, washout=cfg.washout, train_frac=cfg.train_frac)
    out = skill_vs_horizon(x, res, ro, H)

    say(0.55, "size-matched ESN control")
    esn = make_reservoir("esn", n_qubits=cfg.n_qubits,
                         virtual_nodes=cfg.virtual_nodes, seed=cfg.seed)
    ro_e = fit_readout(x, esn, washout=cfg.washout, train_frac=cfg.train_frac)
    out_e = skill_vs_horizon(x, esn, ro_e, H)

    say(0.8, "scoring")
    q = out["nmse"]["reservoir"]
    p = out["nmse"]["persistence"]
    e = out_e["nmse"]["reservoir"]
    ones = [1.0] * H

    has_skill = _wins(q, ones)
    beats_p = _wins(q, p, MATERIAL)
    beats_e = _wins(q, e, MATERIAL)
    esn_beats = _wins(e, q, MATERIAL)
    useful = sorted(set(has_skill) & set(beats_p))

    # An example trajectory makes the compounding visible in a way the
    # aggregate curve cannot: one origin, rolled out H steps, against truth.
    say(0.88, "example trajectory")
    origin = int(ro.test_idx[len(ro.test_idx) // 2])
    ex = rollout(x, res, ro, origin, H)

    # Where the horizon sits relative to the physical ceiling.
    try:
        budget = chaos_analyse(cfg.dataset)
        H_max = budget["H_max_steps"]
        chaos_verdict = budget["verdict"]
    except Exception:
        H_max, chaos_verdict = float("inf"), "unavailable"

    say(1.0, "done")
    return {
        "config": asdict(cfg), "dataset": meta,
        "horizons": out["horizons"], "n_origins": out["n_origins"],
        "nmse": {"reservoir": q, "persistence": p, "esn": e},
        "clip_rate": out["clip_rate"],
        "skill_at": has_skill, "beats_persistence_at": beats_p,
        "beats_esn_at": beats_e, "esn_beats_at": esn_beats,
        "useful_horizons": useful,
        "useful_lead": (max(useful) if useful else 0),
        "material_margin": MATERIAL,
        "H_max_steps": H_max, "chaos_verdict": chaos_verdict,
        "example": {"origin": origin,
                    "truth": x[origin + 1: origin + H + 1].tolist(),
                    "yhat": ex.yhat.tolist(),
                    "history": x[max(0, origin - 3 * H): origin + 1].tolist(),
                    "n_clipped": ex.n_clipped},
        "verdict": _verdict(useful, beats_e, esn_beats, has_skill, H,
                            meta, cfg),
        "not_implemented": [
            "Step 3 stochastic ensemble (K trajectories) -- without it the "
            "mean rollout is smooth and systematically UNDER-alarms, because "
            "ridge minimises MSE and returns E[y|past].",
            "Steps 4-5 scorers, fusion and EVT thresholds.",
            "Step 6 injected ground truth; Step 7 hit-rate vs lead time.",
        ],
    }


def _verdict(useful, beats_e, esn_beats, has_skill, H, meta, cfg) -> dict:
    """Plain-language reading, written so the unflattering case is the default."""
    notes: list[str] = []
    if not has_skill:
        headline = "No skill at any lead: the rollout never beats the mean predictor"
        notes.append("Every other number here is void.")
    elif not useful:
        headline = "Has skill, but never materially beats persistence"
        notes.append(
            "On a strongly autocorrelated series, repeating the last "
            "observation is already the better forecast; the model adds "
            "nothing operationally.")
    else:
        headline = (f"Usable lead: {max(useful)} step(s) "
                    f"({max(useful) * meta['dt']:.3g} {meta['time_unit']})")

    if cfg.kind != "esn":
        if len(beats_e) > 2 * max(len(esn_beats), 1):
            notes.append(
                f"Beats the size-matched ESN at {len(beats_e)}/{H} leads "
                f"(ESN better at {len(esn_beats)}/{H}). At ONE seed this is a "
                "provisional signal, not an established result.")
        elif len(esn_beats) > 2 * max(len(beats_e), 1):
            notes.append(
                f"The size-matched ESN is better at {len(esn_beats)}/{H} "
                f"leads. Reported, not hidden.")
        else:
            notes.append(
                "Parity with the size-matched ESN: no quantum advantage is "
                "demonstrated, which is the expected outcome at these sizes.")
        if cfg.kind == "xxz_hx":
            notes.append(
                "Note xxz_hx is a clean chain with NO disorder, so it is "
                "deterministic in the seed while the ESN is not -- a "
                "single-seed comparison pits a fixed point against one draw "
                "of a random variable. Vary the seed before concluding.")
    return {"headline": headline, "notes": notes,
            "has_skill": bool(has_skill), "usable": bool(useful)}
