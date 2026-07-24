"""autonomous.py -- rollout(model, history, n_steps, domain_policy) -- no-future guard (G5).

Closed-loop autonomous forecasting. After an origin the model consumes only its
own predictions (fed back through an optional domain policy). The harness owns the
full series and passes the rollout ONLY the history slice (G5): this function never
sees, and cannot read, anything at or beyond the origin. ``autonomous_from_series``
makes that structural by copying ``series[:origin]`` before the loop.

Model protocol (spec s16.1):
    warm(history) -> state
    step(state, u) -> (state, yhat)

Domain policies (spec s16.3), all telemetered, none silent: hard_clip,
smooth_bounded (logistic into [lo,hi]), terminate (stop on violation),
wide_encoding (remap into a wider band). Implemented in P4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DomainPolicy:
    """Bounds the fed-back value and records telemetry (spec s16.3/s16.4)."""

    kind: str = "hard_clip"
    lo: float = 0.0
    hi: float = 1.0
    boundary_tol: float = 0.01           # within 1% of a band edge counts as dwell
    # telemetry
    n_seen: int = 0
    n_clipped: int = 0
    first_clip_t: int | None = None
    pre_clip: list = field(default_factory=list)
    max_violation: float = 0.0
    boundary_dwell: int = 0
    terminated_at: int | None = None

    def apply(self, u, t):
        """Return (u_out, terminated). Records clip/dwell/violation telemetry."""
        self.n_seen += 1
        self.pre_clip.append(float(u))
        span = self.hi - self.lo
        violated = u < self.lo or u > self.hi
        if violated:
            self.n_clipped += 1
            if self.first_clip_t is None:
                self.first_clip_t = t
            self.max_violation = max(self.max_violation,
                                     max(self.lo - u, u - self.hi))
        if self.kind == "terminate" and violated:
            self.terminated_at = t
            return float(np.clip(u, self.lo, self.hi)), True
        if self.kind == "smooth_bounded":
            # logistic squashing of the *excursion*, keeps interior near-identity
            z = (u - self.lo) / span if span else 0.0
            u_out = self.lo + span / (1.0 + np.exp(-4.0 * (z - 0.5)))
        else:                              # hard_clip / wide_encoding / terminate
            u_out = float(np.clip(u, self.lo, self.hi))
        if abs(u_out - self.lo) <= self.boundary_tol * span or \
           abs(u_out - self.hi) <= self.boundary_tol * span:
            self.boundary_dwell += 1
        return float(u_out), False

    def telemetry(self):
        n = max(self.n_seen, 1)
        return {
            "kind": self.kind, "n": self.n_seen, "n_clipped": self.n_clipped,
            "clip_fraction": self.n_clipped / n,
            "first_clip_t": self.first_clip_t,
            "max_violation": self.max_violation,
            "boundary_dwell": self.boundary_dwell,
            "boundary_dwell_fraction": self.boundary_dwell / n,
            "terminated_at": self.terminated_at,
        }


def make_policy(kind, lo=0.0, hi=1.0, **kw):
    return DomainPolicy(kind=kind, lo=lo, hi=hi, **kw)


def rollout(model, history, n_steps, policy=None):
    """Autonomous closed-loop rollout from the end of ``history`` (G5-safe).

    Only ``history`` is read; there is no argument for the future. Returns a dict
    with ``predictions`` (len <= n_steps), ``terminated_at``, and ``telemetry``.
    """
    history = np.asarray(history, dtype=float)
    state = model.warm(history)
    u = float(history[-1])
    preds = []
    terminated_at = None
    for t in range(n_steps):
        state, yhat = model.step(state, u)
        preds.append(float(yhat))
        if policy is not None:
            u, terminated = policy.apply(yhat, t)
        else:
            u, terminated = float(yhat), False
        if terminated:
            terminated_at = t
            break
    return {
        "predictions": np.asarray(preds),
        "terminated_at": terminated_at,
        "telemetry": policy.telemetry() if policy is not None else {},
    }


def autonomous_from_series(model, series, origin, n_steps, policy=None):
    """G5 wrapper: copy the warmup slice so the future is physically inaccessible."""
    warmup = np.asarray(series, dtype=float)[:origin].copy()
    return rollout(model, warmup, n_steps, policy=policy)


def make_origins(n_total, warmup, n_steps, n_origins, spacing):
    """Evenly spaced rollout origins with room for warmup and horizon (spec s21).

    Returns up to ``n_origins`` origin indices o with ``warmup <= o`` and
    ``o + n_steps <= n_total``, separated by at least ``spacing`` (>= one
    decorrelation time where the record allows).
    """
    first, last = warmup, n_total - n_steps
    if last < first:
        return []
    cand = list(range(first, last + 1, max(spacing, 1)))
    if len(cand) <= n_origins:
        return cand
    pick = np.linspace(0, len(cand) - 1, n_origins).round().astype(int)
    return [cand[i] for i in sorted(set(pick))]


def multi_origin_horizons(model_factory, series, origins, warmup, n_steps,
                          sigma, eps=0.5, K=3, policy_factory=None,
                          baseline_factory=None):
    """Run an autonomous rollout at each origin; return per-origin horizon dicts.

    ``model_factory()`` / ``baseline_factory()`` build fresh models per origin (no
    state leaks across origins). Combines rollout + accumulated_error +
    prediction_horizon; the true future is used ONLY to score, never inside the
    rollout (G5 is enforced by ``autonomous_from_series``).
    """
    from .accumulated_error import instantaneous_ne, cnrmse
    from .prediction_horizon import h_error, h_skill, h_reliable, h_effective

    series = np.asarray(series, dtype=float)
    results = []
    for o in origins:
        pol = policy_factory() if policy_factory else None
        r = autonomous_from_series(model_factory(), series, o, n_steps, policy=pol)
        preds = r["predictions"]
        h = len(preds)
        truth = series[o:o + h]
        ne = instantaneous_ne(truth, preds, sigma)
        he = h_error(ne, eps, K)
        if baseline_factory is not None:
            b = autonomous_from_series(baseline_factory(), series, o, n_steps)
            hb = len(b["predictions"])
            m = min(h, hb)
            hs = h_skill(cnrmse(truth[:m], preds[:m], sigma),
                         cnrmse(series[o:o + m], b["predictions"][:m], sigma), K)
        else:
            hs = h
        out = h_effective(he, hs, h_reliable(None, h))
        out["origin"] = int(o)
        results.append(out)
    return results
