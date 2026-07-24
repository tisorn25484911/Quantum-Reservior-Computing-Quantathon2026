"""failure_modes.py -- failure taxonomy registry (blow-up, mean-collapse, saturation, ...).

Classifies an autonomous trajectory against a training-statistics reference using
the PREREGISTERED, dev-tunable-only thresholds in
``configs/preregistration.yaml::failure_rules`` (spec s16.3). Multiple labels may
apply; the first-failure step is recorded per label. A trajectory that trips no
rule is "stable".

Modes: blowup, fixed_point, mean_collapse, variance_collapse, variance_explosion,
spurious_cycle, saturation, stochastic_instability. Implemented in P4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FailureRules:
    blowup_range_mult: float = 1.5
    fixed_point_std: float = 0.05
    mean_collapse: float = 0.10
    var_collapse_mult: float = 0.25
    var_explosion_mult: float = 4.0
    spurious_cycle_power: float = 0.60
    window: int = 24

    @classmethod
    def from_config(cls, cfg):
        fr = cfg.get("failure_rules", cfg)
        known = {k: fr[k] for k in cls.__dataclass_fields__ if k in fr}
        return cls(**known)


@dataclass(frozen=True)
class TrainStats:
    mu: float
    sigma: float
    lo: float
    hi: float

    @classmethod
    def from_series(cls, y_train):
        y = np.asarray(y_train, dtype=float)
        return cls(float(y.mean()), float(y.std()), float(y.min()), float(y.max()))

    @property
    def train_range(self):
        return self.hi - self.lo

    @property
    def var(self):
        return self.sigma ** 2


def _first_true(mask):
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    return int(idx[0]) if idx.size else None


def _rolling(x, w, fn):
    """Rolling statistic over windows of length w; array aligned to window END."""
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return np.array([]), np.array([], dtype=int)
    out = np.array([fn(x[i - w + 1:i + 1]) for i in range(w - 1, len(x))])
    ends = np.arange(w - 1, len(x))
    return out, ends


def classify(traj, stats, rules, clip_fraction=0.0, boundary_dwell_fraction=0.0,
             clip_budget=0.20, boundary_budget=0.30, ensemble=None):
    """Return ``{label: first_failure_step}`` for every triggered mode.

    ``ensemble`` (optional list of same-length trajectories from different seeds)
    enables the stochastic-instability check: seed-to-seed divergence while the
    deterministic core stays bounded.
    """
    traj = np.asarray(traj, dtype=float)
    w = rules.window
    labels = {}

    # blow-up: |x| beyond a multiple of the train range from the mean
    blow = np.abs(traj - stats.mu) > rules.blowup_range_mult * stats.train_range
    if blow.any():
        labels["blowup"] = _first_true(blow)

    rstd, ends = _rolling(traj, w, np.std)
    if rstd.size:
        # fixed point: rolling std collapses
        fp = rstd < rules.fixed_point_std * stats.sigma
        if fp.any():
            labels["fixed_point"] = int(ends[_first_true(fp)])
        # variance collapse / explosion
        rvar = rstd ** 2
        vc = rvar < rules.var_collapse_mult * stats.var
        if vc.any():
            labels["variance_collapse"] = int(ends[_first_true(vc)])
        ve = rvar > rules.var_explosion_mult * stats.var
        if ve.any():
            labels["variance_explosion"] = int(ends[_first_true(ve)])

    rmean, ends_m = _rolling(traj, w, np.mean)
    if rmean.size:
        mc = np.abs(rmean - stats.mu) < rules.mean_collapse * stats.sigma
        # only a collapse if variance is also low (a flat line near the mean)
        rstd_m, _ = _rolling(traj, w, np.std)
        mc = mc & (rstd_m < rules.fixed_point_std * stats.sigma)
        if mc.any():
            labels["mean_collapse"] = int(ends_m[_first_true(mc)])

    # spurious limit cycle: one spectral peak dominates the de-meaned trajectory
    if len(traj) >= 2 * w:
        d = traj - traj.mean()
        ps = np.abs(np.fft.rfft(d)) ** 2
        ps[0] = 0.0
        if ps.sum() > 0 and ps.max() / ps.sum() > rules.spurious_cycle_power:
            labels["spurious_cycle"] = 0

    # saturation: too much clipping or boundary dwell
    if clip_fraction > clip_budget or boundary_dwell_fraction > boundary_budget:
        labels["saturation"] = 0

    # stochastic instability: seeds diverge though the mean stays bounded
    if ensemble is not None and len(ensemble) >= 2:
        arr = np.asarray(ensemble, dtype=float)
        spread = arr.std(axis=0)
        core_bounded = np.abs(arr.mean(axis=0) - stats.mu).max() < \
            rules.blowup_range_mult * stats.train_range
        div = spread > stats.sigma
        if core_bounded and div.any():
            labels["stochastic_instability"] = _first_true(div)

    return labels
