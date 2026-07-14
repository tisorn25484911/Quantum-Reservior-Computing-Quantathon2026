"""qc.py -- L1 quality control: the mask policy as code (Phase 8).

Policy (playbook rule 3, non-negotiable): missing or flagged data is
MASKED, never silently interpolated; gaps longer than the series'
autocorrelation time SPLIT the series into independent segments (models
must not learn across a gap they cannot see through). Every QC pass
emits a gap report -- the report is a first-class artifact, stored next
to the data version.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QCReport:
    n_total: int
    n_masked: int
    n_gaps: int
    longest_gap: int
    split_points: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"QC: {self.n_total} steps, {self.n_masked} masked "
                f"({100 * self.n_masked / max(self.n_total, 1):.2f}%), "
                f"{self.n_gaps} gaps (longest {self.longest_gap}), "
                f"{len(self.split_points)} splits")


def autocorr_time(x: np.ndarray, max_lag: int = 200) -> int:
    """First lag where the autocorrelation of the (masked-removed) series
    drops below 1/e -- the gap-splitting threshold."""
    x = x[np.isfinite(x)]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 1
    for lag in range(1, min(max_lag, len(x) // 2)):
        r = float(np.dot(x[:-lag], x[lag:])) / denom
        if r < 1.0 / np.e:
            return lag
    return max_lag


def apply_qc(x: np.ndarray, bad_mask: np.ndarray | None = None,
             split_over: int | None = None):
    """Mask -> gap report -> split.

    Returns (segments, report): a list of contiguous np arrays with NaN
    where masked (short gaps stay as NaN inside a segment; long gaps
    split), plus the QCReport. `bad_mask` marks known-bad samples (QC
    flags, negative irradiance at night, sensor dropouts); non-finite
    values are always masked too.
    """
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad_mask is not None:
        bad |= np.asarray(bad_mask, dtype=bool)
    x[bad] = np.nan

    if split_over is None:
        split_over = autocorr_time(x)

    # find gap runs
    gaps, in_gap, start = [], False, 0
    for i, b in enumerate(bad):
        if b and not in_gap:
            in_gap, start = True, i
        elif not b and in_gap:
            gaps.append((start, i))
            in_gap = False
    if in_gap:
        gaps.append((start, len(x)))

    splits = [g for g in gaps if g[1] - g[0] > split_over]
    segments, prev = [], 0
    for lo, hi in splits:
        if lo - prev > 0:
            segments.append(x[prev:lo])
        prev = hi
    if len(x) - prev > 0:
        segments.append(x[prev:])
    segments = [s for s in segments if np.isfinite(s).sum() > 0]

    report = QCReport(
        n_total=len(x), n_masked=int(bad.sum()), n_gaps=len(gaps),
        longest_gap=max((g[1] - g[0] for g in gaps), default=0),
        split_points=[g[0] for g in splits],
        notes=[f"split threshold = {split_over} steps (autocorr time)",
               "policy: mask, never interpolate"])
    return segments, report


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.sin(np.arange(500) / 10.0) + 0.1 * rng.normal(size=500)
    x[100:103] = np.nan          # short gap: stays inside a segment
    x[300:380] = np.nan          # long gap: splits
    segs, rep = apply_qc(x)
    print(rep.summary())
    print(f"segments: {[len(s) for s in segs]}")
    assert len(segs) == 2 and rep.n_masked == 83
    print("qc self-test PASS")
