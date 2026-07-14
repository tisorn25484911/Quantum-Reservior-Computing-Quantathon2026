"""monitors.py -- L6 governance monitors (Phase 9/12 hooks).

Input drift (rolling two-sample KS statistic against a frozen reference
window), rolling conformal coverage (imported from serving.conformal's
CoverageMonitor), feature EFFECTIVE RANK as the QRC-specific health
metric -- rank collapse is the leading indicator that quantum features
have degenerated (Part XI; measured in this repo's own stage-6 pilot at
4.4/1024) -- and alert-fatigue counters (in decision.alerts).
"""

from __future__ import annotations

from collections import deque

import numpy as np


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.concatenate([a, b])
    Fa = np.searchsorted(a, grid, side="right") / len(a)
    Fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(Fa - Fb)))


class DriftMonitor:
    """KS drift of each input channel vs its frozen train-span sample."""

    def __init__(self, reference: np.ndarray, window: int = 500,
                 threshold: float = 0.25):
        self.ref = np.asarray(reference, dtype=float)
        self.ref = self.ref[np.isfinite(self.ref)]
        self.buf: deque = deque(maxlen=window)
        self.threshold = threshold

    def observe(self, x: float) -> None:
        if np.isfinite(x):
            self.buf.append(float(x))

    def statistic(self) -> float:
        if len(self.buf) < 50:
            return 0.0
        return ks_statistic(self.ref, np.array(self.buf))

    def healthy(self) -> bool:
        return self.statistic() < self.threshold


class EffectiveRankMonitor:
    """Participation ratio of a rolling feature buffer; alarms on a
    RELATIVE collapse vs the value frozen at promotion time."""

    def __init__(self, rank_at_promotion: float, window: int = 300,
                 collapse_ratio: float = 0.5):
        self.r0 = rank_at_promotion
        self.buf: deque = deque(maxlen=window)
        self.collapse_ratio = collapse_ratio

    def observe(self, feature_row: np.ndarray) -> None:
        self.buf.append(np.asarray(feature_row, dtype=float))

    def rank(self) -> float:
        if len(self.buf) < 30:
            return float("nan")
        F = np.array(self.buf)
        cov = np.atleast_2d(np.cov(F.T))
        eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
        s = eig.sum()
        return float(s ** 2 / np.sum(eig ** 2)) if s > 0 else 0.0

    def healthy(self) -> bool:
        r = self.rank()
        return np.isnan(r) or r >= self.collapse_ratio * self.r0


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = rng.normal(size=2000)
    dm = DriftMonitor(ref)
    for x in rng.normal(size=400):
        dm.observe(x)
    ok1 = dm.healthy()
    for x in rng.normal(loc=1.5, size=400):
        dm.observe(x)
    ok2 = not dm.healthy()
    print(f"drift: in-dist healthy {ok1}, shifted flagged {ok2}")

    em = EffectiveRankMonitor(rank_at_promotion=8.0)
    for _ in range(200):
        em.observe(rng.normal(size=(10,)))
    ok3 = em.healthy()                      # iid: rank ~ 10 >= 4
    em2 = EffectiveRankMonitor(rank_at_promotion=8.0)
    v = rng.normal(size=10)
    for _ in range(200):
        em2.observe(v * rng.normal() + 0.01 * rng.normal(size=10))
    ok4 = not em2.healthy()                 # rank-1 collapse flagged
    print(f"effective rank: healthy {ok3} ({em.rank():.1f}), collapse "
          f"flagged {ok4} ({em2.rank():.1f})")
    assert ok1 and ok2 and ok3 and ok4
    print("monitors self-test PASS")
