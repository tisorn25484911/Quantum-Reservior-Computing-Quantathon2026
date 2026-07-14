"""conformal.py -- adaptive conformal intervals + rolling coverage
monitor (Phase 9; L6 hook).

Split conformal: calibration residuals -> the (1-alpha) empirical
quantile widens the head's interval; finite-sample, distribution-free.
Adaptive conformal (ACI, Gibbs & Candes 2021 convention): alpha_t is
adjusted online, alpha_{t+1} = alpha_t + lr * (alpha - miss_t), so the
effective miscoverage TRACKS the target under regime shift -- the
product's contractual guarantee ("90% intervals that cover 90%,
monitored weekly") rather than an accuracy claim.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SplitConformal:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def calibrate(self, y_cal: np.ndarray, lo_cal: np.ndarray,
                  hi_cal: np.ndarray) -> "SplitConformal":
        scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
        n = len(scores)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.q = float(np.sort(scores)[min(k, n) - 1])
        return self

    def interval(self, lo: np.ndarray, hi: np.ndarray):
        return lo - self.q, hi + self.q


class AdaptiveConformal:
    """Online alpha_t update on top of the split quantile."""

    def __init__(self, alpha: float = 0.1, lr: float = 0.02):
        self.alpha_target = alpha
        self.alpha_t = alpha
        self.lr = lr
        self.scores: deque = deque(maxlen=500)

    def start(self, y_cal, lo_cal, hi_cal) -> "AdaptiveConformal":
        for y, lo, hi in zip(y_cal, lo_cal, hi_cal):
            self.scores.append(max(lo - y, y - hi))
        return self

    def _q(self) -> float:
        s = np.sort(np.asarray(self.scores))
        n = len(s)
        a = float(np.clip(self.alpha_t, 1e-3, 0.999))
        k = int(np.ceil((n + 1) * (1 - a)))
        return float(s[min(max(k, 1), n) - 1])

    def interval(self, lo: float, hi: float):
        q = self._q()
        return lo - q, hi + q

    def update(self, y: float, lo: float, hi: float) -> bool:
        """Observe the outcome; returns covered?"""
        l, h = self.interval(lo, hi)
        covered = l <= y <= h
        self.alpha_t += self.lr * (self.alpha_target - (0.0 if covered
                                                        else 1.0))
        self.alpha_t = float(np.clip(self.alpha_t, 1e-3, 0.5))
        self.scores.append(max(lo - y, y - hi))
        return covered


class CoverageMonitor:
    """Rolling empirical coverage; Phase-9 acceptance = within 3 points
    of target on held-out streams; alarms feed L6."""

    def __init__(self, target: float, window: int = 200,
                 tolerance: float = 0.03):
        self.target, self.tolerance = target, tolerance
        self.buf: deque = deque(maxlen=window)

    def observe(self, covered: bool) -> None:
        self.buf.append(bool(covered))

    def coverage(self) -> float:
        return float(np.mean(self.buf)) if self.buf else float("nan")

    def healthy(self) -> bool:
        return (len(self.buf) < 30
                or abs(self.coverage() - self.target) <= self.tolerance)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 4000
    y = rng.normal(size=T)
    lo, hi = np.full(T, -0.5), np.full(T, 0.5)   # deliberately too narrow
    ac = AdaptiveConformal(alpha=0.1).start(y[:500], lo[:500], hi[:500])
    mon = CoverageMonitor(target=0.9)
    for t in range(500, T):
        mon.observe(ac.update(y[t], lo[t], hi[t]))
    print(f"adaptive conformal: rolling coverage {mon.coverage():.3f} "
          f"(target 0.9) healthy={mon.healthy()}")
    assert mon.healthy()
    print("conformal self-test PASS")
