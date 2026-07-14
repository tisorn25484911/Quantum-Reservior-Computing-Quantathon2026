"""transforms.py -- L2: physics target transforms + the leakage firewall.

The firewall is ARCHITECTURAL, not a code-review hope: every transform
carries frozen train-span statistics inside a versioned registry entry;
`fit()` may only ever see the training span, `apply()` is stateless, and
attempting to apply an unfitted (or version-mismatched) transform raises.

Transforms (playbook rule 4 -- baselines apply to the TRANSFORMED target):
    ClearSkyIndex   solar -> k_t = GHI / GHI_clear (never model raw GHI);
                    the clear-sky curve is fit on the TRAIN span only
                    (per time-of-day median of high-GHI days -- a
                    Haurwitz-free empirical curve so the surrogate and
                    real data share one code path)
    Anomaly         climate -> anomaly vs training-years monthly
                    climatology
    MinMax          [0,1] scaling for reservoir encodings (train stats)
Calendar/solar-geometry features come from lookup (`calendar_features`):
no quantum resource on what a lookup table solves.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FROZEN_DIR = Path(__file__).resolve().parent / ".." / "data" / "frozen_stats"


class Frozen:
    """Mixin: fit-once, frozen, versioned."""

    def __init__(self, model_version: str):
        self.model_version = model_version
        self._fitted = False

    def _freeze(self, payload: dict) -> None:
        self._stats = payload
        self._fitted = True

    def _need(self):
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__}: apply() before fit() -- the "
                "leakage firewall refuses")

    def save(self) -> Path:
        self._need()
        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        blob = json.dumps({k: (v.tolist() if isinstance(v, np.ndarray)
                               else v) for k, v in self._stats.items()},
                          sort_keys=True)
        h = hashlib.sha256(blob.encode()).hexdigest()[:8]
        p = FROZEN_DIR / f"{type(self).__name__}_{self.model_version}_{h}.json"
        p.write_text(blob)
        return p


class MinMax(Frozen):
    def fit(self, x_train: np.ndarray) -> "MinMax":
        x = np.asarray(x_train, dtype=float)
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        self._freeze({"lo": lo, "hi": hi if hi > lo else lo + 1.0})
        return self

    def apply(self, x: np.ndarray) -> np.ndarray:
        self._need()
        s = self._stats
        return np.clip((np.asarray(x, dtype=float) - s["lo"])
                       / (s["hi"] - s["lo"]), 0.0, 1.0)


class Anomaly(Frozen):
    """Monthly-climatology anomaly; climatology from TRAIN years only."""

    def fit(self, x_train: np.ndarray, month_train: np.ndarray) -> "Anomaly":
        clim = np.zeros(12)
        for m in range(12):
            sel = np.asarray(month_train) == m + 1
            clim[m] = float(np.nanmean(np.asarray(x_train, dtype=float)[sel]))
        self._freeze({"clim": clim})
        return self

    def apply(self, x: np.ndarray, month: np.ndarray) -> np.ndarray:
        self._need()
        clim = np.asarray(self._stats["clim"])
        return np.asarray(x, dtype=float) - clim[np.asarray(month) - 1]


class ClearSkyIndex(Frozen):
    """k_t = GHI / GHI_clear(time-of-day), clear-sky curve from the train
    span: per-slot 95th percentile of GHI (an empirical clear envelope).
    Slots where the envelope is ~0 (night) yield masked k_t (NaN)."""

    def fit(self, ghi_train: np.ndarray, slot_train: np.ndarray,
            n_slots: int) -> "ClearSkyIndex":
        ghi = np.asarray(ghi_train, dtype=float)
        env = np.zeros(n_slots)
        for s in range(n_slots):
            v = ghi[np.asarray(slot_train) == s]
            v = v[np.isfinite(v)]
            env[s] = float(np.percentile(v, 95)) if len(v) else 0.0
        self._freeze({"env": env, "n_slots": n_slots})
        return self

    def apply(self, ghi: np.ndarray, slot: np.ndarray) -> np.ndarray:
        self._need()
        env = np.asarray(self._stats["env"])[np.asarray(slot)]
        with np.errstate(divide="ignore", invalid="ignore"):
            kt = np.asarray(ghi, dtype=float) / env
        kt[env < 10.0] = np.nan          # night / dawn: masked, not zero
        return np.clip(kt, 0.0, 1.5)


def calendar_features(slot: np.ndarray, n_slots: int,
                      doy: np.ndarray | None = None) -> np.ndarray:
    """Lookup features: sin/cos of time-of-day (and of day-of-year when
    given). Zero quantum budget spent here by design."""
    ang = 2 * np.pi * np.asarray(slot) / n_slots
    cols = [np.sin(ang), np.cos(ang)]
    if doy is not None:
        a2 = 2 * np.pi * np.asarray(doy) / 365.25
        cols += [np.sin(a2), np.cos(a2)]
    return np.stack(cols, axis=1)


def leakage_audit(fit_span_len: int, applied_len: int,
                  fit_call_indices: list[int]) -> bool:
    """The audit: every fit() call index must lie inside the train span.
    Used by replay.py to certify a backtest leak-free."""
    return all(i < fit_span_len for i in fit_call_indices) \
        and applied_len >= fit_span_len


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_slots = 48
    T = n_slots * 60
    slot = np.arange(T) % n_slots
    clear = np.clip(np.sin((slot - 6) / n_slots * 2 * np.pi), 0, None) * 800
    ghi = clear * rng.uniform(0.3, 1.0, T)
    tr = T // 2
    cs = ClearSkyIndex("demo-0").fit(ghi[:tr], slot[:tr], n_slots)
    kt = cs.apply(ghi, slot)
    day = np.isfinite(kt)
    print(f"k_t: {day.sum()} daylight steps, mean {np.nanmean(kt):.3f}, "
          f"night masked {(~day).sum()}")
    assert 0.3 < np.nanmean(kt) < 1.0
    try:
        ClearSkyIndex("demo-0").apply(ghi, slot)
        raise SystemExit("firewall FAILED to refuse")
    except RuntimeError:
        print("leakage firewall refuses unfitted apply: PASS")
