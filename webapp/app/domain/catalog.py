"""catalog.py -- dataset discovery and access for the web layer.

Wraps ``DataBase_Analysis.dataloader`` so the routes deal in plain dicts and
never import pandas or numpy types they would have to serialise by hand.
Loading is cached: the underlying loader reads CSV/NPZ from disk on every call,
and the forecast page hits the same series repeatedly.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import TIER_LABELS, TIER_NOTE

import dataloader as _dl   # noqa: E402  (sys.path wired in config)


# Series this demo is aimed at: the applied, forecast-relevant records. The
# chaotic tier stays available (it is the controlled testbed) but is not the
# headline, because a Lorenz forecast is not an application claim.
FEATURED = ("solar", "load", "opsd", "nino34", "tao")


@lru_cache(maxsize=32)
def _load(key: str):
    return _dl.load(key)


def load_series(key: str):
    """Load a series by short key, cached. Raises KeyError for an unknown key."""
    try:
        return _load(key)
    except Exception as exc:                       # loader raises various types
        raise KeyError(f"unknown or unreadable dataset: {key!r} ({exc})")


def describe(key: str) -> dict:
    """Summary of one dataset, JSON-safe, including its provenance tier."""
    s = load_series(key)
    x = np.asarray(s.x, dtype=float)
    finite = x[np.isfinite(x)]
    lag1 = (float(np.corrcoef(finite[:-1], finite[1:])[0, 1])
            if len(finite) > 2 else float("nan"))
    return {
        "key": s.key,
        "name": s.name,
        "tier": s.tier,
        "tier_label": TIER_LABELS.get(s.tier, s.tier),
        "tier_note": TIER_NOTE.get(s.tier, ""),
        "n": int(len(x)),
        "dt": float(s.dt),
        "time_unit": s.time_unit or "sample",
        "unit": s.unit or "",
        "n_realizations": int(s.n_realizations),
        "lyap_true": (None if s.lyap_true is None else float(s.lyap_true)),
        "has_dates": s.index is not None,
        "mean": float(np.mean(finite)) if len(finite) else None,
        "std": float(np.std(finite)) if len(finite) else None,
        "min": float(np.min(finite)) if len(finite) else None,
        "max": float(np.max(finite)) if len(finite) else None,
        "n_missing": int(np.sum(~np.isfinite(x))),
        "lag1_autocorr": lag1,
        "featured": s.key in FEATURED,
    }


def list_datasets(tier: str | None = None) -> list[dict]:
    """Every dataset present on disk, optionally filtered to one tier.

    A dataset that fails to load is reported with an ``error`` field rather
    than dropped: a file that is present but unreadable is something the team
    needs to see, not something the UI should silently hide.
    """
    out = []
    for key in _dl.list_datasets():
        try:
            d = describe(key)
        except Exception as exc:
            d = {"key": key, "name": key, "tier": "unknown", "error": str(exc),
                 "tier_label": "unreadable", "featured": False}
        if tier and d.get("tier") != tier:
            continue
        out.append(d)
    # Featured first, then by tier, then alphabetically -- stable and readable.
    return sorted(out, key=lambda d: (not d.get("featured"),
                                      d.get("tier", ""), d["key"]))


def series_values(key: str, max_points: int | None = None
                  ) -> tuple[np.ndarray, dict]:
    """The scalar series as a finite float array, plus its descriptor.

    Non-finite samples are interpolated over rather than dropped, so the time
    axis stays uniform -- the reservoir is driven at a fixed step and a silently
    shortened series would misalign every forecast horizon.
    """
    s = load_series(key)
    x = np.asarray(s.x, dtype=float)
    bad = ~np.isfinite(x)
    if bad.any():
        idx = np.arange(len(x))
        if bad.all():
            raise ValueError(f"dataset {key!r} has no finite samples")
        x = np.interp(idx, idx[~bad], x[~bad])
    if max_points and len(x) > max_points:
        x = x[-int(max_points):]          # most recent window
    return x, describe(key)
