"""loaders.py -- dataset loading and the ONE chronological split.

Single source of truth for how a series is split in time. Per the Phase-0
soundness review (IMPLEMENTATION_PHASES.md), the irreversible decision -- final
20% is an untouched test span; the first 80% is development, validated by
rolling origin -- is encoded HERE so no later code can re-split. Every
experiment obtains its indices from `split()`.

Raw files under data/raw/ are never modified. `load_raw()` validates that the
series is monthly and gap-free and returns a frozen `Series`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"

# Frozen chronology (Phase 0). Changing this is a preregistration change.
TEST_FRACTION = 0.20

# name -> (filename, value column, human identity) resolved in the P0 audit.
DATASETS = {
    "enso": ("enso.csv", "sst",
             "statsmodels 'elnino' = NOAA Nino 1+2 region (0-10S, 90-80W) RAW "
             "monthly SST [degC]; NOT an anomaly, NOT ONI/Nino3.4/MEI"),
    "pdo": ("pdo.csv", "value", "NOAA PSL Mantua-style PDO index (already an anomaly)"),
    "soi": ("soi.csv", "value",
            "NOAA CPC SOI = Tahiti-Darwin SLP ANOMALY (not Troup-standardised)"),
}


@dataclass(frozen=True)
class Series:
    """A validated, immutable monthly climate series (raw units)."""
    name: str
    dates: np.ndarray
    month: np.ndarray          # calendar month 1..12
    values: np.ndarray         # float64, raw units
    units: str
    identity: str
    sha256: str

    def __len__(self) -> int:
        return len(self.values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(name: str) -> Series:
    """Load and validate one raw series (monthly, gap-free, no NaN)."""
    fname, col, identity = DATASETS[name]
    path = RAW / fname
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    values = df[col].to_numpy(dtype=float)
    assert not np.isnan(values).any(), f"{name}: NaN in values"
    mi = df["date"].dt.year * 12 + df["date"].dt.month
    assert (mi.diff().dropna() == 1).all(), f"{name}: not gap-free monthly"
    assert df["date"].is_unique, f"{name}: duplicate dates"
    units = "degC" if name == "enso" else "index"
    return Series(name, df["date"].to_numpy(), df["date"].dt.month.to_numpy(),
                  values, units, identity, _sha256(path))


def split(n: int, test_fraction: float = TEST_FRACTION) -> dict:
    """The ONE chronological split: dev = first (1-f), test = final f."""
    cut = int(round(n * (1.0 - test_fraction)))
    return {"dev": np.arange(0, cut), "test": np.arange(cut, n),
            "cut": cut, "n": n, "test_fraction": test_fraction}


def rolling_origin_folds(dev_n: int, n_folds: int = 6, min_train_frac: float = 0.4):
    """Expanding-window folds INSIDE the development span only (train < val)."""
    start = int(round(min_train_frac * dev_n))
    bounds = np.linspace(start, dev_n, n_folds + 1).astype(int)
    for f in range(n_folds):
        tr = np.arange(0, bounds[f])
        va = np.arange(bounds[f], bounds[f + 1])
        if len(va):
            yield tr, va


def train_cutoff(n: int) -> int:
    """First index of the untouched test span (== split()['cut'])."""
    return split(n)["cut"]
