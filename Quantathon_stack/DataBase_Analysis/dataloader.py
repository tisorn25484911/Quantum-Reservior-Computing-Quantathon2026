"""dataloader.py -- uniform access to everything under ``Quantathon_stack/Data``.

The three data tiers arrive in different containers (``.npz`` ensembles,
``.csv`` records with a date column, ``.csv`` records with a timestamp column)
and on different clocks (natural time units, months, half-hours, hours). Every
downstream tool here -- the FFT, the Lyapunov estimator, the plots -- needs the
same three things: a scalar series, a sample spacing, and the name of the time
unit that spacing is measured in. This module is the one place that knows how
to get from a file to those three things.

    from dataloader import load, list_datasets

    s = load("lorenz63")     # Series(x=(4000,), dt=0.01, ensemble=(12, 4000))
    s = load("nino34")       # Series(x=(918,), dt=1/12, time_unit="yr")

Datasets are addressed by short key, not by path, so a caller never has to
know which tier a series lives in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Data"


# ----------------------------------------------------------------------
# Container
# ----------------------------------------------------------------------
@dataclass
class Series:
    """A scalar series plus everything needed to interpret its time axis.

    Attributes
    ----------
    key, name  : short handle and human label
    x          : (N,) the analysed observable
    dt         : sample spacing, in `time_unit`
    time_unit  : unit of `dt` ("yr", "h", "natural", ...)
    unit       : physical unit of `x` ("degC", "MW", "W/m^2", "")
    tier       : "chaotic" | "real" | "surrogate"
    ensemble   : (R, N) realizations if the source has them, else None
    lyap_true  : published lambda_1 if known, else None
    index      : original pandas index (dates) for real/surrogate, else None
    """

    key: str
    name: str
    x: np.ndarray
    dt: float
    time_unit: str = ""
    unit: str = ""
    tier: str = ""
    ensemble: np.ndarray | None = None
    lyap_true: float | None = None
    index: pd.Index | None = None

    @property
    def t(self) -> np.ndarray:
        """Time axis in `time_unit`, starting at zero."""
        return np.arange(len(self.x)) * self.dt

    @property
    def n_realizations(self) -> int:
        return 0 if self.ensemble is None else int(self.ensemble.shape[0])

    def __repr__(self) -> str:
        ens = f", ensemble={self.ensemble.shape}" if self.ensemble is not None else ""
        lam = f", lyap_true={self.lyap_true:.4f}" if self.lyap_true is not None else ""
        return (f"Series({self.key!r}, N={len(self.x)}, dt={self.dt:.6g} "
                f"{self.time_unit}{ens}{lam})")


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
# dt for the real/surrogate tiers is a property of the *record*, not of the
# file, so it is declared here rather than sniffed: monthly data on a calendar
# has unequal spacing in days, and pretending otherwise by differencing the
# timestamps would put a spurious 12-month modulation into the spectrum.
_CHAOTIC = {
    "lorenz63":      ("lorenz63.npz",       "Lorenz-63 x"),
    "vallieselnino": ("vallieselnino.npz",  "Vallis ENSO oscillator"),
    "lorenz84":      ("lorenz84.npz",       "Lorenz-84 circulation"),
    "hadley":        ("hadley.npz",         "Hadley cell convection"),
    "rikitake":      ("rikitakedynamo.npz", "Rikitake dynamo"),
}

_TABULAR = {
    "nino34": dict(
        path="real/nino34_anom.csv", column="anom", index="date",
        dt=1.0 / 12.0, time_unit="yr", unit="degC", tier="real",
        name="Nino 3.4 SST anomaly",
    ),
    "nino12": dict(
        path="real/enso_nino12_1950_2010.csv", column="sst", index="date",
        dt=1.0 / 12.0, time_unit="yr", unit="degC", tier="real",
        name="Nino 1+2 SST (1950-2010)",
    ),
    "solar": dict(
        path="surrogate/solar_ghi.csv", column="ghi", index="timestamp",
        dt=0.5, time_unit="h", unit="W/m^2", tier="surrogate",
        name="Solar GHI (30 min)",
    ),
    "load": dict(
        path="surrogate/system_load.csv", column="load_mw", index="timestamp",
        dt=1.0, time_unit="h", unit="MW", tier="surrogate",
        name="System load (hourly)",
    ),
    # ---- observed records added with the expanded real tier ----------------
    # `longest_run=True` crops to the longest gapless stretch instead of
    # interpolating. Use it wherever the outages are long enough that a fill
    # would dominate the divergence curve -- see the TAO note below.
    "tao": dict(
        path="real/tao_daily_sst.csv", column="0N140W", index="date",
        dt=1.0, time_unit="day", unit="degC", tier="real",
        name="TAO 0N140W SST (daily)", longest_run=True,
    ),
    "nyc": dict(
        path="real/ghcnd_new_york_central_park.csv", column="tmax", index="date",
        dt=1.0, time_unit="day", unit="degC", tier="real",
        name="New York Central Park Tmax",
    ),
    "lax": dict(
        path="real/ghcnd_los_angeles_airport.csv", column="tmax", index="date",
        dt=1.0, time_unit="day", unit="degC", tier="real",
        name="Los Angeles Airport Tmax",
    ),
    "potomac": dict(
        path="real/usgs_potomac_daily_discharge.csv", column="discharge_cfs",
        index="date", dt=1.0, time_unit="day", unit="cfs", tier="real",
        name="Potomac discharge (daily)", log=True,
    ),
    "potomac15": dict(
        path="real/usgs_potomac_15min_discharge.csv", column="discharge_cfs",
        index="timestamp", dt=0.25, time_unit="h", unit="cfs", tier="real",
        name="Potomac discharge (15 min)", log=True, resample="15min",
    ),
    "opsd": dict(
        path="real/opsd_hourly_load.csv", column="DE", index="timestamp",
        dt=1.0, time_unit="h", unit="MW", tier="real",
        name="German load, actual (hourly)",
    ),
    "brest": dict(
        path="real/psmsl_brest_monthly_sealevel.csv", column="rlr_mm",
        index="date", dt=1.0 / 12.0, time_unit="yr", unit="mm", tier="real",
        name="Brest sea level (monthly)", longest_run=True,
    ),
    "cuxhaven": dict(
        path="real/psmsl_cuxhaven_monthly_sealevel.csv", column="rlr_mm",
        index="date", dt=1.0 / 12.0, time_unit="yr", unit="mm", tier="real",
        name="Cuxhaven sea level (monthly)",
    ),
    # ---- CP aquaculture use case (fetch_cp_data.py) ------------------------
    # Upper Gulf of Thailand off the Mae Klong estuary -- the CP-waters
    # counterpart to `tao`: same variable and cadence, company-relevant site.
    "got_sst": dict(
        path="real/oisst_gulf_thailand_daily_sst.csv", column="sst",
        index="date", dt=1.0, time_unit="day", unit="degC", tier="real",
        name="Gulf of Thailand SST (OISST daily)", longest_run=True,
    ),
    "got_ersst": dict(
        path="real/ersst_gulf_thailand_monthly_sst.csv", column="sst",
        index="date", dt=1.0 / 12.0, time_unit="yr", unit="degC", tier="real",
        name="Gulf of Thailand SST (ERSSTv5 monthly reconstruction)",
    ),
    "oni": dict(
        path="real/oni_index.csv", column="oni", index="date",
        dt=1.0 / 12.0, time_unit="yr", unit="degC", tier="real",
        name="Oceanic Nino Index (3-mo running mean)",
    ),
    "maeklong_rain": dict(
        path="real/nasa_power_mae_klong_daily_rain.csv", column="rain_mm",
        index="date", dt=1.0, time_unit="day", unit="mm", tier="real",
        name="Mae Klong basin rainfall (NASA POWER daily)",
    ),
    # ---- added from the manually-placed new_data/ archive -------------------
    "got_hadisst": dict(
        path="real/hadisst_gulf_thailand_monthly_sst.csv", column="sst",
        index="date", dt=1.0 / 12.0, time_unit="yr", unit="degC", tier="real",
        name="Gulf of Thailand SST (HadISST1 monthly, cross-check)",
    ),
    "maeklong_spei": dict(
        path="real/spei03_mae_klong_monthly.csv", column="spei",
        index="date", dt=1.0 / 12.0, time_unit="yr", unit="", tier="real",
        name="Mae Klong basin SPEI-03 (drought index, monthly)",
    ),
}

CHAOTIC_KEYS = tuple(_CHAOTIC)
REAL_KEYS = ("nino34", "nino12", "tao", "nyc", "lax", "potomac", "potomac15",
             "opsd", "brest", "cuxhaven",
             "got_sst", "got_ersst", "oni", "maeklong_rain",
             "got_hadisst", "maeklong_spei")
SURROGATE_KEYS = ("solar", "load")
ALL_KEYS = CHAOTIC_KEYS + REAL_KEYS + SURROGATE_KEYS


def list_datasets(tier: str | None = None) -> list[str]:
    """Dataset keys, optionally restricted to one tier."""
    if tier is None:
        return list(ALL_KEYS)
    return {"chaotic": list(CHAOTIC_KEYS),
            "real": list(REAL_KEYS),
            "surrogate": list(SURROGATE_KEYS)}[tier]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _load_chaotic(key: str, member: int) -> Series:
    fname, name = _CHAOTIC[key]
    path = DATA / "chaotic" / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python fetch_data.py --dysts` in {DATA}")
    with np.load(path, allow_pickle=True) as d:
        ens = np.asarray(d["x"], float)
        dt = float(d["dt"])
        lyap = float(d["lyap"])
    return Series(key=key, name=name, x=ens[member], dt=dt,
                  time_unit="natural", unit="", tier="chaotic",
                  ensemble=ens, lyap_true=lyap)


def _longest_valid_run(x: np.ndarray) -> slice:
    """Slice covering the longest run of consecutive non-NaN samples."""
    ok = np.isfinite(x)
    if ok.all():
        return slice(0, len(x))
    # boundaries of each constant-validity block
    edges = np.flatnonzero(np.diff(ok.astype(int))) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(x)]
    runs = [(a, b) for a, b in zip(starts, stops) if ok[a]]
    if not runs:
        raise ValueError("series is entirely NaN")
    a, b = max(runs, key=lambda ab: ab[1] - ab[0])
    return slice(int(a), int(b))


def _load_tabular(key: str, fill_gaps: bool) -> Series:
    spec = _TABULAR[key]
    path = DATA / spec["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python fetch_data.py` in {DATA}")
    df = pd.read_csv(path, parse_dates=[spec["index"]])
    idx = df[spec["index"]]

    # The instantaneous-discharge feed is not on a clean grid: ~44% of its
    # steps differ from the nominal 15 minutes. Everything downstream assumes
    # a uniform dt, so put it on one before anything else touches it.
    if spec.get("resample"):
        df = (df.set_index(spec["index"])
                .resample(spec["resample"]).mean()
                .reset_index())
        idx = df[spec["index"]]

    x = df[spec["column"]].to_numpy(float)

    # Discharge spans three orders of magnitude and is strongly right-skewed,
    # so a handful of flood peaks would otherwise set the scale for both the
    # spectrum and the neighbour search. Analyse log10(Q) instead.
    if spec.get("log"):
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log10(np.where(x > 0, x, np.nan))

    if spec.get("longest_run"):
        # Interpolating across a multi-year outage measures the interpolator,
        # not the system. Take the longest continuous stretch instead and let
        # the caller see how much of the record survived.
        sl = _longest_valid_run(x)
        x, idx = x[sl], idx.iloc[sl].reset_index(drop=True)

    # Remaining short interior gaps are filled: leaving NaNs in place would
    # poison both the FFT and the divergence curve. Linear, interior only, so
    # the fill adds no power of its own beyond the trend already there.
    if fill_gaps and np.isnan(x).any():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy(float)

    return Series(key=key, name=spec["name"], x=x, dt=spec["dt"],
                  time_unit=spec["time_unit"], unit=spec["unit"],
                  tier=spec["tier"], index=idx)


def load(key: str, member: int = 0, fill_gaps: bool = True) -> Series:
    """Load one dataset by key.

    `member` picks which realization of a chaotic system becomes the scalar
    `x`; the full ensemble is kept on the Series regardless, because the
    ensemble estimator in lyapunov.py needs all of it.
    """
    key = key.lower().removesuffix(".npz").removesuffix(".csv")
    if key in _CHAOTIC:
        return _load_chaotic(key, member)
    if key in _TABULAR:
        return _load_tabular(key, fill_gaps)
    raise KeyError(f"unknown dataset {key!r}; known: {', '.join(ALL_KEYS)}")


def load_all(tier: str | None = None, **kw) -> dict[str, Series]:
    """Every dataset that is actually present on disk, keyed by name.

    Missing files are skipped rather than raised: the chaotic tier is opt-in
    (`fetch_data.py --dysts`) and a partial directory is a normal state.
    """
    out = {}
    for k in list_datasets(tier):
        try:
            out[k] = load(k, **kw)
        except FileNotFoundError:
            continue
    return out


if __name__ == "__main__":
    for k, s in load_all().items():
        print(f"{k:16s} {s!r}")
