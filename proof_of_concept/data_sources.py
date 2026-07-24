"""data_sources.py -- real monthly climate drivers + leak-free transforms.

Three REAL, physically-coupled drivers of the El Nino-Southern Oscillation
(ENSO) system, pinned as reproducible CSV caches in ``data/``:

  * ``sst`` -- Nino-region sea-surface temperature, raw degrees C.
      NOAA monthly Nino SST via the statsmodels ``elnino`` dataset,
      1950-01..2010-12 (732 months). Ocean side of ENSO.
  * ``soi`` -- Southern Oscillation Index (Tahiti - Darwin sea-level
      pressure anomaly). NOAA CPC, 1951-01.. . Atmospheric side of ENSO;
      strongly NEGATIVE during El Nino (when SST is warm).
  * ``pdo`` -- Pacific Decadal Oscillation index. NOAA PSL, 1948-01.. .
      Decadal North-Pacific SST pattern that modulates ENSO.

Why these three: a *compound* El Nino event is warm ocean AND collapsed
Southern Oscillation occurring together -- a genuine multi-driver co-exceedance
with real tail dependence, not two independent alarms. That coupling is what
the multivariate reservoir is meant to exploit (see README).

Leak rule (non-negotiable): every fitted statistic -- the monthly
climatology used to deseasonalise, the [0,1] encoder scaler, and the
event thresholds -- is computed on the TRAIN SPAN ONLY. The train span
ends exactly at the last training TARGET month implied by the shared
(L, HORIZON, train_frac) harness, so the leak boundary cannot drift.

The pinned CSVs are authoritative (upstream packages/URLs cannot silently
change the numbers). ``refresh_caches()`` re-derives them from source; run
``python data_sources.py --refresh`` to regenerate.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"

# Shared harness constants (kept equal to the single-variable reference).
L = 24            # input window length, months
HORIZON = 3       # forecast lead time, months ahead
TRAIN_FRAC = 0.8

# name -> (csv, human label, event spec). Event spec is either
#   ("abs", v)  : |anomaly| > v in the anomaly's own units (physical), or
#   ("sd",  k)  : |anomaly| > k * train-span std (dimensionless).
DATASETS = {
    "sst": ("enso.csv", "Nino-region SST anomaly [C]", ("abs", 0.5)),
    "soi": ("soi.csv",  "Southern Oscillation Index anomaly", ("sd", 1.0)),
    "pdo": ("pdo.csv",  "Pacific Decadal Oscillation anomaly", ("sd", 1.0)),
}

# Compound event sign per driver: +1 means "high is the alarming tail", -1
# "low is". A warm (El Nino) compound event is SST high AND SOI low.
COMPOUND_SIGNS = {"sst": +1, "soi": -1, "pdo": +1}

MONTH_COLS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
SOI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
PDO_URL = "https://psl.noaa.gov/data/correlation/pdo.data"


# ------------------------------------------------------------------ loading
def load_series(name, cache_dir=DATA):
    """Pinned monthly series -> DataFrame[date, year, month, value]."""
    fname = DATASETS[name][0]
    path = Path(cache_dir) / fname
    if not path.exists():
        refresh_caches(only=name, cache_dir=cache_dir)
    df = pd.read_csv(path, parse_dates=["date"])
    if "sst" in df.columns:                      # normalise the value column
        df = df.rename(columns={"sst": "value"})
    df = df[["date", "year", "month", "value"]].sort_values("date")
    df = df.reset_index(drop=True)
    _validate(df, name)
    return df


def _validate(df, name):
    assert df.value.notna().all(), f"{name}: NaN value"
    mi = df.date.dt.year * 12 + df.date.dt.month
    assert (mi.diff().dropna() == 1).all(), f"{name}: calendar gap"
    assert len(df) >= 200, f"{name}: suspiciously short ({len(df)} months)"


# ------------------------------------------------- leak-free transforms
def train_month_cutoff(T, L=L, H=HORIZON, train_frac=TRAIN_FRAC):
    """Months [0, cutoff) are the train span. cutoff is the first test TARGET
    month: samples k = L-1 .. T-H-1, chrono cut at train_frac, last train
    target = (L-1)+(cut-1)+H, so n_train_months = L + H + cut - 1."""
    n_samples = T - L - H + 1
    cut = int(n_samples * train_frac)
    return L + H + cut - 1


def monthly_climatology(month, value, n_train):
    """Per-calendar-month mean over the first n_train months ONLY."""
    clim = np.zeros(12)
    m_tr, v_tr = month[:n_train], value[:n_train]
    for j in range(12):
        clim[j] = v_tr[m_tr == j + 1].mean()
    return clim


def deseasonalize(month, value, n_train):
    """Full-series anomaly against the train-only monthly climatology."""
    clim = monthly_climatology(month, value, n_train)
    return value - clim[month - 1], clim


def fit_scaler(y, n_train):
    """Affine map -> [0,1] fit on the train span; clips outside (test months
    may exceed the train range and the quantum encoding needs bounded input)."""
    lo, hi = float(y[:n_train].min()), float(y[:n_train].max())

    def scaler(x):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    return scaler, (lo, hi)


def event_threshold(name, y, n_train):
    """Resolve the DATASETS event spec into a numeric anomaly threshold,
    fit on the train span only when it is std-relative."""
    kind, v = DATASETS[name][2]
    if kind == "abs":
        return float(v)
    if kind == "sd":
        return float(v * y[:n_train].std())
    raise ValueError(kind)


def raw_series(name):
    """(month, value) arrays of the pinned raw series -- for callers that must
    re-fit the anomaly/scaler transforms themselves (e.g. strict per-fold CV,
    where the climatology may only see each fold's own train months)."""
    df = load_series(name)
    return df.month.to_numpy(), df.value.to_numpy(dtype=float)


# ------------------------------------------------------------------ prepare
def prepare_univariate(name, train_frac=TRAIN_FRAC, verbose=True):
    """Everything one univariate forecast needs, leak-safe."""
    df = load_series(name)
    month = df.month.to_numpy()
    raw = df.value.to_numpy(dtype=float)
    n_train = train_month_cutoff(len(df), train_frac=train_frac)
    y, clim = deseasonalize(month, raw, n_train)
    scaler, (lo, hi) = fit_scaler(y, n_train)
    thr = event_threshold(name, y, n_train)
    if verbose:
        print(f"[{name}] {len(df)} months "
              f"{df.date.iloc[0]:%Y-%m}..{df.date.iloc[-1]:%Y-%m}; "
              f"train [0,{n_train}) -> first test target "
              f"{df.date.iloc[n_train]:%Y-%m}; event |anom|>{thr:.3f}")
    return dict(name=name, label=DATASETS[name][1], df=df,
                dates=df.date.to_numpy(), month=month, raw=raw,
                y=y, clim=clim, scaler=scaler, u=scaler(y),
                scaler_range=(lo, hi), n_train=n_train,
                event_threshold=thr)


def _common_span(dfs):
    """Intersection of monthly indices across a list of DataFrames -> the
    integer month indices (year*12+month) present in ALL of them, contiguous."""
    keys = [set((d.date.dt.year * 12 + d.date.dt.month).tolist()) for d in dfs]
    common = sorted(set.intersection(*keys))
    common = np.array(common)
    assert (np.diff(common) == 1).all(), "common span has calendar gaps"
    return common


def prepare_compound(names=("sst", "soi"), train_frac=TRAIN_FRAC, verbose=True):
    """Align several drivers on their common monthly span and apply the SAME
    leak-free anomaly + scaler transforms per channel.

    Returns U (T, C) scaled to [0,1] for the reservoir encoding, Y (T, C)
    anomalies in physical units for event labelling, plus per-channel event
    thresholds and the compound (joint co-exceedance) label."""
    names = list(names)
    dfs = [load_series(n) for n in names]
    common = _common_span(dfs)
    mask = [np.isin(d.date.dt.year * 12 + d.date.dt.month, common) for d in dfs]
    dfs = [d[m].reset_index(drop=True) for d, m in zip(dfs, mask)]
    T = len(common)
    n_train = train_month_cutoff(T, train_frac=train_frac)
    dates = dfs[0].date.to_numpy()

    Y = np.zeros((T, len(names)))
    U = np.zeros((T, len(names)))
    thr = np.zeros(len(names))
    for c, (nm, d) in enumerate(zip(names, dfs)):
        month = d.month.to_numpy()
        y, _ = deseasonalize(month, d.value.to_numpy(float), n_train)
        scaler, _ = fit_scaler(y, n_train)
        Y[:, c], U[:, c] = y, scaler(y)
        thr[c] = event_threshold(nm, y, n_train)

    signs = np.array([COMPOUND_SIGNS[n] for n in names])
    # per-channel exceedance in its alarming direction, then the AND across
    # channels = the compound co-exceedance.
    exceed = (Y * signs) > thr                    # (T, C) bool
    compound = exceed.all(axis=1)
    if verbose:
        print(f"[compound {'+'.join(names)}] {T} months "
              f"{pd.Timestamp(dates[0]):%Y-%m}..{pd.Timestamp(dates[-1]):%Y-%m}; "
              f"train [0,{n_train}); compound base rate "
              f"{compound.mean():.3f} ({compound.sum()} months)")
    return dict(names=names, dates=dates, U=U, Y=Y, thresholds=thr,
                signs=signs, exceed=exceed, compound=compound,
                n_train=n_train, T=T)


# ------------------------------------------------------------------ refresh
def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def _year_rows(text, start_after=None, stop_at=None, missing=(-999.9,)):
    """Parse NOAA 'YEAR jan..dec' blocks -> [(date, year, month, value)].
    Un-glues fused negatives (NOAA writes ``-999.9-999.9``) before splitting."""
    lines = text.splitlines()
    if start_after is not None:
        i = next(k for k, ln in enumerate(lines) if start_after in ln)
        lines = lines[i + 1:]
    recs = []
    for ln in lines:
        if stop_at is not None and stop_at in ln:
            break
        parts = ln.replace("-", " -").split()
        if len(parts) < 13:
            continue
        try:
            yr = int(parts[0])
        except ValueError:
            continue
        if not 1800 <= yr <= 2100:
            continue
        for m, v in enumerate(parts[1:13], start=1):
            try:
                x = float(v)
            except ValueError:
                continue
            if any(abs(x - mv) < 1e-3 for mv in missing) or abs(x) > 90:
                continue
            recs.append((pd.Timestamp(yr, m, 1), yr, m, x))
    return recs


def _longest_gapless(df):
    mi = (df.date.dt.year * 12 + df.date.dt.month).to_numpy()
    breaks = np.where(np.diff(mi) != 1)[0]
    bounds = np.concatenate([[0], breaks + 1, [len(df)]])
    a, b = max(zip(bounds[:-1], bounds[1:]), key=lambda ab: ab[1] - ab[0])
    return df.iloc[a:b].reset_index(drop=True)


def refresh_caches(only=None, cache_dir=DATA):
    """Re-derive the pinned CSVs from their upstream sources."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if only in (None, "sst"):
        import statsmodels.api as sm
        wide = sm.datasets.elnino.load_pandas().data
        recs = []
        for _, row in wide.iterrows():
            yr = int(row["YEAR"])
            for m, col in enumerate(MONTH_COLS, start=1):
                recs.append((pd.Timestamp(yr, m, 1), yr, m, float(row[col])))
        df = pd.DataFrame(recs, columns=["date", "year", "month", "sst"])
        df.to_csv(cache_dir / "enso.csv", index=False)
        print(f"refreshed enso.csv ({len(df)} months)")
    if only in (None, "soi"):
        recs = _year_rows(_fetch(SOI_URL), start_after="ANOMALY",
                          stop_at="STANDARDIZED", missing=(-999.9,))
        df = _longest_gapless(pd.DataFrame(
            recs, columns=["date", "year", "month", "value"]
        ).drop_duplicates("date").sort_values("date").reset_index(drop=True))
        df.to_csv(cache_dir / "soi.csv", index=False, float_format="%.4f")
        print(f"refreshed soi.csv ({len(df)} months)")
    if only in (None, "pdo"):
        recs = _year_rows(_fetch(PDO_URL),
                          missing=(-9.9, -99.99, 99.99, -999.0))
        df = _longest_gapless(pd.DataFrame(
            recs, columns=["date", "year", "month", "value"]
        ).drop_duplicates("date").sort_values("date").reset_index(drop=True))
        df.to_csv(cache_dir / "pdo.csv", index=False, float_format="%.4f")
        print(f"refreshed pdo.csv ({len(df)} months)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch and re-pin the CSV caches from source")
    args = ap.parse_args()
    if args.refresh:
        refresh_caches()
    for nm in DATASETS:
        prepare_univariate(nm)
    prepare_compound(("sst", "soi"))
    prepare_compound(("sst", "soi", "pdo"))
