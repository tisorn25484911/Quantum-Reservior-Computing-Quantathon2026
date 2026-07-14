"""datasets.py -- ENSO loader + leak-free anomaly / scaler transforms.

Data: NOAA monthly Nino SST via the statsmodels `elnino` dataset (REAL,
1950--2010, 61 years x 12 = 732 months). The first load pins a copy to
data/raw/enso.csv; afterwards the pinned copy is authoritative
(reproducibility -- upstream package updates cannot silently change data).

Leak rule (CLAUDE.md, non-negotiable): every statistic -- the monthly
climatology AND the [0,1] scaler -- is computed on the TRAIN SPAN ONLY.
The train span is derived from the shared harness split so that its last
month is exactly the last training TARGET month:

    samples k = L-1 .. T-H-1        (n_samples = T - L - H + 1)
    chrono cut = int(train_frac * n_samples)
    last train target  = month index (L-1) + (cut-1) + H
    n_train_months     = L + H + cut - 1
    first test target  = month index n_train_months   (adjacent, no gap)

Anomaly = SST - climatology[calendar month], applied to the FULL series
using the train-only climatology. L and HORIZON are imported from
baselines so the leak boundary cannot drift from the harness.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import HORIZON, L

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "raw" / "enso.csv"
TRAIN_FRAC = 0.8
MONTH_COLS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def print_config():
    print(f"datasets config: cache={CACHE.relative_to(REPO)} "
          f"L={L} H={HORIZON} train_frac={TRAIN_FRAC}")


# ------------------------------------------------------------------ load
def load_enso(cache=CACHE):
    """Monthly long-format series: columns date, year, month (1-12), sst."""
    cache = Path(cache)
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        src = "pinned cache"
    else:
        import statsmodels.api as sm
        wide = sm.datasets.elnino.load_pandas().data
        recs = []
        for _, row in wide.iterrows():
            yr = int(row["YEAR"])
            for m, col in enumerate(MONTH_COLS, start=1):
                recs.append((pd.Timestamp(yr, m, 1), yr, m, float(row[col])))
        df = pd.DataFrame(recs, columns=["date", "year", "month", "sst"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        src = "statsmodels elnino -> pinned"
    _validate(df)
    print(f"enso: {len(df)} months "
          f"{df.date.iloc[0]:%Y-%m}..{df.date.iloc[-1]:%Y-%m} ({src})")
    return df


def _validate(df):
    assert list(df.columns) == ["date", "year", "month", "sst"], df.columns
    assert len(df) == 732, f"expected 732 months, got {len(df)}"
    assert df.sst.notna().all(), "NaN in SST"
    mi = df.date.dt.year * 12 + df.date.dt.month
    assert (mi.diff().dropna() == 1).all(), "calendar gap in monthly series"
    assert df.sst.between(15, 35).all(), "SST outside plausible range [15,35] C"


# ------------------------------------------------- leak-free transforms
def train_month_cutoff(T, L=L, H=HORIZON, train_frac=TRAIN_FRAC):
    """Months [0, cutoff) form the train span (see module docstring)."""
    n_samples = T - L - H + 1
    cut = int(n_samples * train_frac)
    return L + H + cut - 1


def monthly_climatology(month, sst, n_train):
    """Per-calendar-month mean over the first n_train months ONLY."""
    clim = np.zeros(12)
    m_tr, s_tr = month[:n_train], sst[:n_train]
    for j in range(12):
        clim[j] = s_tr[m_tr == j + 1].mean()
    return clim


def anomaly(month, sst, n_train):
    """Full-series anomaly against the train-only climatology."""
    clim = monthly_climatology(month, sst, n_train)
    return sst - clim[month - 1], clim


def fit_scaler(y, n_train):
    """Affine map -> [0,1] fit on train span; clips outside (test months
    may exceed the train range -- quantum encoding needs bounded input)."""
    lo, hi = float(y[:n_train].min()), float(y[:n_train].max())

    def scaler(x):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    return scaler, (lo, hi)


def prepare(train_frac=TRAIN_FRAC, cache=CACHE):
    """One call -> everything downstream needs. Prints the leak boundary."""
    df = load_enso(cache)
    month = df.month.to_numpy()
    sst = df.sst.to_numpy(dtype=float)
    n_train = train_month_cutoff(len(df), train_frac=train_frac)
    y, clim = anomaly(month, sst, n_train)
    scaler, (lo, hi) = fit_scaler(y, n_train)
    print(f"train span: months [0,{n_train}) = "
          f"{df.date.iloc[0]:%Y-%m}..{df.date.iloc[n_train - 1]:%Y-%m}; "
          f"first test target {df.date.iloc[n_train]:%Y-%m}")
    print(f"scaler: train anomaly [{lo:+.3f},{hi:+.3f}] C -> [0,1], clip outside")
    return dict(df=df, dates=df.date.to_numpy(), month=month, sst=sst,
                y=y, clim=clim, scaler=scaler, scaler_range=(lo, hi),
                n_train_months=n_train)


if __name__ == "__main__":
    print_config()
    d = prepare()
    print(f"anomaly: mean={d['y'].mean():+.4f} std={d['y'].std():.4f} "
          f"min={d['y'].min():+.3f} max={d['y'].max():+.3f}")
