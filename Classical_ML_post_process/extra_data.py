"""Loaders and label builders for the demonstration datasets.

Two labelled anomaly sources are wired up here so the EGADS anomalous-series
detector can be evaluated against independent ground truth:

1. ONI ENSO episodes (``data/oni_enso_episodes.csv``)
   NOAA/CPC Oceanic Nino Index episodes -- the official start/end months of
   every El Nino (warm) and La Nina (cold) event. These give a *monthly*
   boolean label aligned to the SST/SOI/PDO climate channels loaded by
   ``retrieve_data``, so a compound-ENSO stretch is the "anomaly" to detect.

2. Marine heat waves (``data/got_sst_mhw_labeled.csv`` + ``..._events.json``)
   A *daily* SST record with the Hobday-et-al. MHW flag per day (sst above the
   seasonally-varying 90th-percentile threshold for >= 5 days) plus a catalogue
   of the detected events. This gives a daily multivariate detection problem.

Nothing here is invented labelling: the boolean targets come straight from the
published episode / event tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import retrieve_data as rd

DATA = Path(__file__).resolve().parent / "data"


# ------------------------------------------------------------------- ONI ENSO
def load_oni_episodes(path=DATA / "oni_enso_episodes.csv"):
    """ENSO episode table -> DataFrame[kind, start, end, months, peak]."""
    df = pd.read_csv(path, parse_dates=["start", "end"])
    return df


def oni_monthly_labels(dates, kinds=("el_nino", "la_nina"),
                       path=DATA / "oni_enso_episodes.csv"):
    """Boolean label aligned to `dates`: True where an ENSO episode is active.

    Parameters
    ----------
    dates : array-like of datetime64 (monthly, first-of-month)
    kinds : subset of {"el_nino", "la_nina"} to count as an event.

    Returns
    -------
    ndarray(bool) of len(dates).
    """
    ep = load_oni_episodes(path)
    ep = ep[ep["kind"].isin(kinds)]
    dts = pd.to_datetime(pd.Series(np.asarray(dates)))
    # month index (year*12+month) for robust inclusive interval tests
    mi = dts.dt.year * 12 + dts.dt.month
    label = np.zeros(len(dts), dtype=bool)
    for _, row in ep.iterrows():
        s = row["start"].year * 12 + row["start"].month
        e = row["end"].year * 12 + row["end"].month
        label |= ((mi >= s) & (mi <= e)).to_numpy()
    return label


def build_enso_labeled(names=("sst", "soi", "pdo"), use_scaled=False,
                       train_frac=rd.TRAIN_FRAC, verbose=True):
    """Aligned multivariate ENSO channels + ONI/compound labels.

    Reuses ``retrieve_data.prepare_compound`` to get the leak-free, common-span
    anomaly matrix for the requested channels, then attaches the independent
    ONI episode labels on the same monthly grid.

    Returns a dict with:
        dates          (T,) datetime64
        channel_names  list[str]
        X              (T, C) matrix of channel values (anomalies, or the
                       [0,1]-scaled encoding if use_scaled=True)
        label_warm     (T,) bool  El Nino episodes active
        label_cold     (T,) bool  La Nina episodes active
        label_enso     (T,) bool  either episode active
        label_compound (T,) bool  the co-exceedance compound label from
                       retrieve_data (SST/SOI/PDO jointly in their alarming tail)
        n_train        int chronological train cutoff (index)
    """
    comp = rd.prepare_compound(names=names, train_frac=train_frac,
                               verbose=verbose)
    dates = comp["dates"]
    X = comp["U"] if use_scaled else comp["Y"]
    out = dict(
        dates=dates,
        channel_names=list(names),
        X=X,
        label_warm=oni_monthly_labels(dates, kinds=("el_nino",)),
        label_cold=oni_monthly_labels(dates, kinds=("la_nina",)),
        label_enso=oni_monthly_labels(dates, kinds=("el_nino", "la_nina")),
        label_compound=comp["compound"],
        n_train=comp["n_train"],
    )
    if verbose:
        print(f"[enso-labeled] {len(dates)} months, channels {list(names)}; "
              f"ONI warm={out['label_warm'].sum()} cold={out['label_cold'].sum()} "
              f"any={out['label_enso'].sum()}; "
              f"compound={out['label_compound'].sum()}")
    return out


# ---------------------------------------------------------------- marine heat
def load_mhw_labeled(path=DATA / "got_sst_mhw_labeled.csv"):
    """Daily SST + MHW flag table -> DataFrame.

    Columns: date, sst, clim, thresh, intensity, mhw (0/1), event_id, category.
    """
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def load_mhw_events(path=DATA / "got_sst_mhw_events.json"):
    """MHW event catalogue -> DataFrame[event_id, start, end, ... , category]."""
    with open(path) as fh:
        events = json.load(fh)
    df = pd.DataFrame(events)
    for col in ("start", "end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def build_mhw_labeled(channels=("anomaly", "sst"), verbose=True):
    """Daily marine-heat-wave detection problem as a multivariate array.

    Parameters
    ----------
    channels : subset/order of {"sst", "anomaly", "intensity", "thresh_gap"}
        "anomaly"    = sst - clim (deseasonalised SST)
        "thresh_gap" = sst - thresh (how far above the MHW threshold)

    Returns dict with dates, channel_names, X (T, C), label (bool = daily MHW
    flag), events (catalogue DataFrame).
    """
    df = load_mhw_labeled()
    cols = {
        "sst": df["sst"].to_numpy(float),
        "anomaly": (df["sst"] - df["clim"]).to_numpy(float),
        "intensity": df["intensity"].to_numpy(float),
        "thresh_gap": (df["sst"] - df["thresh"]).to_numpy(float),
    }
    X = np.column_stack([cols[c] for c in channels])
    out = dict(
        dates=df["date"].to_numpy(),
        channel_names=list(channels),
        X=X,
        label=df["mhw"].to_numpy().astype(bool),
        events=load_mhw_events(),
    )
    if verbose:
        print(f"[mhw-labeled] {len(df)} days "
              f"{df['date'].iloc[0]:%Y-%m-%d}..{df['date'].iloc[-1]:%Y-%m-%d}; "
              f"channels {list(channels)}; MHW days={out['label'].sum()} "
              f"({out['label'].mean():.3f}); events={len(out['events'])}")
    return out
