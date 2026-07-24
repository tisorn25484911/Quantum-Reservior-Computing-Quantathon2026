"""label_anomalies.py -- turn the CP driver series into LABELLED anomaly data.

The two-step plan (see plan.md / TODO.md) needs a Step-2 anomaly detector that
learns from labelled events. Rather than hunt for a generic labelled benchmark,
we label the anomaly the product is actually about, using peer-reviewed
domain standards applied to data already in the repo:

  1. MARINE HEATWAVES on `got_sst` -- Hobday et al. (2016), "A hierarchical
     approach to defining marine heatwaves", Prog. Oceanogr. 141, 227-238.
     An MHW is a period of >=5 consecutive days with SST above a seasonally
     varying 90th-percentile threshold; gaps <=2 days join two events.
     Category (Hobday et al. 2018) = how many multiples of (threshold minus
     climatology-mean) the temperature reaches: 1 Moderate, 2 Strong,
     3 Severe, 4+ Extreme.

  2. ENSO EPISODES from `oni` -- NOAA CPC operational definition: El Nino /
     La Nina when the Oceanic Nino Index is >= +0.5 / <= -0.5 for at least 5
     consecutive overlapping seasons (i.e. 5 consecutive monthly ONI values,
     since ONI is already a 3-month running mean).

Outputs (under Anomaly_Forecast/labels/):
  got_sst_mhw_labeled.csv   date, sst, clim, thresh, intensity, mhw, event_id, category
  got_sst_mhw_events.json   one record per event (start,end,duration,imax,imean,cum,category)
  oni_enso_episodes.csv     start, end, months, kind (el_nino/la_nina), peak
  mhw_labels.png            SST with climatology, threshold, and shaded events

Pure numpy/pandas; no new dependencies, no credentials. The MHW climatology is
computed on a fixed 30-year baseline (1982-2011, the record's first full 30
years) as Hobday recommends, so the threshold is a stable reference rather than
something that shifts with the sample -- and, usefully for Step 2, it is fixed
independently of any later train/val/test split of the detector.

    python label_anomalies.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "DataBase_Analysis"))
from dataloader import _TABULAR, DATA                          # noqa: E402

LABELS = _HERE / "labels"
BASELINE = (1982, 2011)          # Hobday-style fixed 30-yr climatology period
WINDOW_HALF = 5                  # +/-5 days -> 11-day sampling window
SMOOTH = 31                      # running-mean smoothing of clim & threshold
MIN_DURATION = 5                 # days
MAX_GAP = 2                      # days; shorter gaps join two events
PCTILE = 90


# ----------------------------------------------------------------------
# Marine heatwaves (Hobday 2016)
# ----------------------------------------------------------------------
def _doy_no_leap(dates: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-year in 1..365, folding Feb-29 onto 59 so the climatology has a
    fixed 365-length support (the standard MHW handling of leap days)."""
    doy = np.asarray(dates.dayofyear).copy()
    is_leap = np.asarray(dates.is_leap_year)
    after_feb29 = is_leap & (doy > 59)
    doy[after_feb29] -= 1
    return np.clip(doy, 1, 365)


def _circular_runmean(a: np.ndarray, w: int) -> np.ndarray:
    pad = w // 2
    ext = np.concatenate([a[-pad:], a, a[:pad]])
    ker = np.ones(w) / w
    return np.convolve(ext, ker, mode="same")[pad:pad + len(a)]


def marine_heatwave_labels(dates, sst):
    """Return a per-day frame with climatology, threshold, intensity and event
    labels, plus a list of event records."""
    dates = pd.DatetimeIndex(dates)
    sst = np.asarray(sst, float)
    doy = _doy_no_leap(dates)
    years = dates.year.to_numpy()

    base = (years >= BASELINE[0]) & (years <= BASELINE[1]) & np.isfinite(sst)
    clim = np.full(366, np.nan)
    thr = np.full(366, np.nan)
    for d in range(1, 366):
        # 11-day window around doy d, circular over 1..365
        offs = ((np.arange(d - WINDOW_HALF, d + WINDOW_HALF + 1) - 1) % 365) + 1
        sel = base & np.isin(doy, offs)
        vals = sst[sel]
        if len(vals):
            clim[d] = np.mean(vals)
            thr[d] = np.percentile(vals, PCTILE)
    clim[1:] = _circular_runmean(clim[1:], SMOOTH)      # smooth doy 1..365
    thr[1:] = _circular_runmean(thr[1:], SMOOTH)

    clim_day = clim[doy]
    thr_day = thr[doy]
    intensity = sst - clim_day                          # anomaly above mean
    exceed = np.isfinite(sst) & (sst > thr_day)

    # runs of exceedance, joining gaps <= MAX_GAP
    event_id = np.zeros(len(sst), dtype=int)
    events = []
    i, n, eid = 0, len(sst), 0
    while i < n:
        if not exceed[i]:
            i += 1
            continue
        j = i
        while j + 1 < n:
            if exceed[j + 1]:
                j += 1
            else:  # look across a short gap
                k = j + 1
                while k < n and k - j <= MAX_GAP and not exceed[k]:
                    k += 1
                if k < n and k - j <= MAX_GAP and exceed[k]:
                    j = k
                else:
                    break
        if j - i + 1 >= MIN_DURATION:
            eid += 1
            sl = slice(i, j + 1)
            event_id[sl] = eid
            inten = intensity[sl]
            diff = (thr_day[sl] - clim_day[sl])
            diff[diff <= 0] = np.nan
            cat = int(np.nanmax(np.floor(inten / diff))) if np.isfinite(diff).any() else 1
            cat = max(1, min(cat, 4))
            events.append(dict(
                event_id=eid,
                start=str(dates[i].date()), end=str(dates[j].date()),
                duration_days=int(j - i + 1),
                i_max=float(np.nanmax(inten)), i_mean=float(np.nanmean(inten)),
                i_cumulative=float(np.nansum(inten)),
                category=["Moderate", "Strong", "Severe", "Extreme"][cat - 1],
            ))
        i = j + 1

    # per-day category
    cat_day = np.zeros(len(sst), dtype=int)
    for ev in events:
        m = event_id == ev["event_id"]
        cat_day[m] = ["Moderate", "Strong", "Severe", "Extreme"].index(ev["category"]) + 1

    frame = pd.DataFrame(dict(
        date=dates, sst=sst, clim=clim_day, thresh=thr_day,
        intensity=intensity, mhw=exceed.astype(int) * (event_id > 0).astype(int),
        event_id=event_id, category=cat_day,
    ))
    # a day only counts as MHW if it belongs to a qualifying event
    frame["mhw"] = (event_id > 0).astype(int)
    return frame, events


# ----------------------------------------------------------------------
# ENSO episodes from ONI
# ----------------------------------------------------------------------
def enso_episodes(dates, oni, thr=0.5, min_len=5):
    dates = pd.DatetimeIndex(dates)
    oni = np.asarray(oni, float)
    out = []
    for kind, mask in (("el_nino", oni >= thr), ("la_nina", oni <= -thr)):
        i, n = 0, len(oni)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                seg = oni[i:j + 1]
                peak = float(seg.max() if kind == "el_nino" else seg.min())
                out.append(dict(kind=kind, start=str(dates[i].date()),
                                end=str(dates[j].date()),
                                months=int(j - i + 1), peak=peak))
            i = j + 1
    out.sort(key=lambda e: e["start"])
    return out


# ----------------------------------------------------------------------
def _load_csv(key):
    spec = _TABULAR[key]
    df = pd.read_csv(DATA / spec["path"], parse_dates=[spec["index"]])
    return df[spec["index"]], df[spec["column"]].to_numpy(float)


def main():
    LABELS.mkdir(exist_ok=True)

    # -- marine heatwaves on got_sst --
    d, sst = _load_csv("got_sst")
    sst = pd.Series(sst).interpolate(limit_direction="both").to_numpy()
    frame, events = marine_heatwave_labels(d, sst)
    frame.to_csv(LABELS / "got_sst_mhw_labeled.csv", index=False)
    (LABELS / "got_sst_mhw_events.json").write_text(json.dumps(events, indent=2))

    mhw_days = int(frame["mhw"].sum())
    print(f"MARINE HEATWAVES on got_sst ({frame['date'].iloc[0].date()} -> "
          f"{frame['date'].iloc[-1].date()}):")
    print(f"  {len(events)} events, {mhw_days} MHW days "
          f"({mhw_days/len(frame):.1%} of the record)")
    by_cat = pd.Series([e["category"] for e in events]).value_counts()
    print("  by category:", dict(by_cat))
    longest = max(events, key=lambda e: e["duration_days"])
    hottest = max(events, key=lambda e: e["i_max"])
    print(f"  longest : {longest['duration_days']} d  "
          f"{longest['start']}->{longest['end']}  ({longest['category']})")
    print(f"  hottest : +{hottest['i_max']:.2f} degC  {hottest['start']}  "
          f"({hottest['category']})")

    # -- ENSO episodes from ONI --
    do, oni = _load_csv("oni")
    eps = enso_episodes(do, oni)
    pd.DataFrame(eps).to_csv(LABELS / "oni_enso_episodes.csv", index=False)
    el = [e for e in eps if e["kind"] == "el_nino"]
    la = [e for e in eps if e["kind"] == "la_nina"]
    print(f"\nENSO EPISODES from ONI: {len(el)} El Nino, {len(la)} La Nina")
    strongest = max(el, key=lambda e: e["peak"])
    print(f"  strongest El Nino: peak ONI {strongest['peak']:.1f}  "
          f"{strongest['start']}->{strongest['end']}")

    # -- figure --
    _plot(frame, events)
    print(f"\nwrote labels/ (2 CSV, 1 JSON, mhw_labels.png)")


def _plot(frame, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # show the last ~8 years for legibility
    f = frame[frame["date"] >= frame["date"].max() - pd.Timedelta(days=8*365)]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(f["date"], f["sst"], lw=.7, color="#1b3a4b", label="SST")
    ax.plot(f["date"], f["clim"], lw=1, color="#6f8a8c", label="climatology")
    ax.plot(f["date"], f["thresh"], lw=1, ls="--", color="#d97a2b",
            label="90th-pct threshold")
    ax.fill_between(f["date"], f["sst"], f["thresh"],
                    where=f["mhw"] == 1, color="#e0483a", alpha=.6,
                    interpolate=True, label="marine heatwave")
    ax.set_ylabel("SST (degC)")
    ax.set_title("Gulf of Thailand SST -- marine-heatwave labels "
                 "(Hobday 2016), last 8 years")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(LABELS / "mhw_labels.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
