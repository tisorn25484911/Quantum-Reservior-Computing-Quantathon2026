"""fetch_cp_data.py -- CP aquaculture use-case data, reproducibly.

Four records chosen to ground the Forecast-then-Detect line in the CP
(Charoen Pokphand) shrimp-aquaculture use case: SST anomalies and salinity
shocks (rain-driven) are the physical drivers of pond losses in the upper
Gulf of Thailand, and ENSO modulates both.

  real/oisst_gulf_thailand_daily_sst.csv
      NOAA OISST v2.1, daily, 0.25 deg, grid point 13.125N 100.125E --
      the upper Gulf of Thailand directly off the Mae Klong estuary
      (Samut Songkhram). 1981-09 -> present. The CP-waters counterpart
      to `tao_daily_sst.csv`: same variable, same cadence, company-relevant
      location.

  real/ersst_gulf_thailand_monthly_sst.csv
      NOAA ERSSTv5 monthly *reconstruction*, 2 deg, grid point 12N 100E
      (central Gulf of Thailand), 1854 -> present. ~170 years of context:
      how unusual is a modern warm event against the full record.

  real/oni_index.csv
      Oceanic Nino Index (CPC official): 3-month running-mean Nino-3.4
      anomaly, the operational El Nino / La Nina definition (+/-0.5 degC).
      Complements `nino34_anom.csv` (monthly, PSL) -- ONI is the index the
      industry actually keys decisions off.

  real/nasa_power_mae_klong_daily_rain.csv
      NASA POWER (MERRA-2 corrected) daily precipitation at 13.75N 99.75E,
      the lower Mae Klong basin (Ratchaburi / Samut Songkhram), 1981 ->
      present. Rain drives salinity swings in estuarine shrimp ponds.

All sources are open, no credentials. Cached like fetch_data.py: existing
outputs are never re-fetched unless --force.

Usage
-----
    python fetch_cp_data.py           # fetch whatever is missing
    python fetch_cp_data.py --force   # re-download everything
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_data import _http_get, _say

HERE = Path(__file__).resolve().parent
REAL = HERE / "real"

# CoastWatch ERDDAP. Verified live 2026-07-23; both points return valid SST
# (13.125N 100.125E is ocean despite being ~20 km off the Mae Klong mouth).
# The OISST aggregate stores one NetCDF file per day, so a point time-series
# over the whole 45-year record makes the server open ~16k files and it times
# out. It serves ~1 year/minute reliably, so fetch year by year and cache each
# chunk under real/_oisst_chunks/. {y0},{y1} are calendar-year bounds.
OISST_CHUNK_URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv"
    "?sst%5B({y0}-01-01T12:00:00Z):({y1}-12-31T12:00:00Z)%5D"
    "%5B(0.0)%5D%5B(13.125)%5D%5B(100.125)%5D"
)
OISST_START_YEAR = 1982   # first full year (record starts 1981-09)
ERSST_URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nceiErsstv5.csv"
    "?sst%5B(1854-01-01):(last)%5D%5B(0.0)%5D%5B(12)%5D%5B(100)%5D"
)
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=PRECTOTCORR&community=AG"
    "&longitude=99.75&latitude=13.75&start={start}&end={end}&format=CSV"
)

# ONI season label -> centre month of the 3-month window
_ONI_SEASONS = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def _erddap_frame(text):
    df = pd.read_csv(io.StringIO(text), skiprows=[1])
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    return df[["date", "sst"]]


def _oisst_daily(out):
    """Fetch the daily SST point year by year, caching each chunk."""
    chunks = out.parent / "_oisst_chunks"
    chunks.mkdir(exist_ok=True)
    end_year = pd.Timestamp.today().year
    frames = []
    for y in range(OISST_START_YEAR, end_year + 1):
        cf = chunks / f"{y}.csv"
        if cf.exists():
            frames.append(pd.read_csv(cf, parse_dates=["date"]))
            continue
        _say(f"  OISST {y} ...")
        try:
            text = _http_get(OISST_CHUNK_URL.format(y0=y, y1=y), timeout=150)
        except RuntimeError as exc:
            _say(f"    {y} failed ({exc}); stopping, rerun to resume")
            break
        if text.lstrip().startswith("Error"):
            _say(f"    {y} not yet available; stopping")
            break
        fr = _erddap_frame(text)
        fr.to_csv(cf, index=False)
        frames.append(fr)
    if not frames:
        raise RuntimeError("no OISST chunks fetched")
    df = pd.concat(frames).drop_duplicates("date").set_index("date").sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df.to_csv(out, index_label="date")
    _say(f"  wrote {out.name}: {len(df)} days, "
         f"{df.index.min():%Y-%m} to {df.index.max():%Y-%m}, "
         f"{df.sst.isna().mean():.2%} missing")


def _ersst_monthly(out):
    _say("  ERSSTv5 Gulf of Thailand monthly ...")
    text = _http_get(ERSST_URL, timeout=300)
    if text.lstrip().startswith("Error"):
        raise RuntimeError(f"ERDDAP error: {text[:200]}")
    df = _erddap_frame(text)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.set_index("date").sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="MS"))
    df.to_csv(out, index_label="date")
    _say(f"  wrote {out.name}: {len(df)} months, "
         f"{df.index.min():%Y-%m} to {df.index.max():%Y-%m}, "
         f"{df.sst.isna().mean():.2%} missing")


def _oni(out):
    _say("  ONI ...")
    text = _http_get(ONI_URL, timeout=60)
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in _ONI_SEASONS:
            seas, year = parts[0], int(parts[1])
            month = _ONI_SEASONS[seas]
            # DJF labelled with the January year already -- no shift needed
            rows.append((pd.Timestamp(year=year, month=month, day=1),
                         float(parts[2]), float(parts[3])))
    if not rows:
        raise RuntimeError("ONI parse produced no rows")
    df = pd.DataFrame(rows, columns=["date", "total", "oni"]).set_index("date")
    df = df.sort_index()
    df.to_csv(out, index_label="date")
    _say(f"  wrote {out.name}: {len(df)} months, "
         f"{df.index.min():%Y-%m} to {df.index.max():%Y-%m}")


def _mae_klong_rain(out):
    _say("  NASA POWER Mae Klong rain (1981 -> present, one request) ...")
    end = pd.Timestamp.today().strftime("%Y%m%d")
    text = _http_get(POWER_URL.format(start="19810101", end=end), timeout=600)
    body = text.split("-END HEADER-")[-1].strip()
    df = pd.read_csv(io.StringIO(body))
    df["date"] = (pd.to_datetime(df["YEAR"], format="%Y")
                  + pd.to_timedelta(df["DOY"] - 1, unit="D"))
    df = df.rename(columns={"PRECTOTCORR": "rain_mm"})[["date", "rain_mm"]]
    df.loc[df.rain_mm <= -998, "rain_mm"] = np.nan   # -999 missing code
    df = df.set_index("date").sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df.to_csv(out, index_label="date")
    _say(f"  wrote {out.name}: {len(df)} days, "
         f"{df.index.min():%Y-%m} to {df.index.max():%Y-%m}, "
         f"{df.rain_mm.isna().mean():.2%} missing")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true",
                   help="re-download everything")
    a = p.parse_args()
    REAL.mkdir(parents=True, exist_ok=True)

    jobs = [
        (REAL / "oisst_gulf_thailand_daily_sst.csv", _oisst_daily),
        (REAL / "ersst_gulf_thailand_monthly_sst.csv", _ersst_monthly),
        (REAL / "oni_index.csv", _oni),
        (REAL / "nasa_power_mae_klong_daily_rain.csv", _mae_klong_rain),
    ]
    for out, fn in jobs:
        if out.exists() and not a.force:
            _say(f"  {out.name} cached")
            continue
        fn(out)
    _say("done")


if __name__ == "__main__":
    main()
