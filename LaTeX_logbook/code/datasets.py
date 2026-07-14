"""datasets.py -- Data sources for the worked examples.

Two physically motivated *synthetic surrogates* (clearly labelled as such in
the accompanying document) and one *real* climate series:

  * solar_surrogate()  -- 30-min GHI for one year at a low-latitude site,
    built as (Haurwitz clear-sky model) x (two-state Markov cloud regime with
    beta-distributed clear-sky index), with sensor-outage gaps injected.
  * load_surrogate()   -- hourly system load for three years: weekly/diurnal
    profiles x annual seasonality + temperature response (CDD/HDD) + AR(1)
    residual + holiday suppression.
  * enso_real()        -- REAL monthly Nino-region sea-surface temperature,
    1950--2010, bundled with statsmodels (NOAA/NCEP source).

The surrogates reproduce the statistical signatures that matter for reservoir
design (diurnal/weekly cycles, regime switching, conditional heteroscedastic
noise, autocorrelation structure) so that the entire analysis methodology
transfers unchanged to the real NSRDB / ENTSO-E / UCI files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Solar surrogate (NSRDB-style 30-min GHI, one year, latitude 1.35 N)
# ----------------------------------------------------------------------
def _solar_elevation(doy, hour_utc8, lat_deg=1.35):
    """Solar elevation angle (rad) from day-of-year and local solar hour."""
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    hra = np.deg2rad(15.0 * (hour_utc8 - 12.0))
    lat = np.deg2rad(lat_deg)
    sin_alpha = (np.sin(lat) * np.sin(decl)
                 + np.cos(lat) * np.cos(decl) * np.cos(hra))
    return np.arcsin(np.clip(sin_alpha, -1.0, 1.0))


def solar_surrogate(seed=42, gap_frac=0.02):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=2 * 24 * 365, freq="30min")
    doy = idx.dayofyear.to_numpy()
    hod = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    alpha = _solar_elevation(doy, hod)
    sin_a = np.clip(np.sin(alpha), 0.0, None)
    # Haurwitz clear-sky model (W m^-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ghi_cs = np.where(sin_a > 0, 1098.0 * sin_a * np.exp(-0.057 / np.maximum(sin_a, 1e-6)), 0.0)
    # Two-state (clear/cloudy) Markov regime for the clear-sky index k_t,
    # with a mild monsoon seasonality in the cloudy-state probability.
    p_cloud = 0.45 + 0.15 * np.sin(2 * np.pi * (doy - 320) / 365.0)
    state = np.zeros(len(idx), dtype=int)
    for t in range(1, len(idx)):
        stay = 0.965 if state[t - 1] == 1 else 0.985
        if rng.random() < stay:
            state[t] = state[t - 1]
        else:
            state[t] = int(rng.random() < p_cloud[t])
    kt = np.where(state == 1,
                  rng.beta(2.2, 3.5, size=len(idx)) * 0.75,          # cloudy
                  np.minimum(rng.beta(9.0, 1.6, size=len(idx)) * 1.05, 1.05))  # clear
    # Smooth k_t slightly (30-min persistence of cloud fields)
    kt = pd.Series(kt).rolling(3, min_periods=1, center=True).mean().to_numpy()
    ghi = ghi_cs * kt
    df = pd.DataFrame({"ghi": ghi, "ghi_clear": ghi_cs, "kt": kt,
                       "elevation_deg": np.rad2deg(alpha)}, index=idx)
    # Inject sensor-outage gaps (contiguous blocks) to exercise gap analysis.
    n_gaps = int(gap_frac * len(df) / 8)
    for _ in range(n_gaps):
        start = rng.integers(0, len(df) - 12)
        df.iloc[start:start + rng.integers(2, 12),
                df.columns.get_loc("ghi")] = np.nan
    return df


# ----------------------------------------------------------------------
# Load surrogate (hourly system load, three years)
# ----------------------------------------------------------------------
def load_surrogate(seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=24 * 365 * 3, freq="h")
    hod, dow, doy = idx.hour.to_numpy(), idx.dayofweek.to_numpy(), idx.dayofyear.to_numpy()
    diurnal = (0.72 + 0.16 * np.exp(-0.5 * ((hod - 11.0) / 3.2) ** 2)
               + 0.20 * np.exp(-0.5 * ((hod - 19.5) / 2.4) ** 2))
    weekend = np.where(dow >= 5, 0.86, 1.0)
    annual = 1.0 + 0.06 * np.cos(2 * np.pi * (doy - 200) / 365.0)
    # Ambient temperature (deg C) with diurnal + annual cycles and AR noise.
    temp = (27.5 + 4.5 * np.cos(2 * np.pi * (doy - 120) / 365.0) * 0.35
            + 3.0 * np.sin(2 * np.pi * (hod - 9) / 24.0))
    eps = np.zeros(len(idx))
    for t in range(1, len(idx)):
        eps[t] = 0.97 * eps[t - 1] + rng.normal(0, 0.35)
    temp = temp + eps
    cdd = np.clip(temp - 24.0, 0, None)            # cooling degree signal
    hdd = np.clip(16.0 - temp, 0, None)            # (rarely active here)
    base = 5200.0
    resid = np.zeros(len(idx))
    for t in range(1, len(idx)):
        resid[t] = 0.9 * resid[t - 1] + rng.normal(0, 28.0)
    load = base * diurnal * weekend * annual + 55.0 * cdd + 30.0 * hdd + resid
    # Public-holiday suppression on ~11 scattered days per year.
    for year in (2023, 2024, 2025):
        days = rng.choice(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"),
                          size=11, replace=False)
        for d in days:
            mask = (idx.normalize() == pd.Timestamp(d).normalize())
            load[mask] *= 0.88
    return pd.DataFrame({"load_mw": load, "temp_c": temp,
                         "cdd": cdd, "hdd": hdd}, index=idx)


# ----------------------------------------------------------------------
# ENSO -- real data (statsmodels bundled copy of NOAA monthly Nino SST)
# ----------------------------------------------------------------------
def enso_real():
    from statsmodels.datasets import elnino
    wide = elnino.load_pandas().data          # columns: YEAR, JAN..DEC
    long = wide.melt(id_vars="YEAR", var_name="month", value_name="sst")
    month_order = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    long["m"] = long["month"].map({m: i + 1 for i, m in enumerate(month_order)})
    long = long.sort_values(["YEAR", "m"])
    idx = pd.to_datetime({"year": long["YEAR"].astype(int),
                          "month": long["m"], "day": 1})
    s = pd.Series(long["sst"].to_numpy(), index=idx, name="sst_c")
    # Anomaly relative to the monthly climatology (standard ENSO practice).
    clim = s.groupby(s.index.month).transform("mean")
    return pd.DataFrame({"sst_c": s, "anomaly_c": s - clim})


if __name__ == "__main__":
    sol, ld, en = solar_surrogate(), load_surrogate(), enso_real()
    print("solar", sol.shape, "NaN:", int(sol['ghi'].isna().sum()))
    print("load ", ld.shape)
    print("enso ", en.shape, en.index.min().date(), "->", en.index.max().date())
