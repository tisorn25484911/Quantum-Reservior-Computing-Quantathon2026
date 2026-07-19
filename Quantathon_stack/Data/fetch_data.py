"""fetch_data.py -- build the Quantathon dataset directory, reproducibly.

Three tiers, written to sibling directories of this file:

  real/       observed records, downloaded from a public URL
  surrogate/  physically motivated synthetic series, generated from
              QRC_code_stack/stage1_numpy_core/datasets.py
  chaotic/    chaotic systems with a PUBLISHED largest Lyapunov exponent and
              multiple realizations from perturbed initial conditions

The chaotic tier is the one that matters for validating a Lyapunov estimator:
it is the only tier where ground truth exists, and the only tier with more than
one realization per system. Each .npz there carries

    x      (R, N) float   R realizations of a scalar observable
    dt     scalar         sample spacing, natural time units of the system
    lyap   scalar         published largest Lyapunov exponent
    ic_eps scalar         size of the initial-condition perturbation

Usage
-----
    python fetch_data.py                # everything cached/quick
    python fetch_data.py --dysts        # also integrate the dysts systems
    python fetch_data.py --force        # re-download and regenerate

The --dysts tier needs `pip install dysts numba` and is slow (minutes per
system, and the 4-D systems are much worse), which is why it is opt-in and
cached: an existing output file is never re-integrated unless --force is given.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGE1 = REPO / "QRC_code_stack" / "stage1_numpy_core"

REAL, SURR, CHAOS = HERE / "real", HERE / "surrogate", HERE / "chaotic"

NINO34_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"

# TAO/TRITON equatorial moorings. ERDDAP wants longitude in 0-360; the familiar
# names are the western-hemisphere ones, so both are kept.
TAO_SITES = [("0N165E", 0, 165), ("0N170W", 0, 190), ("0N155W", 0, 205),
             ("0N140W", 0, 220), ("0N110W", 0, 250)]
TAO_URL = ("https://data.pmel.noaa.gov/generic/erddap/tabledap/pmelTaoDySst.csv"
           "?time,latitude,longitude,T_25&latitude={lat}&longitude={lon}")

GHCND_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station}.dly"
GHCND_STATIONS = [
    ("USW00094728", "new_york_central_park"),
    ("USW00023174", "los_angeles_airport"),
]

USGS_DV = ("https://waterservices.usgs.gov/nwis/dv/?sites={site}"
           "&parameterCd=00060&startDT=1900-01-01&endDT={end}&format=rdb")
USGS_IV = ("https://waterservices.usgs.gov/nwis/iv/?sites={site}"
           "&parameterCd=00060&startDT={start}&endDT={end}&format=rdb")
USGS_SITE = "01646500"        # Potomac River at Little Falls Pump Station, MD

PSMSL_URL = "https://psmsl.org/data/obtaining/rlr.monthly.data/{sid}.rlrdata"
PSMSL_STATIONS = [("1", "brest"), ("10", "cuxhaven")]

OPSD_URL = ("https://data.open-power-system-data.org/time_series/latest/"
            "time_series_60min_singleindex.csv")


def _say(msg):
    print(msg, flush=True)


def _http_get(url, timeout=60):
    """Fetch a URL as text, falling back to curl.

    The python.org macOS builds ship without a system CA bundle wired up, so
    urllib raises CERTIFICATE_VERIFY_FAILED where curl (which uses the system
    trust store) succeeds. Try certifi, then curl, before giving up.
    """
    try:
        import ssl

        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=timeout, context=ctx) as fh:
            return fh.read().decode()
    except ImportError:
        pass
    except urllib.error.URLError:
        pass
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            return fh.read().decode()
    except urllib.error.URLError as exc:
        _say(f"  urllib failed ({exc.reason}); retrying with curl")

    import subprocess
    res = subprocess.run(["curl", "-sSL", "--max-time", str(timeout), url],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"download failed for {url}: {res.stderr.strip()}")
    return res.stdout


# ----------------------------------------------------------------------
# real
# ----------------------------------------------------------------------
def build_real(force=False):
    REAL.mkdir(parents=True, exist_ok=True)

    out = REAL / "nino34_anom.csv"
    if force or not out.exists():
        _say(f"downloading {NINO34_URL}")
        text = _http_get(NINO34_URL)
        # PSL fixed-width: header line of (start_year, end_year), then one row
        # per year of 12 monthly values, then a trailer whose first token is the
        # missing-value code. Guard on the 13-token shape rather than line count.
        rows = []
        for line in text.splitlines()[1:]:
            parts = line.split()
            if len(parts) == 13 and parts[0].isdigit():
                for month, val in enumerate([float(v) for v in parts[1:]], 1):
                    rows.append((int(parts[0]), month, val))
        df = pd.DataFrame(rows, columns=["year", "month", "anom"])
        df = df[df.anom > -90].reset_index(drop=True)
        if df.empty:
            raise RuntimeError("Nino3.4 parse produced no rows -- did the URL "
                               "return an HTML error page?")
        df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
        df[["date", "year", "month", "anom"]].to_csv(out, index=False)
        _say(f"  wrote {out.name}: {len(df)} months, "
             f"{df.date.min():%Y-%m} to {df.date.max():%Y-%m}")

    # the record the existing pipeline was built on, copied so the Quantathon
    # stack does not reach across into QRC_code_stack at runtime
    src = STAGE1 / "data" / "raw" / "enso.csv"
    dst = REAL / "enso_nino12_1950_2010.csv"
    if src.exists() and (force or not dst.exists()):
        dst.write_bytes(src.read_bytes())
        _say(f"  copied {dst.name} (bundled Nino-1+2-type record)")

    _tao(force)
    _ghcnd(force)
    _usgs(force)
    _psmsl(force)
    _opsd(force)


# ---------------------------------------------------------------- TAO/TRITON
def _tao(force=False):
    """Daily SST from equatorial Pacific moorings, one column per site.

    Wide format on a common daily index: the five moorings observe the same
    ENSO process at different longitudes, so the columns are as close to
    'multiple realizations' as real observations get. They are NOT independent
    realizations -- the sites are dynamically coupled along the equator -- so
    treat inter-site spread as a lower bound on ensemble spread.
    """
    out = REAL / "tao_daily_sst.csv"
    if out.exists() and not force:
        return
    frames = []
    for name, lat, lon in TAO_SITES:
        _say(f"  TAO {name} ...")
        text = _http_get(TAO_URL.format(lat=lat, lon=lon), timeout=120)
        df = pd.read_csv(io.StringIO(text), skiprows=[1])
        df = df[["time", "T_25"]].rename(columns={"T_25": name})
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
        frames.append(df.set_index("time"))
    wide = pd.concat(frames, axis=1).sort_index()
    # reindex onto a gapless daily axis so the missing spans are explicit NaNs
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    wide.to_csv(out, index_label="date")
    _say(f"  wrote {out.name}: {wide.shape[0]} days x {wide.shape[1]} moorings, "
         f"{wide.index.min():%Y-%m} to {wide.index.max():%Y-%m}")


# ---------------------------------------------------------------- GHCN-Daily
def _parse_dly(text, elements=("TMAX", "TMIN", "PRCP")):
    """GHCN-Daily .dly fixed-width -> tidy frame.

    Layout per line: ID(11) YEAR(4) MONTH(2) ELEMENT(4) then 31 groups of
    VALUE(5) MFLAG(1) QFLAG(1) SFLAG(1). -9999 is missing. TMAX/TMIN are in
    tenths of degC and PRCP in tenths of mm, so everything is scaled by 10.
    """
    recs = {}
    for line in text.splitlines():
        if len(line) < 269:
            continue
        elem = line[17:21]
        if elem not in elements:
            continue
        year, month = int(line[11:15]), int(line[15:17])
        for d in range(31):
            raw = line[21 + d * 8: 26 + d * 8]
            try:
                val = int(raw)
            except ValueError:
                continue
            if val == -9999:
                continue
            try:
                date = pd.Timestamp(year=year, month=month, day=d + 1)
            except ValueError:
                continue        # e.g. day 31 of a 30-day month
            recs.setdefault(date, {})[elem.lower()] = val / 10.0
    df = pd.DataFrame.from_dict(recs, orient="index").sort_index()
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))


def _ghcnd(force=False):
    for station, label in GHCND_STATIONS:
        out = REAL / f"ghcnd_{label}.csv"
        if out.exists() and not force:
            continue
        _say(f"  GHCNd {station} ({label}) ...")
        df = _parse_dly(_http_get(GHCND_URL.format(station=station), timeout=180))
        df.to_csv(out, index_label="date")
        _say(f"  wrote {out.name}: {len(df)} days, "
             f"{df.index.min():%Y-%m} to {df.index.max():%Y-%m}, "
             f"cols={list(df.columns)}")


# ---------------------------------------------------------------- USGS
def _read_rdb(text):
    """USGS RDB: '#' comments, a header row, then a types row to discard."""
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if len(lines) < 3:
        raise RuntimeError("USGS returned no data rows")
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", skiprows=[1],
                       dtype=str)


def _usgs(force=False):
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    out = REAL / "usgs_potomac_daily_discharge.csv"
    if force or not out.exists():
        _say("  USGS daily discharge ...")
        df = _read_rdb(_http_get(USGS_DV.format(site=USGS_SITE, end=end), timeout=120))
        val = [c for c in df.columns if c.endswith("_00060_00003")][0]
        df = df[["datetime", val]].rename(columns={val: "discharge_cfs"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")
        df = df.set_index("datetime").sort_index()
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
        df.to_csv(out, index_label="date")
        _say(f"  wrote {out.name}: {len(df)} days, "
             f"{df.index.min():%Y-%m} to {df.index.max():%Y-%m}")

    out = REAL / "usgs_potomac_15min_discharge.csv"
    if force or not out.exists():
        # the instantaneous service refuses very long spans; five years keeps
        # the request inside its limit and the file inside a sane size
        start = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
        _say("  USGS 15-min discharge (5 years) ...")
        df = _read_rdb(_http_get(USGS_IV.format(site=USGS_SITE, start=start, end=end),
                                 timeout=300))
        val = [c for c in df.columns if c.endswith("_00060")][0]
        df = df[["datetime", val]].rename(columns={val: "discharge_cfs"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")
        df = df.set_index("datetime").sort_index()
        df.to_csv(out, index_label="timestamp")
        _say(f"  wrote {out.name}: {len(df)} readings, "
             f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")


# ---------------------------------------------------------------- PSMSL
def _psmsl(force=False):
    for sid, label in PSMSL_STATIONS:
        out = REAL / f"psmsl_{label}_monthly_sealevel.csv"
        if out.exists() and not force:
            continue
        _say(f"  PSMSL {label} ...")
        text = _http_get(PSMSL_URL.format(sid=sid), timeout=60)
        rows = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 2:
                continue
            frac, mm = float(parts[0]), float(parts[1])
            if mm <= -99999:
                mm = np.nan
            year = int(frac)
            month = int(round((frac - year) * 12 + 0.5))
            rows.append((pd.Timestamp(year=year, month=min(max(month, 1), 12), day=1),
                         mm))
        df = pd.DataFrame(rows, columns=["date", "rlr_mm"]).set_index("date")
        df.to_csv(out, index_label="date")
        _say(f"  wrote {out.name}: {len(df)} months, "
             f"{df.index.min():%Y-%m} to {df.index.max():%Y-%m}")


# ---------------------------------------------------------------- OPSD
def _opsd(force=False):
    """European hourly electricity load -- the real counterpart to the surrogate.

    The published file is ~130 MB of every variable for every country; only the
    load columns are kept, which is ~1% of that.
    """
    out = REAL / "opsd_hourly_load.csv"
    if out.exists() and not force:
        return
    _say("  OPSD hourly load (downloading ~130 MB, keeping load columns only) ...")
    text = _http_get(OPSD_URL, timeout=900)
    df = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=[0],
                     low_memory=False)
    suffix = "_load_actual_entsoe_transparency"
    keep = {}
    for c in df.columns:
        if not c.endswith(suffix):
            continue
        code = c[: -len(suffix)]
        # The prefix is either a bare country code (DE) or a bidding zone /
        # TSO area (DE_50hertz, NO_1). Keep only whole countries: the zones are
        # components of a country already present, and collapsing them all to
        # the country code silently produced duplicate names.
        if "_" in code:
            continue
        # drop anything with real holes; a 35%-missing series is not a series
        if df[c].isna().mean() > 0.01:
            continue
        keep[c] = code
    if not keep:
        raise RuntimeError("no sufficiently complete load columns found in OPSD")
    sub = df[list(keep)].rename(columns=keep)
    assert sub.columns.is_unique, f"duplicate country codes: {list(sub.columns)}"
    sub.index = sub.index.tz_localize(None)
    sub.to_csv(out, index_label="timestamp")
    _say(f"  wrote {out.name}: {sub.shape[0]} hours x {sub.shape[1]} countries, "
         f"{sub.index.min():%Y-%m} to {sub.index.max():%Y-%m}")


# ----------------------------------------------------------------------
# surrogate
# ----------------------------------------------------------------------
def build_surrogate(force=False):
    SURR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(STAGE1))
    import datasets as ds

    jobs = [
        ("solar_ghi.csv", lambda: ds.solar_surrogate(),
         "30-min GHI, one year, lat 1.35N; Haurwitz clear-sky x Markov cloud"),
        ("system_load.csv", lambda: ds.load_surrogate(),
         "hourly system load, three years; weekly/diurnal x seasonal + CDD/HDD"),
    ]
    for name, fn, note in jobs:
        out = SURR / name
        if force or not out.exists():
            df = fn()
            df.to_csv(out, index_label="timestamp")
            _say(f"  wrote {name}: {df.shape[0]} rows x {df.shape[1]} cols  ({note})")


# ----------------------------------------------------------------------
# chaotic -- Lorenz-63, integrated here so the tier is never empty
# ----------------------------------------------------------------------
def _lorenz63_realizations(n_real=12, n=4000, dt=0.01, eps=1e-6, seed=0,
                           sigma=10.0, rho=28.0, beta=8.0 / 3.0, burn=4000):
    """R realizations of Lorenz-63, all starting from one attractor point.

    Perturbing *after* the burn-in is what makes these a usable ensemble: every
    member starts on the attractor, separated by a known eps, so their pairwise
    divergence measures lambda_1 directly with no delay embedding involved.
    """
    def f(v):
        x, y, z = v
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def step(s):
        k1 = f(s); k2 = f(s + dt / 2 * k1)
        k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    s = np.array([1.0, 1.0, 1.0])
    for _ in range(burn):
        s = step(s)

    rng = np.random.default_rng(seed)
    out = np.empty((n_real, n))
    for r in range(n_real):
        v = s + eps * rng.standard_normal(3)
        for i in range(n):
            out[r, i] = v[0]
            v = step(v)
    return out, dt


def build_chaotic(force=False):
    CHAOS.mkdir(parents=True, exist_ok=True)
    out = CHAOS / "lorenz63.npz"
    if force or not out.exists():
        x, dt = _lorenz63_realizations()
        np.savez(out, x=x, dt=dt, lyap=0.9056, ic_eps=1e-6,
                 note="Lorenz-63 x-component; lambda_1 = 0.9056 is the "
                      "standard published value for sigma=10, rho=28, beta=8/3")
        _say(f"  wrote {out.name}: {x.shape[0]} realizations x {x.shape[1]}, "
             f"dt={dt}, lambda_1=0.9056")


# ----------------------------------------------------------------------
# chaotic -- dysts systems (opt-in, slow)
# ----------------------------------------------------------------------
DYSTS_SYSTEMS = [
    ("VallisElNino", "ENSO recharge-oscillator model, 3-D"),
    ("Lorenz84", "atmospheric general circulation, 3-D"),
    ("Hadley", "Hadley cell convection, 3-D"),
    ("RikitakeDynamo", "geomagnetic reversal dynamo, 3-D"),
    ("Lorenz96", "standard data-assimilation testbed, 4-D (SLOW)"),
    ("AtmosphericRegime", "atmospheric regime transitions, 3-D"),
]


def build_dysts(force=False, n_real=8, n=2500, eps=1e-6, only=None):
    try:
        import dysts.flows as flows
    except ImportError:
        _say("  dysts not installed -- skipping. `pip install dysts numba`")
        return
    CHAOS.mkdir(parents=True, exist_ok=True)
    for name, note in DYSTS_SYSTEMS:
        if only and name not in only:
            continue
        out = CHAOS / f"{name.lower()}.npz"
        if out.exists() and not force:
            _say(f"  {out.name} cached")
            continue
        _say(f"  integrating {name} ({n_real} realizations)...")
        m = getattr(flows, name)()
        ic0 = np.array(m.ic, dtype=float)
        rng = np.random.default_rng(1)
        reals, t = [], None
        for _ in range(n_real):
            m.ic = ic0 + eps * rng.standard_normal(ic0.shape)
            t, tr = m.make_trajectory(n, resample=True, return_times=True)
            reals.append(np.asarray(tr)[:, 0])
        dt = float(np.median(np.diff(np.asarray(t).ravel())))
        np.savez(out, x=np.array(reals), dt=dt,
                 lyap=float(m.maximum_lyapunov_estimated), ic_eps=eps, note=note)
        _say(f"    wrote {out.name}: dt={dt:.5f} "
             f"lambda_1={m.maximum_lyapunov_estimated:.4f}")


# ----------------------------------------------------------------------
def load(path):
    """Convenience loader: .npz -> (x, dt, lyap); .csv -> DataFrame."""
    path = Path(path)
    if not path.is_absolute():
        for d in (CHAOS, REAL, SURR, HERE):
            if (d / path).exists():
                path = d / path
                break
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=True)
        return d["x"], float(d["dt"]), float(d["lyap"])
    return pd.read_csv(path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dysts", action="store_true",
                   help="also integrate the dysts chaotic systems (slow)")
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict --dysts to these system names")
    p.add_argument("--force", action="store_true",
                   help="re-download and regenerate everything")
    a = p.parse_args()

    _say("real/")
    build_real(a.force)
    _say("surrogate/")
    build_surrogate(a.force)
    _say("chaotic/")
    build_chaotic(a.force)
    if a.dysts:
        build_dysts(a.force, only=a.only)
    else:
        _say("  (skipping dysts systems; pass --dysts to integrate them)")
    _say("done")


if __name__ == "__main__":
    main()
