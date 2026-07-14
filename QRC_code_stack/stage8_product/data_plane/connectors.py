"""connectors.py -- L1 ingestion: the dataset registry as code (Phase 8).

Every source below carries its FULL acquisition detail (URL pattern,
format, resolution, licence/attribution, expected size) so a stranger can
fetch it with no other document. The vetting checklist (qrc-project-
playbook sec. 2) is embedded as `VETTING` and must be walked per source
BEFORE any model touches its data. Raw files are immutable: connectors
only READ from data/raw/<source>/, never write into it (downloads land
there once, by hand or by the printed command).

Implemented and runnable TODAY (no network, no keys):
    enso_noaa()        real NOAA Nino SST from the pinned stage-1 cache
    solar_surrogate()  the stage-1 statistical surrogate -- ALWAYS
                       labelled "surrogate/synthetic" wherever a number
                       appears (playbook surrogate policy)

Documented stubs (raise with exact acquisition instructions until the
data is present locally):
    surfrad()          NOAA SURFRAD 1-min ground truth (Track A primary)
    nsrdb()            NREL NSRDB 30-min gridded (Track A secondary)
    era5()             ECMWF ERA5 reanalysis (exogenous channels)
    site_scada()       private customer feed (schema contract only)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "stage8_product" / "data" / "raw"

VETTING = """Vetting checklist (ALL must pass before modelling):
1. provenance + licence verified on the provider page; version pinned
2. length >= a few thousand steps at modelling resolution
3. missing data located; policy = MASK, never silently interpolate;
   gaps > autocorrelation time split the series
4. target transform identified from physics (solar -> k_t, climate ->
   anomaly, load -> calendar residual); baselines apply to TRANSFORMED
5. leakage audit: every statistic computed on the training span only
6. a living classical benchmark literature exists"""


@dataclass
class SourceSpec:
    """One registry entry; `detail` is the complete acquisition recipe."""
    name: str
    kind: str          # 'real' | 'surrogate' | 'private'
    resolution: str
    licence: str
    detail: str


REGISTRY: dict[str, SourceSpec] = {
    "enso_noaa": SourceSpec(
        "enso_noaa", "real", "monthly",
        "NOAA public domain; cite NOAA CPC Nino SST",
        "Ships with statsmodels ('elnino'); pinned cache at "
        "stage1_numpy_core/data/raw/enso.csv (732 months, 1950-01..2010-12)."),
    "solar_surrogate": SourceSpec(
        "solar_surrogate", "surrogate", "30-min",
        "self-generated; no licence constraint",
        "stage1_numpy_core/datasets.py solar_surrogate(seed=42): clear-sky "
        "geometry x stochastic cloud process, EDA-matched to the tropical "
        "archetype. SURROGATE -- label every figure/table; real-data "
        "pathway = surfrad()."),
    "surfrad": SourceSpec(
        "surfrad", "real", "1-min",
        "NOAA public domain; attribution: NOAA GML SURFRAD",
        "https://gml.noaa.gov/aftp/data/radiation/surfrad/<station>/<year>/"
        "<station><yy><doy>.dat  -- 7 US stations (bon, tbl, dra, fpk, gwn, "
        "psu, sxf); ~1.5 MB/day/station ASCII; columns incl. dw_solar "
        "(global downwelling W/m^2), qc flags. Aggregate 1-min -> 5-min "
        "AFTER masking qc!=0. ~0.5 GB/station-year. Track A ground truth."),
    "nsrdb": SourceSpec(
        "nsrdb", "real", "30-min gridded",
        "NREL; free API key required; cite NSRDB (Sengupta et al. 2018)",
        "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download"
        ".csv?api_key=<KEY>&wkt=POINT(<lon> <lat>)&names=<years>&"
        "attributes=ghi,dni,clearsky_ghi,cloud_type&interval=30 -- CSV per "
        "point-year (~1 MB). Includes clearsky_ghi (their k_t path). "
        "Track A secondary / gridded expansion."),
    "era5": SourceSpec(
        "era5", "real", "hourly reanalysis",
        "Copernicus CDS licence; free account; cite Hersbach et al. 2020",
        "cdsapi python client; dataset 'reanalysis-era5-single-levels', "
        "variables e.g. ssrd, tcc, t2m; ~2 GB/variable-year global, "
        "~MBs for a site box. Exogenous L2 channels, NOT a competitor."),
    "site_scada": SourceSpec(
        "site_scada", "private", "1-15 min",
        "customer-owned; derived-features-only leave the boundary",
        "Contract schema: timestamp (UTC, monotone), pv_power_kw, "
        "poa_irradiance_wm2 (optional), availability_flag. Adapter must "
        "declare timezone, curtailment convention, and sensor calibration "
        "date. Raw NEVER leaves the customer boundary."),
}


def enso_noaa() -> tuple[np.ndarray, dict]:
    """Real monthly Nino SST from the pinned stage-1 cache."""
    import csv
    cache = REPO / "stage1_numpy_core" / "data" / "raw" / "enso.csv"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} missing -- stage 1 ships it; see MOVE_MAP.md")
    with open(cache) as f:
        rows = list(csv.DictReader(f))
    key = [k for k in rows[0] if k.lower() not in ("year", "month",
                                                   "date")][0]
    sst = np.array([float(r[key]) for r in rows])
    return sst, {"source": "enso_noaa", "kind": "real", "n": len(sst),
                 "spec": REGISTRY["enso_noaa"]}


def solar_surrogate(seed: int = 42) -> tuple[np.ndarray, dict]:
    """Stage-1 statistical surrogate. Kind='surrogate' propagates into
    every downstream label (figures, tables, alerts)."""
    import sys
    sys.path.insert(0, str(REPO / "stage1_numpy_core"))
    from datasets import solar_surrogate as _gen
    df = _gen(seed=seed)   # DataFrame: ghi, ghi_clear, kt, elevation_deg
    ghi = df["ghi"].to_numpy(dtype=float)   # carries injected NaN gaps
    return ghi, {"source": "solar_surrogate", "kind": "surrogate",
                 "n": len(ghi), "seed": seed,
                 "spec": REGISTRY["solar_surrogate"]}


def _stub(name: str):
    spec = REGISTRY[name]
    raise FileNotFoundError(
        f"{name}: no local data under {RAW / name}/.\n"
        f"Acquisition: {spec.detail}\nLicence: {spec.licence}\n\n{VETTING}")


def surfrad(station: str = "bon", year: int = 2023):
    """Track A primary. Download per the registry detail into
    data/raw/surfrad/<station>/<year>/ then implement the column parse
    against the SURFRAD README (masking qc!=0)."""
    path = RAW / "surfrad" / station / str(year)
    if not path.exists():
        _stub("surfrad")
    raise NotImplementedError(
        "surfrad parser lands with the Track-A study (stage-7 solar plan); "
        "vetting checklist first")


def nsrdb(*a, **k):
    _stub("nsrdb")


def era5(*a, **k):
    _stub("era5")


def site_scada(*a, **k):
    _stub("site_scada")


if __name__ == "__main__":
    print("registry:")
    for s in REGISTRY.values():
        print(f"  {s.name:16s} [{s.kind:9s}] {s.resolution:16s} {s.licence}")
    sst, meta = enso_noaa()
    print(f"\nenso_noaa: {meta['n']} months, mean {sst.mean():.2f} C [real]")
    ghi, meta = solar_surrogate()
    print(f"solar_surrogate: {meta['n']} steps [SURROGATE - label "
          "everywhere]")
