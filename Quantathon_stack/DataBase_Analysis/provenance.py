"""provenance.py -- what each dataset is, where it came from, what it means.

The numeric modules deliberately know nothing about meaning: `dataloader.py`
turns a file into (x, dt, unit) and stops there. That is the right boundary for
code, but it makes a report of bare exponents unreadable -- 0.0626 /day is not
interpretable without knowing that it is a 157-year daily temperature record
from one Manhattan weather station.

This module is the missing half: one `Doc` per dataset key, carrying the
source, the physical meaning of the observable, and the specific things that
would mislead someone who took the number at face value.

    from provenance import DOCS
    DOCS["nyc"].source        # where it came from
    DOCS["nyc"].watch_out     # what will bite you

Keys match `dataloader.ALL_KEYS`; `check()` asserts they stay in step.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Doc:
    """Human-facing documentation for one dataset.

    Attributes
    ----------
    title      : one-line identification
    source     : provider, product name, and URL
    observable : what the recorded number physically is, including units
    meaning    : the process generating it, and why it is in this collection
    watch_out  : the specific gotchas -- empty list means genuinely clean
    """

    title: str
    source: str
    observable: str
    meaning: str
    watch_out: list[str] = field(default_factory=list)


_DYSTS = ("[dysts](https://github.com/williamgilpin/dysts) -- 135 chaotic "
          "systems, each annotated with a numerically estimated largest "
          "Lyapunov exponent (Gilpin, NeurIPS 2021)")

DOCS: dict[str, Doc] = {
    # ------------------------------------------------------------ chaotic
    "lorenz63": Doc(
        title="Lorenz-63, x-component",
        source="integrated locally by `Data/fetch_data.py` (RK4, dt=0.01, "
               "sigma=10, rho=28, beta=8/3)",
        observable="dimensionless convective overturning rate; arbitrary units",
        meaning="The canonical chaotic system: a 3-mode truncation of "
                "Rayleigh-Benard convection. Its reference lambda_1 = 0.9056 "
                "is a standard literature value rather than a re-estimate, "
                "which makes it the cleanest calibration target here.",
        watch_out=["Non-dimensional time. Its 'natural' unit has no physical "
                   "meaning, so T_lambda cannot be compared in days against "
                   "the observed records."],
    ),
    "vallieselnino": Doc(
        title="Vallis ENSO oscillator",
        source=_DYSTS,
        observable="dimensionless zonal current / thermocline anomaly",
        meaning="A low-order coupled ocean-atmosphere model of El Nino: wind "
                "stress drives an equatorial current, which advects the "
                "thermocline, which feeds back on the wind. The most relevant "
                "chaotic system in this collection -- it is a toy model of the "
                "same physics the Nino indices observe, so it is the natural "
                "sanity check for any method later applied to ENSO.",
        watch_out=["A caricature of ENSO, not a simulation of it. Agreement "
                   "with the real indices would be a coincidence of "
                   "timescales, not validation."],
    ),
    "lorenz84": Doc(
        title="Lorenz-84 general circulation",
        source=_DYSTS,
        observable="dimensionless westerly wind amplitude and eddy components",
        meaning="Lorenz's low-order model of the mid-latitude atmospheric "
                "circulation: a westerly jet coupled to two eddy modes, forced "
                "by the equator-to-pole temperature contrast. The conceptual "
                "ancestor of the weather-predictability question that the "
                "GHCN station records answer empirically.",
        watch_out=["Our ensemble estimate is +62.5% against its reference "
                   "value -- the worst error in the chaotic tier, and a "
                   "reminder that the estimator is not uniformly reliable."],
    ),
    "hadley": Doc(
        title="Hadley cell convection",
        source=_DYSTS,
        observable="dimensionless convective amplitude",
        meaning="Thermal convection in a rotating annulus, the idealised "
                "Hadley circulation that transports heat from the tropics "
                "poleward.",
        watch_out=["Ensemble estimate is +57.5% against its reference value."],
    ),
    "rikitake": Doc(
        title="Rikitake two-disc dynamo",
        source=_DYSTS,
        observable="dimensionless current in two coupled disc dynamos",
        meaning="A two-disc dynamo whose current spontaneously reverses "
                "polarity -- the classic minimal model for geomagnetic field "
                "reversals. Included as a slow-exponent test case: at "
                "lambda_1 = 0.13 it stresses the estimator differently from "
                "the fast systems.",
        watch_out=["Ensemble estimate is -20.4%.",
                   "The only chaotic system whose T_lambda/period ratio "
                   "exceeds 1 (1.36), so its exponent and its dominant "
                   "oscillation are not well separated."],
    ),

    # ------------------------------------------------------------ real
    "nino34": Doc(
        title="Nino 3.4 sea-surface-temperature anomaly",
        source="[NOAA PSL](https://psl.noaa.gov/data/timeseries/monthly/NINO34/), "
               "ERSST v6",
        observable="monthly SST anomaly in degC, averaged over the Nino 3.4 "
                   "box (5N-5S, 170W-120W)",
        meaning="The standard index of El Nino / La Nina state, and the "
                "target the whole QRC pipeline was built around. Positive "
                "excursions are El Nino, negative La Nina; the +/-0.5 degC "
                "threshold is the operational event definition.",
        watch_out=["Already an anomaly -- a monthly climatology was subtracted "
                   "upstream, so the annual cycle is gone before you see it. "
                   "Do not deseasonalise twice.",
                   "A gridded reanalysis blending ship, buoy and satellite "
                   "observations, not a direct measurement.",
                   "918 monthly points is a short record for a Lyapunov "
                   "estimate: 13 e-foldings total."],
    ),
    "nino12": Doc(
        title="Nino 1+2 sea-surface temperature (absolute)",
        source="NOAA/NCEP, bundled with statsmodels; pinned in "
               "`Data/real/enso_nino12_1950_2010.csv`",
        observable="monthly mean SST in degC (not an anomaly)",
        meaning="The record the existing QRC walkthrough was built on. The "
                "Nino 1+2 region (0-10S, 90W-80W) hugs the South American "
                "coast, where SST variance phase-locks to boreal spring rather "
                "than the DJF peak of Nino 3.4.",
        watch_out=["**Raw SST, so the annual cycle dominates the spectrum** at "
                   "four times the amplitude of any other line. This is what "
                   "inflates its lambda_1 to 3.6x the Nino 3.4 value -- the "
                   "phase-matching artefact, running live.",
                   "Ends in 2010, and is not directly comparable with "
                   "`nino34` without deseasonalising first.",
                   "Kept for continuity with published results. Prefer "
                   "`nino34` for new work."],
    ),
    "tao": Doc(
        title="TAO/TRITON moored buoy SST, 0N 140W",
        source="[NOAA PMEL](https://www.pmel.noaa.gov/gtmba/pmel-theme/pacific-ocean-tao) "
               "via ERDDAP (`pmelTaoDySst`, variable `T_25`)",
        observable="daily-mean sea surface temperature in degC from a moored "
                   "buoy on the equator at 140W",
        meaning="A direct in-situ measurement of the same ocean the Nino "
                "indices average over, at *daily* rather than monthly cadence. "
                "0N 140W sits in the equatorial cold tongue, the region where "
                "the ENSO SST signal is strongest. The parent file carries "
                "five moorings spanning 165E to 110W; per-site standard "
                "deviation rises from 0.79 degC in the warm pool to 2.21 degC "
                "in the cold tongue, which is the ENSO signal strengthening "
                "eastward.",
        watch_out=["**25-39% missing per site, with gaps of 1,135-4,153 "
                   "consecutive days** from the array's funding-driven "
                   "degradation around 2012-2014.",
                   "Loaded with `longest_run=True`, which crops a 46-year "
                   "record to its longest gapless stretch: 1,684 of 16,919 "
                   "days, about 4.6 years.",
                   "The five moorings are *not* independent realizations -- "
                   "they are dynamically coupled along the equator, so their "
                   "spread is a lower bound on true ensemble spread and cannot "
                   "drive the ensemble estimator."],
    ),
    "nyc": Doc(
        title="New York Central Park, daily maximum temperature",
        source="[NOAA NCEI GHCN-Daily](https://www.ncei.noaa.gov/products/"
               "land-based-station/global-historical-climatology-network-daily), "
               "station USW00094728",
        observable="daily maximum air temperature in degC (TMAX, converted "
                   "from the archive's tenths)",
        meaning="Mid-latitude weather at a single station, and the longest "
                "record in this collection at any cadence: **157.5 years, "
                "57,540 daily observations, 7 missing values**. Its dynamics "
                "are synoptic weather -- the passage of frontal systems -- "
                "rather than climate.",
        watch_out=["Its T_lambda of 16 days lands on the textbook ~2-week "
                   "atmospheric predictability limit without any tuning, which "
                   "is the strongest evidence in this directory that the "
                   "pipeline is sound when the data is good.",
                   "Urban heat-island growth over 157 years is a real "
                   "non-stationarity, even though GHCN applies quality "
                   "control.",
                   "TMAX is a daily *extremum*, not a daily mean, so it is a "
                   "nonlinear functional of the underlying continuous process."],
    ),
    "lax": Doc(
        title="Los Angeles International Airport, daily maximum temperature",
        source="NOAA NCEI GHCN-Daily, station USW00023174",
        observable="daily maximum air temperature in degC",
        meaning="A contrasting climate regime to `nyc`: coastal "
                "Mediterranean, with a far weaker seasonal cycle and much "
                "lower day-to-day variance (sd 4.1 degC against 10.4). Useful "
                "as a control on whether an estimate is tracking weather or "
                "tracking the annual cycle.",
        watch_out=["82 years rather than 157, but comparably clean."],
    ),
    "potomac": Doc(
        title="Potomac River discharge, daily",
        source="[USGS NWIS](https://waterservices.usgs.gov/) gauge 01646500, "
               "'Potomac River near Washington DC, Little Falls Pump Station' "
               "(38.9498N, 77.1276W)",
        observable="daily mean discharge in cubic feet per second, analysed as "
                   "log10(Q)",
        meaning="River flow integrating rainfall, snowmelt and groundwater "
                "over an 11,500 square-mile basin -- a physically filtered, "
                "heavily damped response to weather forcing. Maps to SDG 6 "
                "(clean water and sanitation).",
        watch_out=["**Only 5 samples per T_lambda -- below the ~10 needed for "
                   "a scaling region to exist.** Its own 15-minute record "
                   "disagrees by a factor of 14. Prefer `potomac15`.",
                   "Discharge spans 121 to 426,000 cfs, so it is analysed as "
                   "log10(Q); a few flood peaks would otherwise set the scale "
                   "for the spectrum and the neighbour search alike.",
                   "Flow is regulated upstream and abstracted for DC's water "
                   "supply, so this is not a natural catchment response."],
    ),
    "potomac15": Doc(
        title="Potomac River discharge, 15-minute",
        source="USGS NWIS instantaneous-values service, same gauge 01646500",
        observable="instantaneous discharge in cfs at nominal 15-minute "
                   "spacing, analysed as log10(Q)",
        meaning="The same river as `potomac` at 60x the cadence, and the "
                "highest-frequency real record in this collection (175,284 "
                "samples after resampling). Because it is the same physical "
                "system measured two ways, the pair is a direct test of how "
                "much sampling rate alone moves the exponent.",
        watch_out=["**The raw feed is not on a clean grid** -- 44% of steps "
                   "differ from 15 minutes, some running to 19 hours. The "
                   "loader resamples onto a uniform grid before anything else "
                   "touches it.",
                   "Five years only: the instantaneous service will not serve "
                   "long spans.",
                   "Gives T_lambda = 0.4 days against 5.5 days for the daily "
                   "record of the same river. Sampling rate is setting the "
                   "exponent."],
    ),
    "opsd": Doc(
        title="German electricity load, hourly",
        source="[Open Power System Data](https://data.open-power-system-data.org/time_series/), "
               "originally ENTSO-E Transparency",
        observable="actual national electricity demand in MW, hourly",
        meaning="Real grid demand -- the physical counterpart to the "
                "`load` surrogate. Human activity cycles (diurnal, weekly, "
                "annual) plus a weather-driven heating/cooling response. "
                "Germany is the largest and most complete series in the "
                "25-country file.",
        watch_out=["**The surrogate is over-chaotic by 2.7x**: real demand "
                   "gives T_lambda = 56.5 h against the surrogate's 21.1 h, "
                   "with both showing the same 24 h dominant line. Anything "
                   "tuned on `load` should be re-checked here.",
                   "Only 2014-12 to 2020-09; the upstream release stops there.",
                   "T_lambda/period = 2.35, close enough to the diurnal cycle "
                   "that some phase-matching contribution is plausible."],
    ),
    "brest": Doc(
        title="Brest tide gauge, monthly mean sea level",
        source="[PSMSL](https://psmsl.org/) station 1, Revised Local "
               "Reference series",
        observable="monthly mean sea level in mm on the RLR datum",
        meaning="One of the longest instrumental records on Earth, beginning "
                "**1807**. Sea level integrates thermal expansion, ice-mass "
                "exchange, atmospheric pressure and local land motion. Maps to "
                "SDG 13/14.",
        watch_out=["11.9% missing with a 120-month gap, so it is loaded with "
                   "`longest_run=True`: 509 of 2,622 months, about 42 years.",
                   "That crop is also why it does not fail the way `cuxhaven` "
                   "does -- it drops most of the secular trend along with most "
                   "of the record.",
                   "Contains a strong secular trend (sea-level rise). "
                   "Detrend before any dynamical claim."],
    ),
    "cuxhaven": Doc(
        title="Cuxhaven tide gauge, monthly mean sea level",
        source="PSMSL station 10, Revised Local Reference series",
        observable="monthly mean sea level in mm on the RLR datum",
        meaning="A German Bight record from 1854, far more complete than "
                "Brest (0.44% missing) and therefore analysed at full length.",
        watch_out=["**Returns lambda_1 = -2.2 /yr, which is an estimator "
                   "failure, not a finding.** A strong secular trend plus a "
                   "dominant annual line makes embedded neighbours converge "
                   "rather than diverge, so the fit reports a negative slope "
                   "and T_lambda = infinity.",
                   "Detrend and deseasonalise before believing anything about "
                   "sea-level dynamics from this series.",
                   "North Sea storm surge makes the local signal much noisier "
                   "than open-ocean sea level."],
    ),

    # ------------------------------------------------------------ surrogate
    "solar": Doc(
        title="Solar GHI surrogate, 30-minute",
        source="generated by `QRC_code_stack/stage1_numpy_core/datasets.py` "
               "at a fixed seed",
        observable="global horizontal irradiance in W/m^2, one year, "
                   "latitude 1.35N",
        meaning="Haurwitz clear-sky geometry multiplied by a two-state Markov "
                "cloud process, with sensor-outage gaps injected. Built so "
                "that the deterministic part (solar position) and the "
                "stochastic part (cloud) are separable by construction.",
        watch_out=["**Synthetic.** lambda_1 describes the generator, not any "
                   "sky.",
                   "The injected outages are interpolated on load, which "
                   "biases the exponent optimistic.",
                   "Its real counterpart, BSRN 1-minute irradiance, needs an "
                   "account and so is not vendored."],
    ),
    "load": Doc(
        title="System load surrogate, hourly",
        source="generated by `QRC_code_stack/stage1_numpy_core/datasets.py` "
               "at a fixed seed",
        observable="system electricity load in MW, three years",
        meaning="Weekly and diurnal profiles times an annual seasonality, plus "
                "a CDD/HDD temperature response, an AR(1) residual and holiday "
                "suppression.",
        watch_out=["**Synthetic, and demonstrably over-chaotic**: T_lambda = "
                   "21.1 h against 56.5 h for real German demand (`opsd`). "
                   "Its AR(1) residual is noisier than reality.",
                   "Use `opsd` for any claim meant to be about electricity "
                   "demand."],
    ),
}


def check() -> None:
    """Fail loudly if the registry drifts out of step with the loader."""
    import dataloader as dl
    missing = set(dl.ALL_KEYS) - set(DOCS)
    extra = set(DOCS) - set(dl.ALL_KEYS)
    if missing or extra:
        raise AssertionError(
            f"provenance out of step: missing={sorted(missing)}, "
            f"undocumented-key={sorted(extra)}")


if __name__ == "__main__":
    check()
    for key, doc in DOCS.items():
        print(f"\n=== {key}: {doc.title}")
        print(f"  source     {doc.source}")
        print(f"  observable {doc.observable}")
        for w in doc.watch_out:
            print(f"  ! {w}")
