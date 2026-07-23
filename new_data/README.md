# Climate/ENSO Dataset Collection (raw archives)

Manually-placed raw archives — not fetched by any script here, and not
git-tracked (see the root `.gitignore`). `Quantathon_stack/Data/fetch_cp_data.py`
reads directly out of this directory where noted below; nothing else in the
repo expects it to exist, so a clone without `new_data/` still works (it just
falls back to the slower ERDDAP path for `got_sst`, and skips `got_hadisst` /
`maeklong_spei` with a clear error).

## Wired into the pipeline

- **`oisst_v2.1/`** -- NOAA OISST v2.1, daily SST, 0.25 deg, Sept 1981-2026 (46
  yearly NetCDFs, ~20GB). `fetch_cp_data.py` extracts the upper-Gulf-of-Thailand
  point (13.125N 100.125E) directly from these files instead of the ~45-round
  ERDDAP year-by-year fetch -> `Data/real/oisst_gulf_thailand_daily_sst.csv`,
  loader key `got_sst`. Same source/point either way; this is just the fast path.
- **`hadisst1/`** -- UK Met Office HadISST1, monthly SST, 1 deg, 1870-present.
  Methodologically independent of ERSSTv5 (different interpolation/sea-ice
  treatment) -- a cross-check: a signal appearing in both is unlikely to be a
  reconstruction artifact of either. Extracted at 13.5N 100.5E (nearest valid
  ocean cell to the ERSST/OISST points -- the 1 deg cell closer to 12N/100E is
  land in this grid) -> `Data/real/hadisst_gulf_thailand_monthly_sst.csv`,
  loader key `got_hadisst`. This is HadISST1, not HadISST2 -- HadISST2's SST
  fields aren't publicly downloadable (only its sea-ice data is open).
- **`speibase/`** -- SPEIbase, 3-month Standardized Precipitation-Evapotranspiration
  Index, 0.5 deg, monthly. This vintage runs 1901-2015 (not updated past 2015).
  Extracted at the same point as the Mae Klong rain record (13.75N 99.75E) ->
  `Data/real/spei03_mae_klong_monthly.csv`, loader key `maeklong_spei`. Folds
  temperature-driven evapotranspiration into the rain signal, so it's a better
  single drought/salinity-risk number than raw rain alone (see
  `presentation/TODO.md`'s rain-to-pond-salinity follow-up) -- with the caveat
  that it's frozen at 2015 and needs a bias-corrected join to bring it current.

## Acquired, not wired in

- **`ersstv5/sst.mnmean.nc`** -- already reproducibly fetched via ERDDAP by
  `fetch_cp_data.py` (`got_ersst`); this raw global file wasn't needed.
- **`enso_indices/`** (`nino1+2.data`, `nino3.data`, `nino3.4.data`, `nino4.data`,
  `mei_v2.data`, `soi_cpc.txt`) -- monthly ASCII indices. Overlaps what's already
  registered (`nino34`, `oni`); not wired in because it would duplicate rather
  than extend the current story. Worth registering individually if a future
  analysis specifically wants the Nino 1+2 lead/lag structure or MEI.v2 as a
  single composite ENSO number.
- **`mrc_discharge/`** -- Mekong River Commission daily discharge, per-station
  JSON (Highcharts export format), ~15 gauges. **Wrong watershed for this use
  case**: every station here (Ban Kruat, Satuk, Ubon, Yasothon, ... plus the
  Lao PDR mainstem gauges) sits on the Mun River / Mekong mainstem, which drains
  east through Laos/Cambodia/Vietnam to the South China Sea -- not into the Gulf
  of Thailand where the Mae Klong/CP aquaculture sites are. Kept here (it was
  genuinely hard to acquire -- see below) but not registered as a `got_*` /
  `maeklong_*` driver; it would be the right data for a *separate* Mekong Delta
  / Vietnamese aquaculture pitch, not this one.

## Previously blocked, now resolved by this drop

- Mekong River Commission discharge data was listed as blocked (portal required
  login for bulk export) -- now present, see caveat above on why it's still
  not wired into the Gulf-of-Thailand pipeline.
- ERA5 (blocked on a Copernicus CDS API key) and EM-DAT (blocked on account
  registration) are still not present.
