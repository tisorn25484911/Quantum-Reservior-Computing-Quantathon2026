# CP Aquaculture Use Case — Handoff README

**What this is:** the product/use-case layer added on top of the QRC
Forecast-then-Detect line — a marine anomaly-forecasting pitch (working name
**TideRead**) aimed at CP-scale shrimp/marine aquaculture in the Gulf of Thailand,
plus the company-relevant datasets that prove the pipeline generalises to the
customer's own waters.

**Read this first if you are picking the project up on another machine.** It records
exactly what was done on 2026-07-23, and (as of the same day, after a `new_data/`
archive drop) what the Gulf-of-Thailand result actually shows.

---

## TL;DR — the state in six lines

1. **Use case chosen:** daily-SST marine-heat / salinity anomaly *forecasting* for
   Gulf-of-Thailand aquaculture. Picked because the QRC beats the classical ESN
   robustly on **daily SST** (`tao`, 5/5 seeds), not on monthly ENSO.
2. **Six CP-relevant datasets acquired**, all wired into the pipeline (see below).
3. **Deck built & published** as an Artifact: `presentation/deck.html`.
4. **Speaker notes + Q&A** written: `presentation/speaker_notes.md`.
5. **Gulf-of-Thailand daily SST is downloaded and measured**: full 45-year record
   (16,394 days), 5-seed kill-test, **PASS** — real skill, beats persistence
   day 12→24, never materially loses to the size-matched ESN — with two honest
   caveats (seed-dependent kind, climatology floor at ~day 20). See `TODO.md`.
6. **Two bonus records wired in** from the same archive drop: `got_hadisst`
   (independent SST cross-check) and `maeklong_spei` (drought index).

---

## Environment setup (fresh machine)

One virtualenv at the repo root serves everything.

```bash
cd Quantum-Reservior-Computing-Quantathon2026
python3 -m venv .venv
.venv/bin/pip install -r webapp/requirements.txt
.venv/bin/pip install scikit-learn stumpy      # anomaly-forecast line
```

`fetch_cp_data.py` also uses `pandas`, `numpy` (already in the venv) and, on macOS
python.org builds, `certifi` for TLS — it falls back to `curl` automatically if
`certifi` is missing, so no extra install is required.

If a `new_data/` archive is present at the repo root (see `new_data/README.md`),
`fetch_cp_data.py` reads `got_sst`, `got_hadisst`, and `maeklong_spei` directly
from the local NetCDFs instead of hitting a remote endpoint — add
`.venv/bin/pip install netCDF4` for that path. Without `new_data/`, `got_sst`
still works via the ERDDAP year-by-year fetch; `got_hadisst`/`maeklong_spei`
have no such fallback and raise a clear error if their archive is missing.

**Gotcha (from the root README):** use the repo-root `.venv`. The system `python3`
has NumPy but not matplotlib; the Homebrew one is a different interpreter. Mixing
them causes confusing import errors.

---

## What was done — in detail

### 1. Use-case selection (the reasoning, so it can be defended)

The repo's own evidence decided this, not preference:

- On **`tao`** (daily SST, N≈1684) the quantum reservoir (`xxz_hx`) beats a
  size-matched classical ESN at **all five seeds** across the horizon; the ESN is
  *unstable* there (one seed → NMSE 1.83, worse than the mean predictor).
- On **`nino34`** (monthly ENSO, N=918) the ESN is at parity or ahead.
- ⇒ The defensible product cadence is **daily SST**, which maps onto an
  aquaculture early-warning use case (a heat/salinity anomaly can end a grow-out
  cycle; today's detection gives zero warning time). The pitch **concedes the ENSO
  case openly** — honesty is the selling point. No quantum-advantage claim at
  8 qubits / exact simulation.

### 2. CP-relevant datasets (prove generality to the customer's waters)

New reproducible fetcher: **`Quantathon_stack/Data/fetch_cp_data.py`**.
New registry keys added to **`Quantathon_stack/DataBase_Analysis/dataloader.py`**.

| Key | Record | Source | Cadence | Span | Status |
|---|---|---|---|---|---|
| `got_sst` | Gulf of Thailand SST, off the Mae Klong mouth (13.125N 100.125E) | NOAA OISST v2.1 | daily, 0.25° | 1981-09→2026-07 (16394 d) | ✅ done |
| `got_ersst` | Gulf of Thailand SST reconstruction (12N 100E) | NOAA ERSSTv5 | monthly, 2° | 1854→2026 (2070 mo) | ✅ done |
| `oni` | Oceanic Niño Index (3-mo running mean Niño-3.4) | NOAA CPC | monthly | 1950→2026 (917 mo) | ✅ done |
| `maeklong_rain` | Mae Klong basin rainfall (13.75N 99.75E) | NASA POWER (MERRA-2) | daily | 1981→2026 (16640 d) | ✅ done |
| `got_hadisst` | Gulf of Thailand SST, independent cross-check (13.5N 100.5E) | UK Met Office HadISST1 | monthly, 1° | 1870→2026 (1877 mo) | ✅ done (added from `new_data/`) |
| `maeklong_spei` | Mae Klong basin SPEI-03 (drought/salinity-risk index) | SPEIbase v2.x | monthly, 0.5° | 1901→2015 (1380 mo) | ✅ done (added from `new_data/`; vintage stops 2015) |

All six are open, credential-free. Verified loading through `dataloader.load(key)`:
ERSST warm-pool SST 26–31 °C; ONI −2.0…+2.8; rain 0–109 mm/day; HadISST1
27–31 °C at the Gulf point; SPEI-03 in the usual ±2 standardised range.

**`got_sst` was finished by direct extraction, not the ERDDAP fetch.** The
ERDDAP aggregate stores one NetCDF file per day, so a full-range point request
times out server-side (~1 year/minute) — `fetch_cp_data.py` originally pulled
it year by year to work around that. A `new_data/oisst_v2.1/` archive (46
yearly global NetCDFs, same NOAA OISST v2.1 source) showed up locally on
2026-07-23 (see `new_data/README.md`), so `fetch_cp_data.py` now reads the
Gulf point directly out of those files when the archive is present — same
series, much faster, and it also unblocked `got_hadisst`/`maeklong_spei`
(HadISST1 and SPEIbase, extracted from that same drop, extending generality
with an independent SST cross-check and a drought/salinity-risk covariate).
The ERDDAP path is kept as the fallback for a clone without `new_data/`.

### 3. Presentation deliverables (`presentation/`)

- **`deck.html`** — 8-slide pitch deck. Ocean theme, light/dark aware, scroll-snap +
  keyboard nav, a hand-built CVD-validated SVG skill-decay chart using the real TAO
  numbers from `Anomaly_Forecast/results/step2_tao.json`. Published as a private
  Artifact:
  **https://claude.ai/code/artifact/cc13ebf5-c756-46f9-b05b-a18b5c71c59b**
  (the original 2026-07-23 link was deleted/unwritable by the time of the
  Gulf-SST update, so this is a fresh URL as of that update — re-publish with
  this URL as `url` to update it further).
- **`speaker_notes.md`** — slide-by-slide talking points, full anticipated Q&A, and
  a reproducible numbers cheat-sheet.
- **`README.md`** / **`TODO.md`** — this handoff and the next-steps list.

**Deck honesty invariant (do not break):** the *proof* slide (5) is TAO SST
(robust evidence, deterministic reservoir kind, 5/5 seeds). Slide 6 now also
carries a **measured** Gulf-SST result — full 45-year record, 5 seeds, real
skill, materially beats persistence day 12→24, never materially loses to the
size-matched ESN — but stated with its own two honest caveats every time: the
winning kind (`ising`) is itself seed-dependent (per-seed win rate over the ESN
is ~3/5, not 5/5), and a real climatology floor caps the usable lead at ~day 20,
not day 24. No fabricated numbers anywhere. Keep the "no quantum advantage at
8 qubits" concession — it is what makes the pitch credible.

---

## How the Gulf-SST download got finished, and what ran next

A `new_data/` raw-archive drop (see `new_data/README.md`) landed locally on
2026-07-23, containing the full 46-year OISST v2.1 NetCDF archive plus HadISST1
and SPEIbase. `fetch_cp_data.py` now extracts `got_sst`/`got_hadisst`/
`maeklong_spei` directly from those files instead of the slow ERDDAP fetch:

```bash
cd Quantathon_stack/Data
../../.venv/bin/python fetch_cp_data.py   # extracts got_sst/got_hadisst/maeklong_spei
                                           # from new_data/ if present; ERDDAP otherwise
cd ../DataBase_Analysis
../../.venv/bin/python -c "from dataloader import load; print(load('got_sst'))"
```

Then the Step-2 kill-test, multi-seed (a single-seed `run_step2.py` run is not
enough here — see `TODO.md`'s guardrails on why):

```bash
cd ../Anomaly_Forecast
../../.venv/bin/python seed_sweep.py --dataset got_sst --horizon 24 --max-points 16394
# -> results/step2_seedsweep_got_sst.json ; PASS, see TODO.md for the full readout
```

`.venv/bin/pip install netCDF4` is needed for the `new_data/`-backed extraction
paths only (not for anything else in the repo).

---

## Files changed / added on 2026-07-23

```
Quantathon_stack/Data/fetch_cp_data.py                            (edited: local-archive fast path + 2 new records)
Quantathon_stack/Data/real/ersst_gulf_thailand_monthly_sst.csv    (new, earlier same day)
Quantathon_stack/Data/real/oni_index.csv                          (new, earlier same day)
Quantathon_stack/Data/real/nasa_power_mae_klong_daily_rain.csv    (new, earlier same day)
Quantathon_stack/Data/real/oisst_gulf_thailand_daily_sst.csv      (new: full record, from new_data/)
Quantathon_stack/Data/real/hadisst_gulf_thailand_monthly_sst.csv  (new: from new_data/)
Quantathon_stack/Data/real/spei03_mae_klong_monthly.csv           (new: from new_data/)
Quantathon_stack/DataBase_Analysis/dataloader.py                  (edited: 6 CP keys total, +2 today)
Quantathon_stack/Anomaly_Forecast/seed_sweep.py                   (new: generic multi-seed Step-2 sweep)
Quantathon_stack/Anomaly_Forecast/results/step2_seedsweep_got_sst.json  (new: the Gulf-SST result)
Quantathon_stack/Anomaly_Forecast/results/step2_got_sst.json,
  step2_skill_got_sst.png                                         (new: single-seed=7 view)
new_data/README.md                                                (rewritten: what's wired in / not, and why)
.gitignore                                                        (edited: new_data/* ignored, README kept)
presentation/deck.html                                            (edited: slide 6 measured-result panel)
presentation/speaker_notes.md                                    (edited: slide 6 + Q&A + numbers table)
presentation/README.md                                           (this file, edited)
presentation/TODO.md                                              (edited: moved items to Done)
```

See `presentation/TODO.md` for what to do next.
