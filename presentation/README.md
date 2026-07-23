# CP Aquaculture Use Case — Handoff README

**What this is:** the product/use-case layer added on top of the QRC
Forecast-then-Detect line — a marine anomaly-forecasting pitch (working name
**TideRead**) aimed at CP-scale shrimp/marine aquaculture in the Gulf of Thailand,
plus the company-relevant datasets that prove the pipeline generalises to the
customer's own waters.

**Read this first if you are picking the project up on another machine.** It records
exactly what was done on 2026-07-23, what is still running, and how to resume.

---

## TL;DR — the state in five lines

1. **Use case chosen:** daily-SST marine-heat / salinity anomaly *forecasting* for
   Gulf-of-Thailand aquaculture. Picked because the QRC beats the classical ESN
   robustly only on **daily SST** (`tao`, 5/5 seeds), not on monthly ENSO.
2. **Four CP-relevant datasets acquired** and wired into the pipeline (3 done, 1
   still downloading — see below).
3. **Deck built & published** as an Artifact: `presentation/deck.html`.
4. **Speaker notes + Q&A** written: `presentation/speaker_notes.md`.
5. **One download still running:** Gulf of Thailand *daily* SST (OISST). Resume by
   re-running one command — it picks up from cached chunks.

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
| `got_sst` | Gulf of Thailand SST, off the Mae Klong mouth (13.125N 100.125E) | NOAA OISST v2.1 (CoastWatch ERDDAP) | daily, 0.25° | 1982→2026 | ⏳ **downloading** |
| `got_ersst` | Gulf of Thailand SST reconstruction (12N 100E) | NOAA ERSSTv5 | monthly, 2° | 1854→2026 (2070 mo) | ✅ done |
| `oni` | Oceanic Niño Index (3-mo running mean Niño-3.4) | NOAA CPC | monthly | 1950→2026 (917 mo) | ✅ done |
| `maeklong_rain` | Mae Klong basin rainfall (13.75N 99.75E) | NASA POWER (MERRA-2) | daily | 1981→2026 (16640 d) | ✅ done |

All four are open, credential-free. Verified loading through `dataloader.load(key)`:
ERSST warm-pool SST 26–31 °C; ONI −2.0…+2.8; rain 0–109 mm/day.

**Why OISST daily is special / slow:** the ERDDAP aggregate stores one NetCDF file
per day, so a full-range point request times out server-side (~1 year/minute). The
fetcher therefore pulls **year by year**, caching each year under
`Quantathon_stack/Data/real/_oisst_chunks/`, and assembles the final CSV when the
loop finishes. Safe to interrupt and resume — cached years are not re-fetched.

### 3. Presentation deliverables (`presentation/`)

- **`deck.html`** — 8-slide pitch deck. Ocean theme, light/dark aware, scroll-snap +
  keyboard nav, a hand-built CVD-validated SVG skill-decay chart using the real TAO
  numbers from `Anomaly_Forecast/results/step2_tao.json`. Published as a private
  Artifact:
  **https://claude.ai/code/artifact/34279f95-2114-4861-8dc5-784dc4a77590**
  (to update it from another session, re-publish with that URL as `url`).
- **`speaker_notes.md`** — slide-by-slide talking points, full anticipated Q&A, and
  a reproducible numbers cheat-sheet.
- **`README.md`** / **`TODO.md`** — this handoff and the next-steps list.

**Deck honesty invariant (do not break):** the *proof* slide is TAO SST (robust
evidence). The four CP datasets are framed as **acquired and pipeline-ready — with
Gulf-SST forecast results as the explicit NEXT step, not a done result.** No
fabricated numbers on Gulf data. Keep the "no quantum advantage at 8 qubits"
concession — it is what makes the pitch credible.

---

## How to resume the unfinished download

```bash
cd Quantathon_stack/Data
../../.venv/bin/python fetch_cp_data.py        # resumes OISST from cached chunks
```

When it prints `wrote oisst_gulf_thailand_daily_sst.csv`, the daily Gulf SST record
is complete. Confirm it loads:

```bash
cd ../DataBase_Analysis
../../.venv/bin/python -c "from dataloader import load; print(load('got_sst'))"
```

---

## Files changed / added on 2026-07-23

```
Quantathon_stack/Data/fetch_cp_data.py                       (new)
Quantathon_stack/Data/real/ersst_gulf_thailand_monthly_sst.csv   (new)
Quantathon_stack/Data/real/oni_index.csv                     (new)
Quantathon_stack/Data/real/nasa_power_mae_klong_daily_rain.csv   (new)
Quantathon_stack/Data/real/oisst_gulf_thailand_daily_sst.csv (pending download)
Quantathon_stack/Data/real/_oisst_chunks/                    (intermediate cache — gitignore)
Quantathon_stack/DataBase_Analysis/dataloader.py             (edited: 4 new keys)
presentation/deck.html                                        (new)
presentation/speaker_notes.md                                (new)
presentation/README.md                                        (new — this file)
presentation/TODO.md                                          (new)
```

See `presentation/TODO.md` for what to do next.
