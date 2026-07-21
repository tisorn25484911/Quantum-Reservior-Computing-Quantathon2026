# QRC forecasting demo

A judge-facing web demo for the Quantathon project: drive a real time series
through the project's exactly-simulated quantum reservoir, read out a forecast
with a calibrated uncertainty band, and score it against the baselines that
decide whether the result means anything.

FastAPI + Jinja + HTMX. No build step, no node toolchain, no broker.

## Running it

```bash
# from the repository root
python3 -m venv .venv
.venv/bin/pip install -r webapp/requirements.txt

cd webapp
../.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000>. Interactive API docs are at `/docs`.

> **Environment note.** This repository has several Python installations and
> only some of them carry `matplotlib`/`pandas`. The venv created above is the
> one the app expects; running `uvicorn` from a different interpreter will fail
> on imports. `qutip` is *not* needed — the reservoir used here is pure NumPy.

## What it does

| Page | What it is for |
|---|---|
| `/` | Pick a featured series and launch a run. |
| `/datasets` | All 17 series with provenance tier, sample count, lag-1 autocorrelation. |
| `/datasets/{key}` | One series: preview plot, statistics, and whether it is valid as a memory-capacity drive. |
| `/forecast` | Full configuration: horizon, qubits, input interval, virtual nodes, band level, seed. |
| `/runs/{id}` | Results: forecast figure, metric table against both baselines, model card. |
| `/chaos` | **Predictability**: λ₁, Lyapunov time, `H_max`, divergence curve, spectrum. |
| `/anomaly` | **Forecast-then-Detect**: recursive rollout and its skill decay. |
| `/diagnostics` | Memory capacity of the configured reservoir, on an i.i.d. drive. |
| `/about` | The method, and what these numbers are and are not allowed to claim. |

### Predictability (`/chaos`) and comparison (`/compare`)

**These pages call `DataBase_Analysis` rather than re-implementing it.**
`app/domain/chaos.py` is a thin adapter over `analysis.analyse_one()`,
`visualizer.figure()`, `provenance.DOCS`, and `compare.collect()/figure()`, so the
web pages and `python analysis.py --dataset <key>` agree by construction — same
numbers, same 4-panel figure, same per-dataset axis limits (`FMAX`, `MAX_UNITS`).

That delegation is load-bearing, not stylistic. An earlier version called
`fourier.dominant_periods` directly and omitted the `fmin = 2/(N·dt)` guard that
`analyse_one` applies before picking peaks. Measured across all 17 datasets,
omitting it changes the top three periods on **five** (`lorenz63`, `rikitake`,
`tao`, `nyc`, `cuxhaven`) — `nyc` reports a 57,540-day "dominant period" on a much
shorter record, which is pure near-DC leakage.

The question these pages answer: how far ahead can a series be predicted *at all*,
given the dynamics destroy information at rate λ₁? The ceiling is
`H_max = (1/λ₁)/dt`. Skill surviving well past it indicates leakage; skill dying
far short means the model, not the physics, is the limit.

`/compare` runs the full cross-dataset pass in the background (a Lyapunov fit per
dataset, ~2 min for all 17) and renders `comparison.png` and `spectra_real.png`
from `compare.py`, plus the comparison table.

Both pages also surface `provenance.DOCS` — source, observable, meaning, and the
per-record `watch_out` gotchas — on the dataset and predictability pages.

The page states how much the estimate can be trusted, because it varies by tier:

- **simulated chaotic** — ensemble estimate from independent realizations, and
  checkable against a published λ₁. `lorenz63` gives 0.823 vs published 0.906
  (−9%), which is the accuracy anchor for the estimator.
- **real / surrogate** — single-trajectory Rosenstein, reported as an *upper
  bound*. On a strongly seasonal record it is inflated (it reads the cycle as
  divergence), and the page marks those "unreliable (seasonal)".

### Anomaly forecast (`/anomaly`)

Wraps `Anomaly_Forecast/forecast.py` (plan.md Steps 1–2): recursive rollout where
the reservoir is driven by **its own output** after step 1, so errors compound.
Reports NMSE against lead time versus persistence and a size-matched ESN, the
usable lead, an example trajectory, and the encoding-clip count.

**Only the forecast half exists.** Steps 3–8 of that plan — stochastic ensembles,
matrix-profile / isolation-forest scorers, EVT thresholds, event debouncing,
injected ground truth — are not implemented, so **no anomaly probability is
shown**. A probability from uncalibrated machinery would be the most misleading
number this app could display; the page lists exactly what is missing.

Long runs execute on a worker thread; the page polls a progress partial and
reloads when finished. Completed runs are written to `webapp/results/<id>.json`
and survive a restart.

## The engine is not a mock

The reservoir is `QRC_code_stack/stage1_numpy_core/qrc_core.py` — the exact
density-matrix simulation that repository law R2 designates as the semantic
definition for the whole programme. Driving it from this app runs the same
propagator the handbook's numbers came from. It is classical *simulation of* a
quantum system; there is no quantum hardware in the loop and the UI says so on
every page.

## Three guards, deliberately built in

A demo is the easiest place in a project to overclaim, so these are structural
rather than optional:

1. **Leakage guard.** Scaling constants are fitted on the training span only,
   never the full series. Splits are chronological and never shuffled.
2. **Size-matched baseline.** An echo-state network gets exactly as many nodes
   as the reservoir has read-out features, so both train the same number of
   weights, and it is scored on identical rows. Persistence is shown alongside.
   **When a baseline wins, the run page leads with that.**
3. **Measured coverage.** Bands are split-conformal, calibrated on a slice the
   read-out never saw; realised coverage is measured on the test split and
   displayed against the nominal level.

Verified honest in practice: `tao` at horizon 2 reports *"Reservoir learned, but
does not beat persistence"* with −39% skill and an under-covering band, rather
than presenting a 0.507 NRMSE as a success.

## Memory capacity is measured separately, on purpose

`/diagnostics` drives the reservoir with i.i.d. uniform noise, **not** a
dataset. Memory capacity is only defined for an i.i.d. drive — given an
autocorrelated series a memoryless read-out can infer past inputs from the
present one, and the number then measures the smoothness of the data rather
than the memory of the reservoir. The project's real series have lag-1
autocorrelations of 0.95–0.99, so feeding them to that metric would produce a
large and meaningless value.

The measurement itself is
`Quantathon_stack/Main_run_Evaluation/memory_capacity.py`.

## JSON API

```
GET  /api/health
GET  /api/datasets[?tier=real|surrogate|chaotic]
GET  /api/datasets/{key}
POST /api/runs        {"kind":"forecast|sweep|memory", "dataset":"solar", ...}
GET  /api/runs[?kind=&limit=]
GET  /api/runs/{id}
```

`POST /api/runs` returns `202` with a job id; poll `GET /api/runs/{id}` until
`status` is `done`, at which point the payload carries the full `result`.

## Layout

```
webapp/
  app/
    config.py            paths, provenance vocabulary, sys.path wiring
    main.py              routes: HTML pages, charts, JSON API
    domain/
      catalog.py         dataset discovery and access
      engine.py          forecast / horizon sweep / memory capacity
      metrics.py         scoring, with reference lines
      baselines           (persistence + ESN, inside engine.py)
      charts.py          server-rendered PNGs
      jobs.py            background job registry
    templates/           Jinja pages
    static/              app.css, htmx.min.js (vendored, no CDN)
  results/               finished run payloads (gitignored)
```

## Known limitations

- **Single process.** In-flight jobs die with the server; finished ones persist.
- **One seed, one split per run.** Differences of a few percent between methods
  are not significant, and the app declines to rank them when they are that
  close.
- **Reservoir size is capped at 8 qubits** because the simulation is exact and
  costs ~4^n.
