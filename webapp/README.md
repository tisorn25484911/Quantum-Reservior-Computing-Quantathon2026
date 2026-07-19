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
| `/diagnostics` | Memory capacity of the configured reservoir, on an i.i.d. drive. |
| `/about` | The method, and what these numbers are and are not allowed to claim. |

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
