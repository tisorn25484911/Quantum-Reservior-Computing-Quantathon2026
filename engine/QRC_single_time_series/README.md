# QRC_single_time_series

Quantum Reservoir Computing for **single-series** climate-index forecasting
(ENSO / PDO / SOI), implemented three ways — exact matrix, ideal Qiskit, noisy
Aer simulator — and evaluated in two strictly separated regimes: **teacher-forced
one-step** and **fully autonomous closed-loop**. Central question: *after the
true observations are withheld, how long can a trained QRC forecast by feeding
its own predictions back before leaving a pre-registered error band?*

This is a self-contained research project. **Everything lives under this folder.**
The build order is `IMPLEMENTATION_PHASES.md` (13 phases); the standing plan is
`PLAN.md`; source-text errata and contracts are in `IMPLEMENTATION_NOTES.md`.

> Scope discipline (spec §§2, 4, 25): no quantum-advantage, commercial-readiness,
> or renewable-energy-value claim without a dedicated experiment. The synthetic
> Vallis "ENSO" ODE and the observational `enso.csv` never share a table.

## Status

| Phase | Name | State |
|---|---|---|
| **P0** | Repo/data audit, scaffold, PLAN, preregistration | ✅ complete — `handoffs/HANDOFF_P0.md` |
| P1 | Exact operator core + anchors | pending confirmation |
| P2 | Exact FN reservoir, features, readout, metrics | pending |
| P3 | Trotter mirror + Qiskit ladder (L1–L5) | pending |
| P4 | Autonomous engine, horizons, failure taxonomy | pending |
| P5 | Mackey–Glass reproduction (**GATE**) | pending |
| P6 | Rewind QRC + learned-map Lyapunov | pending |
| P7 | Baseline battery + controls | pending |
| P8 | Controlled-system campaign + dynamical audits | pending |
| P9 | Observational climate campaign | pending |
| P10 | Shots, noise, feature space | pending |
| P11 | Sequential QPU feasibility | pending |
| P12 | Product track, report, acceptance | pending |

The project advances **one phase at a time**, with a written handoff and a
user confirmation between phases.

## Data (Phase-0 audit)

Three real monthly indices, pinned + checksummed in `data/manifests/`:

| file | identity (audited) | span | anomaly? |
|---|---|---|---|
| `enso.csv` | NOAA **Niño 1+2** SST (statsmodels `elnino`), **raw °C** — *not* ONI/Niño3.4/MEI | 1950–2010 | no (raw SST) |
| `pdo.csv` | NOAA PSL **Mantua-style PDO** index | 1948–2025 | yes |
| `soi.csv` | NOAA CPC **SOI** = Tahiti−Darwin SLP **anomaly** (not Troup-standardised) | 1951–2026 | yes |

Chronology is frozen in `data/loaders.py`: **final 20% is an untouched test
span**; rolling-origin validation inside the first 80%. Climate targets are
modelled as anomalies vs a **training-years-only** climatology; every scaler is
train-only.

## Quick start

```bash
PY=../../QRC_main_stack/QRC_code_stack/.venv/bin/python   # pinned interpreter
make audit     # (re)write data manifests + repo inventory
make test      # fast test suite (Phase-0 green; later anchors as they land)
make help      # list all targets; large grids are explicit, never in `make all`
```

Environment is pinned in `requirements-lock.txt` (qiskit 2.5, aer 0.17,
statsmodels, pennylane; `torch` is an optional Phase-7 extra). `utils/resources.py`
logs versions + git hash into every run.

## Layout

`configs/` frozen YAML (incl. `preregistration.yaml`, locked in P0) ·
`src/qrc_single_time_series/` the package (`data/ quantum/ models/ training/
evaluation/ product/ hardware/ utils/`) · `scripts/` one entry point per
experiment group · `tests/` anchors (G2/G3/G5 contracts) · `results/` all
outputs · `notebooks/` narrative (built per phase) · `handoffs/` per-phase
handoff docs.
