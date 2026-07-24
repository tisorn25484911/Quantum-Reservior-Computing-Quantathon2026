# Combined Plan — QRC anomaly *prediction* for aquaculture

Reconciles two parallel repos into one project. Written 2026-07-24.

- **This repo** (`Quantum-Reservior-Computing-Quantathon2026`) — a real-domain
  product line: CP-scale Gulf-of-Thailand aquaculture, real driver data
  (Gulf SST, ERSST, ONI, rainfall, HadISST, SPEI), a *Forecast-then-Detect*
  pipeline, and now **labelled anomalies** (marine heatwaves + ENSO episodes).
- **Linked repo** [`Quantathon2026-QRC_detection`](https://github.com/tisorn25484911/Quantathon2026-QRC_detection)
  → `QRC_single_time_series/` — a rigorous QRC forecasting science project:
  preregistration, a Mackey–Glass validation gate, three faithful
  implementations (exact / Qiskit / noisy Aer), a shot-noise-compounding study,
  a full baseline battery, and a precise **autonomous-horizon taxonomy**.

## The key realisation: they are two halves of one project

The linked repo's own `PRODUCT_STRATEGY.md` (§25.3) states the thing it does
**not** build:

> *"Climate-index forecast accuracy alone is NOT a product; a second-stage
> experiment linking predicted indices to an operational target is required."*

**That missing second stage is exactly what this repo is.** The marine-heatwave
detector on labelled events, tied to a named aquaculture decision, is the
operational target the linked repo says a product needs. So the recommendation
is **combine, not choose**:

| Layer | Owner | What it contributes |
|---|---|---|
| QRC forecasting **rigor** | linked repo | preregistration, Mackey–Glass gate, exact/Qiskit/noisy parity, shot-noise horizon law, baseline battery, autonomous-horizon taxonomy |
| Real **domain + data** | this repo | Gulf-of-Thailand drivers, marine-heatwave & ENSO **labels**, the CP use case |
| The **second-stage** operational experiment | this repo | Forecast-then-Detect: run a detector on the forecast → *predict* the anomaly |

## The project, restated (matches the user's two-step framing)

```
STEP 1  QRC forecasts the driver anomaly, autonomously (closed-loop).
        Target = MHW intensity (SST - seasonal climatology), NOT raw SST.
        Validated on its own train/val/test as a FORECAST-SKILL question.

STEP 2  A detector (ML/DL) learns marine-heatwave events from LABELLED data,
        validated on its own train/val/test as a DETECTION question.

COMPOSE Run the Step-2 detector on the Step-1 forecast -> predicted anomalies,
        with warning time = the forecast horizon.
```

### Why forecast the anomaly, not raw SST (a result from this session)

On **raw** daily SST, persistence is nearly unbeatable (the ocean barely moves
day to day), so the QRC only ties it. On the **deseasonalised MHW intensity**,
persistence is weak and the **QRC beats it at every horizon**, with the margin
*widening* with lead time:

| lead h | QRC (ising) | persistence |
|---|---|---|
| 1 | 0.074 | 0.078 |
| 6 | 0.409 | 0.438 |
| 12 | **0.639** | 0.778 |

So Step 1's target is `got_sst_mhwi` (registered in `dataloader.py`), and this is
also the exact signal Step 2 thresholds — the two steps share one series.

### "Which range does the model suit?" — adopt the linked repo's taxonomy

Replace this repo's brittle single "useful lead" number with the linked repo's
`AUTONOMOUS_FORECASTING.md` horizon set, all reported together:

- **H_skill** — furthest lead with NMSE < 1 (beats the mean predictor).
- **H_beats-baseline** — furthest lead beating persistence / seasonal
  climatology by a material margin.
- **H_reliable(p)** — furthest lead where the calibrated band still holds
  (needs Step 3 stochastic rollout).
- **H_effective = min(...)** — the honest, reportable operating range.

On the Gulf SST this repo already measures H_skill = 24 d and
beats-climatology ≈ 14 d; the intensity target and H_reliable are next.

## Guardrails carried from BOTH repos

- **No quantum-advantage claim** at ≤8 qubits / exact simulation; parity vs a
  size-matched ESN is the ceiling (both repos agree; linked repo H3).
- **Multi-seed** always (this repo's `seed_sweep.py`; linked repo's per-origin
  block bootstrap) — one seed reversed a verdict once already.
- **Composition is not free** (linked repo §25.3; this repo `plan.md` §5): a
  detector trained on *real* data must be **re-calibrated on forecast-of-training**
  output before the composed alarm is trusted — forecasts are smoother and
  lower-variance than observations.
- **Operating-point tuning gave no robust gain** on raw Gulf SST (this session's
  `operating_point_sweep.py`, leakage-safe val/test) — a documented negative
  result; do not re-open it expecting free accuracy.

## Immediate next steps

1. **Step 1 on the anomaly target** — run `seed_sweep.py --dataset got_sst_mhwi`;
   report the full horizon taxonomy against persistence + climatology.
2. **Step 2 detector** — train on `labels/got_sst_mhw_labeled.csv`
   (features from the intensity series; labels = `mhw` / `category`), chronological
   train/val/test, range-based metrics (Tatbul et al.), never point-adjust F1.
3. **Adopt the linked repo's rigor incrementally** — at minimum the Mackey–Glass
   autonomous gate and the shot-noise horizon law, to make Step 1 defensible.
4. **Compose + re-calibrate** — detector on forecast-of-training, then measure
   hit-rate vs lead time on the injected/labelled events.

## Artifacts produced this session

- `Quantathon_stack/Anomaly_Forecast/label_anomalies.py` → `labels/`
  (marine-heatwave labels via Hobday 2016; ENSO episodes from ONI; a figure).
  Validated: hottest MHW = the 1997-98 El Niño; strongest El Niño = 2015-16.
- `Quantathon_stack/Anomaly_Forecast/operating_point_sweep.py` +
  `results/opsweep_got_sst.json` — the leakage-safe negative result.
- `got_sst_mhwi` registered in `dataloader.py` — the anomaly forecast target.
- Closed-loop rollout **proven** future-independent (destroying `x[origin+1:]`
  changes the forecast by 0.0).
