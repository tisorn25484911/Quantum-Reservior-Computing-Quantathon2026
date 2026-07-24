# HANDOFF_P9 — Observational climate campaign (enso / pdo / soi)

Written per G10. In-repo (declared deviation, as P0–P8).

## 1. Objective & status vs exit criteria

| Exit criterion | Status | Evidence |
|---|---|---|
| Train-only EDA (stationarity, ACF, spectrum, season) | ✅ | `data/eda.py`, per-index `eda` block |
| Teacher-forced direct multi-horizon h∈{1,3,6,12} on test span | ✅ | `run_one_step_forecasts.py::_qrc_direct` |
| Preregistered Gate-1 evaluated on the untouched test span | ✅ | block-bootstrap 90% CI, `evaluation/statistics.skill_bootstrap_ci` |
| Autonomous H_effective (Gate-3 input) per index | ✅ | `_autonomous` (20 origins, ε=0.5) |
| Launch-month skill (spring barrier) | ✅ | `_launch_month_skill` (h=3) |
| Hybrid-readout ablation on ENSO (G7) | ✅ | quantum vs classical-lags vs hybrid |
| Synthetic Vallis and observational ENSO never share a table (item 21) | ✅ | separate scripts/JSONs |
| Suite passes | ✅ | **223 passed, 1 skipped** |

## 2. Artifacts & keys

- `data/eda.py` — `decorrelation_time` (sets the bootstrap block), `dominant_period`,
  `seasonal_strength`, `eda_report` (ADF + KPSS via statsmodels, ACF/PACF).
- `evaluation/statistics.py::skill_bootstrap_ci` — paired stationary block bootstrap
  of the persistence skill score (the Gate-1 statistic).
- `scripts/run_one_step_forecasts.py` (`make climate`) → `results/metrics/climate_campaign.json`.
- Tests: `tests/test_climate_campaign.py` (5).

## 3. Key results (untouched test span; QRC N=5, V=6)

| index | h=1 | h=3 | h=6 | h=12 | Gate-1 | genuine skill (NMSE<1) |
|---|---|---|---|---|---|---|
| **ENSO** skill/NMSE | +0.07 / 0.38 | +0.25 / **1.13** | +0.42 / 1.33 | +0.67 / 1.19 | **PASS** | only **h=1** |
| **PDO** skill/NMSE | +0.00 / 0.16 | +0.08 / 0.41 | −0.11 / 0.65 | −0.40 / 0.85 | **fail** | h=1,3,6,12 |
| **SOI** skill/NMSE | +0.29 / 0.63 | +0.32 / 0.80 | +0.36 / 0.92 | +0.46 / **1.03** | **PASS** | h=1,3,6 |

**The leakage-hunt finding (headline honesty result).** ENSO's *rising* persistence
skill with lead is **not** genuine 12-month predictability and **not** leakage: the
absolute test NMSE is **> 1 (worse than climatology) at h ≥ 3**, so the positive
"skill" only reflects persistence collapsing faster than the model's reversion to
climatology. A leak would drive NMSE far *below* 1; NMSE>1 rules it out. Genuine
absolute skill (NMSE<1) exists **only at h=1 for ENSO**. Every horizon now carries a
`beats_climatology` flag so this can never be misread.

**Priors replicated exactly:**
- **PDO** beats climatology at *every* lead (strongly red) yet **fails Gate-1** — it
  cannot beat *persistence*, the hard baseline for a red series (spec §9 prior).
- **SOI** beats persistence easily (persistence is weak for noisy SOI) — genuine
  absolute skill out to h=6.
- **ENSO** univariate index-only skill is weak (1 month) — the realistic weak-end
  regime; >12-month skill would have triggered a hunt, and the hunt was run.

**Hybrid-readout ablation (ENSO, one-step test NMSE):** quantum-only **0.383**,
classical-lags-only **0.375**, hybrid **0.370**. The quantum feature map adds
**nothing** over six classical lags here — a clean G7 negative (parity, not edge).

**Autonomous (ε=0.5):** ENSO median H_eff 4 mo, survival@3 0.70; PDO 3 mo, 0.50;
SOI 4 mo, 0.55. **Gate-3** (H_eff≥3 mo with survival≥0.8) is **not** met by any index
(survival<0.8) — reported, not massaged.

## 4. Decisions & deviations

- **`beats_climatology` flag added** to every horizon and to the Gate-1 block so the
  persistence-vs-climatology distinction is explicit — Gate-1 as preregistered keys
  on persistence skill, but genuine skill claims key on NMSE<1 (spec: no skill claim
  without support). Preregistration itself is unchanged (not an amendment; an
  additional honesty diagnostic).
- **Direct multi-horizon** (one readout per h off a shared feature cache), the
  cheaper and leakage-safer route than recursive multi-step for the teacher-forced
  table; recursive/autonomous is scored separately.
- **Autonomous origins = 20** (short records); horizon 24 mo. Config knobs.
- KPSS emits a benign "p-value outside lookup table" InterpolationWarning (statsmodels).

## 5. Open issues / known gaps

- No index meets Gate-3 (autonomous survival<0.8 at 3 mo) with the small N=5 config;
  the J·τ / input-gain sweeps (preregistered, not yet run per index) may lift this —
  deferred to the P10 feature-space work / final tuning.
- Correlation-dimension / λ for the indices intentionally NOT computed (short, noisy,
  not closed deterministic attractors — spec §18.2).
- Neural baselines absent from the climate table (torch optional extra not installed).

## 6. Entry instructions for Phase 10 (shots, noise, feature space)

1. `make test` green (223 passed, 1 skipped).
2. Noise grids on the **rewind** model (hardware-shaped): synthetic channels
   (readout p, depol 1q/2q, thermal) + the frozen backend snapshot; shot grid per G6.
   `quantum/noise_models.py` + `scripts/export_backend_noise_model.py` exist (P3).
3. `evaluation/diagnostics.py` feature-space suite (spec §27): bias/var/RMSE vs exact
   features, covariance, cosine, singular values, effective rank (1/HHI, Hamhoum
   Eq. 17), condition number, readout-coefficient stability.
4. **Three-arm adjudication** noise-vs-matched-ridge-vs-SVD-truncation; a "noise
   helps" claim only if it survives all three axes AND beats the matched classical
   regulariser. SVD-truncation à la Ahmed–Tennie–Magri is the mitigation arm.
5. The **signal-to-noise ceiling check gates the whole phase** (P8 found the N=5 MG
   shot-slope unmeasurable — use a stronger-signal config here).
6. `scripts/{compare_shot_budgets,compare_noise_models,run_noisy_qrc}.py`
   (`make grid-shots`, `make grid-noise`); write `HANDOFF_P10.md`.

## 7. Suggested skills

`quantum-reservoir-computing` + `qrc-project-playbook` (min per G10); add
`citation-audit` when Ahmed–Tennie–Magri / Domingo et al. enter `references.bib`.
