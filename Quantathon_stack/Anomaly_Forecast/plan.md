---
title: Forecast-then-Detect — anomaly prediction on QRC forecasts
created: 2026-07-21
updated: 2026-07-21
status: steps 1-2 verified
---

# Forecast-then-Detect

Run classical time-series anomaly detectors **on QRC forecast trajectories** instead of
on observed data, so anomalies are flagged *before* they occur.

```
y(1..t) observed  →  QRC recursive rollout, K stochastic samples
                  →  ŷ^(k)(t+1..t+H)
                  →  treat ŷ as if it were observed data
                  →  classical detectors (Blázquez-García et al. taxonomy)
                  →  P(anomaly at t+h)  +  ETA  +  driver
```

Existing pipeline direction is *detection* (needs `y(t)` to exist, zero warning time).
This adds a *prediction* branch (needs only the forecast, warning time up to `H`).

---

## 0. Decisions already made

| Question | Decision |
|---|---|
| Dependencies | install `scikit-learn` + `stumpy` into `.venv` |
| First target dataset | `nino34` (real climate tier) |
| Multi-step scheme | recursive rollout (feed prediction back as next input) |
| Broken `predict_timeseries.ipynb` | ignore; build fresh glue inside this folder |

---

## 1. What already exists and gets reused

Do not rewrite any of this.

| Asset | Path | What it gives |
|---|---|---|
| `QuantumReservoir` | `Hamiltonian_QRC/qrc_core.py` | `.run(inputs)` → feature matrix. `n_qubits`, `J`, `h`, `dt`, `virtual_nodes`, `seed` |
| `ridge_fit` / `ridge_predict` | `Hamiltonian_QRC/qrc_core.py` | closed-form readout `W_out = Y Hᵀ(HHᵀ+λI)⁻¹` |
| `ESN` | `Hamiltonian_QRC/qrc_core.py` | classical reservoir baseline — **mandatory comparison** |
| `nmse` | `Hamiltonian_QRC/qrc_core.py` | forecast metric |
| `add_shot_noise` | `Hamiltonian_QRC/qrc_core.py` | already a stochastic-sampling primitive; candidate for Step 3 |
| Hamiltonians | `Hamiltonian_QRC/Hamiltonians.py` | `ising`, `xxz_hx` |
| `load(key)` → `Series` | `DataBase_Analysis/dataloader.py` | `.x`, `.dt`, `.time_unit`, `.unit`, `.tier`, `.ensemble`, `.lyap_true`, `.index` |
| `list_datasets(tier)` | `DataBase_Analysis/dataloader.py` | keys per tier |
| `lyapunov_rosenstein`, `lyapunov_time`, `estimate` | `DataBase_Analysis/lyapunov.py` | λ₁ and `T_λ = 1/λ₁` → **the physical ceiling on lead time** |
| `linear_memory_capacity`, `effective_rank` | `Main_run_Evaluation/memory_capacity.py` | reservoir diagnostics if forecasts underperform |

### Known breakage (do not depend on)
`Hamiltonian_QRC/predict_timeseries.ipynb` is stale — imports `experiments` (exists
nowhere), `data_loader` (real name `dataloader`), `rosenstein_lambda` (real name
`lyapunov_rosenstein`). `make_reservoir` / `forecast_series` are **not defined anywhere
in the repo**. Left untouched by decision. This folder builds its own glue.

---

## 2. Target dataset — and an honest risk

`nino34` = Niño 3.4 SST anomaly, monthly, `dt = 1/12 yr`, N ≈ 918.

**Risk: N is small for extreme-value threshold fitting.**
POT/GPD needs ≈100+ tail exceedances for a stable `(γ, β)` fit. At N=918 a 2 % tail is
~18 points. Fit will be noisy; per-horizon thresholds `τ_h` make it worse (each `h`
gets its own tail).

Mitigations, in the plan:
1. Pool exceedances across neighbouring horizons when fitting `τ_h` (smooth in `h`).
2. Fall back to empirical quantile thresholds when `n_exceed < 30`; log which was used.
3. Carry a **long-N companion dataset** through every step for threshold sanity:
   `tao` (daily SST, same physical domain) or `opsd` (hourly load). Same code path,
   many more samples — if `τ_h` behaves there but not on `nino34`, the problem is N,
   not the method.
4. Ground truth comes from `lorenz63` injection (Step 6), not from `nino34`.

**Assumption to check in Step 1, not assume:** `nino34` is strongly seasonal/quasi-
periodic, so its Rosenstein λ₁ is likely inflated (the repo's own `lyapunov.py` docs
warn about this). If `T_λ` comes out implausibly short, treat it as scale, not
measurement, and set `H` from forecast-skill decay instead.

---

## 3. Folder layout

```
Anomaly_Forecast/
├── plan.md            ← this file
├── forecast.py        ← QRC glue: fit, 1-step, recursive rollout, K-sample rollout
├── scorers.py         ← s₁ residual, s₂ matrix profile, s₃ isolation forest
├── fuse.py            ← rank normalise + combine
├── threshold.py       ← POT/EVT per horizon + empirical fallback
├── events.py          ← debounce, merge, severity
├── inject.py          ← synthetic anomaly injection (ground truth)
├── evaluate.py        ← hit-rate vs horizon, FA/day, lead-time distribution
├── run_experiment.py  ← wires it all, writes results/
└── results/
```

Flat modules, no package scaffolding, no config system. Each file is one stage of the
pipeline and is independently testable.

---

## 4. Implementation steps

Each step has a **verify** gate. Do not start step *n+1* until *n* verifies.

### Step 1 — Baseline forecast API (`forecast.py`)

Replaces the missing `experiments.py`. Minimum surface:

```python
def make_reservoir(kind, J, hx, **kw)          # "ising" | "xxz_hx" | "esn"
def fit_readout(x, res, washout, train_frac, lam)   # → W_out, split indices
def forecast_1step(x, res, W_out)              # → yhat aligned to y
def rollout(x_hist, res, W_out, H)             # recursive: feed ŷ back in, H steps
```

Recursive rollout detail — the reservoir is driven by its **own output** after step 1:

```
h ← reservoir state after driving with x_hist
for i in 1..H:
    ŷ_i = W_out · h
    h   = reservoir.step(h, ŷ_i)      # own prediction becomes next input
```

`qrc_core.QuantumReservoir.run()` takes a whole input array; rollout needs
single-step advancement. **Open item:** either add a `step()` method to a subclass
here, or re-run `run()` on the growing sequence each iteration (O(H²), acceptable at
H≲50, N≲2000). Start with the O(H²) version — simplest thing that works; optimise only
if it is measurably slow.

**Verify:**
- `forecast_1step` on `nino34` reproduces an NMSE in the same ballpark as the existing
  notebook's reported numbers for `ising` / `xxz_hx`.
- `rollout(H=1)` output is identical to `forecast_1step` (same value, same index).
- ESN baseline runs through the identical code path.

### Step 2 — Predictability ceiling

Run `lyapunov_rosenstein` on `nino34` → λ₁, `T_λ`. Convert to steps:
`H_max = T_λ / dt`. Separately measure **empirical skill decay**: NMSE of `rollout`
vs horizon `h`, on held-out data, for QRC and ESN.

**Verify:** plot NMSE-vs-`h`. The `h` where NMSE crosses persistence-baseline NMSE is
the honest forecast horizon. Set `H` for everything downstream to that value.
Compare it to `H_max` — they should be the same order. If not, note the discrepancy;
do not silently pick the flattering one.

**Gate:** if QRC does not beat *persistence* and *ESN* on this curve, the whole product
premise fails. Stop and report. This is the cheapest possible kill-test — run it early.

### Step 3 — Stochastic rollout (the core idea)

Ridge minimises MSE → readout outputs `E[y|past]` → the mean trajectory is **smooth**
and systematically *less* anomalous than reality. A detector run on the mean forecast
will under-alarm. This is a property of the loss function, not a tuning issue.

Fix: sample `K` trajectories, detect on each, report the **fraction flagged**.

Sampling methods, cheapest first — implement (a), keep (b) as fallback:

- **(a) Residual bootstrap.** At each rollout step add a residual resampled from the
  training-residual pool: `ŷ_i ← W_out·h + ε`, `ε ~ empirical residuals`. Preserves the
  real error magnitude and its distribution shape. Cheap, no retraining.
- **(b) Readout ensemble.** Bootstrap the training rows, fit `K` different `W_out`.
  Ridge is closed-form, so this is cheap too. Captures parameter uncertainty, which (a)
  does not.
- **(c) Shot noise.** `qrc_core.add_shot_noise` already exists — physically motivated
  (finite measurement shots). Worth a look as a *third* channel, since it is free.

**Verify:** the K-sample ensemble is **calibrated** — over held-out windows, the actual
`y(t+h)` should fall inside the ensemble's 90 % interval ≈90 % of the time, at each
`h`. Plot empirical coverage vs nominal. If coverage is far below nominal, the ensemble
is over-confident and every downstream probability is wrong. Fix before proceeding.

### Step 4 — Scorers (`scorers.py`)

Three channels, each returning a score series over a trajectory:

| Channel | Function | Notes |
|---|---|---|
| `s₁` residual | `\|y − ŷ\| / (1.4826 · MAD)` | MAD not std — std is inflated by the anomalies themselves. Detection branch only (needs actual `y`). |
| `s₂` matrix profile | `stumpy.stump(x, m=W)[:,0]` | subsequence/shape anomalies. Do not reimplement. |
| `s₃` isolation forest | `-IsolationForest().score_samples(windows(x, W))` | feed **windowed** features — IF is not time-aware on its own |
| `s₄` spec breach | `1[ŷ ∉ [L,U]]` | only if `nino34` has agreed operating limits — **open item**, see §6 |

Window length `W`: derive from `dataloader`'s `dt` and the dominant period. For monthly
`nino34`, `W = 12` (one year) is the obvious first choice; sweep it.

**Verify:** each scorer, run on a series with one hand-placed synthetic spike, ranks
that spike in its own top-5 scores.

### Step 5 — Fuse + threshold + events

`fuse.py`
```python
S_i = rankdata(s_i) / len(s_i)     # rank → [0,1], distribution-free
S   = max_i S_i                    # OR-logic: each detector catches a different shape
```
Start with `max`. Weighted mean only after Step 8 provides labels.

`threshold.py` — **per horizon `h`**, since `h=1` and `h=H` have different error
distributions.
```python
u     = quantile(S, 0.98)
excess= S[S > u] - u
γ, β  = genpareto.fit(excess, floc=0)          # scipy, already installed
τ_h   = u + (β/γ) * ((q·N/n_excess)**(-γ) - 1)
```
with the `n_exceed < 30 → empirical quantile` fallback from §2.

**Critical calibration rule.** Fit thresholds on **forecast-of-training**, not on
training data. Backtest the QRC over the training period to produce `ŷ_train(t+h)` for
every `t, h`, and fit `τ_h` on scores computed from *those*. Forecast trajectories have
smaller variance and different spectral content than real data — a threshold fit on
real data applied to forecasts will essentially never fire.

`events.py` — turn jittery flags into incidents:
```
debounce   flag'(t) = 1[ Σ_{i=t-n+1}^{t} flag(i) ≥ m ]
merge      contiguous runs with gap ≤ g  →  {start, end, peak}
severity   (max S in event − τ) / τ
```

**Verify:** on the `nino34` detection branch with `q = 1e-3`, the realised false-alarm
rate over held-out data lands within ~2× of `q`. Event count per year is small enough
to read (single digits, not hundreds).

### Step 6 — Ground truth by injection (`inject.py`)

`nino34` has no anomaly labels. Manufacture them.

Injection types (apply to a clean copy, record `(start, end, type)`):
`spike`, `level shift`, `variance burst`, `flatline/freeze`, `subsequence swap`.

Run on **both** `lorenz63` (clean, known λ₁, ground-truth dynamics) and `nino34`.
`lorenz63` is the method validator; `nino34` is the target.

**Verify:** detection branch recovers injected anomalies at recall > 0.8 at a
fixed FA budget. If the *detection* branch cannot find anomalies it was handed, the
*prediction* branch has no chance.

### Step 7 — Prediction evaluation (`evaluate.py`)

This is the deliverable. Standard precision/recall on timestamps does **not** apply.

```
at each t:  rollout K samples → score each → threshold τ_h → p(h) = frac flagged
            predicted event set P_t
when actual arrives: run the SAME detector on real y → true event set A
```

Metrics, all reported **as a function of lead time `h`**:
- **hit rate(h)** — fraction of true events predicted `h` ahead
- **false alarms per year(h)**
- **lead-time distribution** — how early, in months and in Lyapunov times
- **Brier score** on `p(h)` — is the probability itself calibrated?

**The deliverable plot: hit-rate vs `h`.** The `h` at which it collapses is the product
limit. Cross-check it against `T_λ` from Step 2 — they should agree. If hit rate
collapses far earlier than `T_λ`, the QRC is underperforming physics, not hitting it.

**Baselines that must appear on the same plot:** persistence, ESN, and *climatology*
(seasonal mean) — `nino34` is seasonal, so climatology is a genuinely hard baseline.

**Verify:** every number reproducible from `run_experiment.py` with a fixed seed.

### Step 8 — Self-labelling loop (optional, only if Steps 1–7 land)

Prediction grades itself when the future arrives: compare `P_t` against `A`. No human
labeller needed. Those labels then retune `q` and turn `max`-fusion into weighted
fusion. Note it in the plan; do not build it yet.

---

## 5. Dependencies

```bash
"/Users/popsuksumetchoengprachya/Projects/QRC Climate Prediction/.venv/bin/pip" \
    install scikit-learn stumpy
```

Already present: `numpy`, `scipy` (has `genpareto`), `pandas`, `matplotlib`, `nolds`.
`stumpy` pulls `numba` transitively — first `stump()` call pays JIT compile cost.

---

## 6. Open items — decide when reached, do not guess

1. ~~**`step()` on the reservoir.**~~ **RESOLVED (Step 1).** The O(H²) route was not
   viable: it replays the whole history each step, so one H=24 rollout at origin *t*
   costs ~Σ(640+i) ≈ 15.7k reservoir steps ≈ 3.5 s, i.e. ~11 min per configuration over
   ~190 origins — before Step 3 multiplies by *K*. `forecast.py` adds `_QRCDriver` with
   `step()` + state checkpointing (O(H) per rollout, ~5 ms). Verified identical to
   `run()` to 1e-12, so it is a performance refactor, not a different model.
2. **Spec limits `[L, U]` for `nino34`.** Niño 3.4 does have conventional El Niño /
   La Niña thresholds (±0.5 °C anomaly, sustained). Using those makes `s₄` and the
   whole "predict a breach" story concrete and domain-meaningful — **confirm the exact
   convention before hard-coding it.**
3. ~~**Which Hamiltonian.**~~ **RESOLVED (Step 2): `xxz_hx`.** Lower mean NMSE than
   `ising` on both `nino34` and `tao`, and it degrades more gracefully (NMSE 1.33 vs
   1.72 at h=24 on `nino34`). `ising` never materially beats the ESN at any lead.
4. **`K` (ensemble size).** Start 200. Raise until `p(h)` is stable to ±0.02 between
   seeds.
5. ~~**Detrending.**~~ **RESOLVED (Step 2) — and it kills the climatology baseline.**
   `nino34_anom` is indeed already an anomaly series, so its seasonal mean is ≈0 and
   the climatology forecast collapses onto the mean predictor (flat NMSE ≈ 1.03,
   *worse* than persistence at short lead). Climatology is therefore **not a valid
   baseline on `nino34`** and must not be quoted as the "hard baseline" of §7.4 there.
   Either use raw Niño-3.4 SST (climatology intact) or drop the baseline for this
   series. No double-detrending occurs in `dataloader`.

---

## 7. Kill-tests, in order

Cheapest first. Each can end the project early and save the rest of the work.

1. **Step 2** — QRC beats persistence and ESN on multi-step NMSE. If not: no product.
2. **Step 3** — K-sample ensemble is calibrated. If not: all probabilities are wrong.
3. **Step 6** — detection branch finds injected anomalies. If not: prediction cannot.
4. **Step 7** — hit rate at useful `h` beats climatology. If not: no commercial claim.

---

## 8. References

**Primary — the taxonomy this builds on**
- Blázquez-García, Conde, Mori, Lozano. *A Review on Outlier/Anomaly Detection in Time
  Series Data.* ACM Computing Surveys 54(3), 2021.
  DOI [10.1145/3444690](https://doi.org/10.1145/3444690) ·
  open preprint [arXiv:2002.04236](https://arxiv.org/abs/2002.04236)
  → §2 taxonomy (point / subsequence / whole-series); §6 Table 9 software list.

**Thresholding**
- Siffer et al. *Anomaly Detection in Streams with Extreme Value Theory.* KDD 2017 —
  SPOT/DSPOT, the POT method used in Step 5. Code: <https://github.com/asiffer/libspot>
- Hundman et al. *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
  Thresholding.* KDD 2018 — closest published architecture to this plan (forecast →
  residual → dynamic threshold → FP pruning).
  Code: <https://github.com/khundman/telemanom>

**Scorers**
- Yeh et al. *Matrix Profile I.* ICDM 2016 — discord discovery.
  `stumpy`: <https://github.com/TDAmeritrade/stumpy>
- Liu, Ting, Zhou. *Isolation Forest.* ICDM 2008.

**Evaluation — read before reporting any number**
- Kim et al. *Towards a Rigorous Evaluation of Time-Series Anomaly Detection.* AAAI 2022
  — the point-adjust metric inflates F1 so badly a random detector scores ~0.9.
  **Never report point-adjust F1.**
- Wu & Keogh. *Current Time Series Anomaly Detection Benchmarks are Flawed.* TKDE 2021.
- Tatbul et al. *Precision and Recall for Time Series.* NeurIPS 2018 — range-based
  metrics, use these.
- Schmidl, Wenig, Papenbrock. *Anomaly Detection in Time Series: A Comprehensive
  Evaluation.* VLDB 2022 — TimeEval; the honest cross-method comparison.

**Chaotic-system forecasting and extreme events (the premise)**
- Pathak et al. *Model-Free Prediction of Large Spatiotemporally Chaotic Systems from
  Data: A Reservoir Computing Approach.* PRL 120, 024102 (2018).
- Farazmand & Sapsis. *A variational approach to probing extreme events in turbulent
  dynamical systems.* Science Advances 3, e1701533 (2017).
- Rosenstein, Collins, De Luca. *A practical method for calculating largest Lyapunov
  exponents from small data sets.* Physica D 65, 117 (1993) — the estimator already
  implemented in `lyapunov.py`.

**In-repo**
- `Quantathon_stack/Data/README.md` — dataset provenance
- `Quantathon_stack/DataBase_Analysis/analysis.md`, `comparison.md`
- `QRC_code_stack/HANDOFF.md`, `RUNBOOK.md`

---

## 9. Immediate next action

~~Install deps, then Step 1 + Step 2 only.~~ **Done — see §10.** Next action is Step 3
(stochastic rollout), with `H = 7` and `xxz_hx` fixed by the Step 2 result.

---

## 10. Steps 1–2 results (2026-07-21)

Deps installed into the repo-root `.venv` (`scikit-learn` 1.9.0, `stumpy` 1.14.1).
Note §5's install path points at a different machine (`/Users/popsuksumetchoengprachya/…`);
the working venv here is `<repo>/.venv`.

Code: `forecast.py` (Step 1), `run_step2.py` (Step 2). Artefacts in `results/`.

### Step 1 — gates PASSED

- stepwise driving ≡ `run()` to 1e-12, for all three kinds
- `rollout(H=1)` ≡ `forecast_1step`, same value and index, at 3 origins
- checkpointed rollout ≡ replayed-history rollout
- 1-step test NMSE **0.0582 ising / 0.0535 xxz_hx** vs the notebook's reported
  **0.0577 / 0.0496** — independent cross-check, same ballpark

### Step 2 — kill-test: PASS on the product premise, PARITY vs classical

`nino34`, H=24, 223 held-out origins, seed 7:

| | `nino34` (target) | `tao` (long-N companion) |
|---|---|---|
| skill (NMSE<1) | h=1–7 | all 24 |
| materially beats persistence (>5%) | all 24 | h=6–8 |
| **useful lead** | **h=1–7 (0.58 yr)** | h=6–8 (8 d) |
| vs size-matched ESN | QRC better h=1 only; parity h=2–11; **ESN better h=12–24** | QRC better h=3–24 |
| λ₁ / H_max | 0.173 /yr → 69.4 steps | → 10.2 steps |
| empirical / physical | **0.10 — discrepant** | 0.79 — consistent |

**Set `H = 7` downstream.**

Three findings that change the plan:

1. **The §2 λ₁ warning was correct.** `nino34` empirical horizon is ~10× shorter than
   `H_max`; on `tao` the same code gives ratio 0.79. So the discrepancy is a property of
   the seasonal series, not of the method. `H_max` is a scale on `nino34`, not a
   measurement — as §2 anticipated.
2. **§2's "honest horizon = crossing with persistence" does not work here.** Persistence
   degrades so fast (NMSE 2.69 at h=24) that a reservoir at NMSE 1.72 still "beats" it.
   The binding constraint is the no-skill line NMSE = 1. Horizon is set from that.
3. **No quantum advantage on the target — and it is worse than one seed suggested.**
   See the seed sweep below, which overturned the single-seed reading.

### Seed sweep — this changes the Step 2 conclusion

5 seeds (7, 11, 23, 42, 101), H=12, `xxz_hx` vs size-matched ESN, 5% margin.

**`xxz_hx` is deterministic** (clean NN chain, no disorder — `rng` is documented as
ignored), so its curve is identical at every seed. Only the ESN varies. A single-seed
comparison is therefore a *fixed point against one draw of a random variable*, which is
not a fair test and was silently the test being run.

| | h=1 | h=6 | horizons won (of 12) |
|---|---|---|---|
| `xxz_hx` | 0.0525 ± 0.000 | 0.875 ± 0.000 | — |
| ESN on `nino34` | 0.0586 ± 0.0028 | **0.827 ± 0.064** | ESN wins 10–11/12 at seeds 23, 42, 101 |
| ESN on `tao` | 0.1105 ± 0.0215 | **0.885 ± 0.502** | QRC wins 10–11/12 at **every** seed |

1. **On `nino34` the ESN is generally the better model.** At h=6 the mean ESN (0.827)
   beats the QRC (0.875). The seed-7 result that produced "parity h=2–11" was a weak ESN
   draw; 3 of 5 seeds have the ESN winning almost every horizon. Against plan.md §7.1
   this is closer to a **FAIL** on the target dataset than to the qualified pass first
   reported.
2. **On `tao` the QRC advantage is robust** — 10–11 of 12 horizons at all five seeds,
   and the ESN is wildly unstable there (σ = 0.50 at h=6; one seed hits NMSE 1.83,
   i.e. worse than the mean predictor). The QRC's determinism is a genuine reliability
   argument on this series.
3. **Any future ESN comparison must be over several seeds**, quoting the ESN
   mean/spread, not one draw.

### Carried into Step 3

- **Clipping is live.** Fed-back predictions hit the [0,1] encoding bound on 1.6% of
  `ising` steps (0.0% `xxz_hx`). `rollout()` reports unclipped values and counts clips;
  the count matters for §3's under-alarm argument, since clipping truncates exactly the
  excursions the detector needs.
- Step 3's calibration gate is now the next kill-test.
