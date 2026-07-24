# QRC proof of concept — *forecast-then-detect* for compound climate events

A small, self-contained quantum-reservoir-computing (QRC) pipeline that forecasts real
climate drivers and detects **compound extreme events** downstream — and that tests its
own central claim to destruction rather than advertising it.

```
   drivers  ─►  QUANTUM RESERVOIR  ─►  features  ─┬─►  ridge     ─►  point forecast   (NMSE)
  (SST,SOI,…)   (frozen dynamics)                 └─►  logistic  ─►  P(exceedance)     (detection)
```

The reservoir does the one thing it is genuinely good at — turning a time series into a
rich bank of temporal features — and the rare-event / compound-event logic sits
**downstream** on those features. The value is read at the **detection** metric, never at
forecast NMSE, because a forecaster that is better in the bulk can be worthless in the tail.

> **All quantum results are simulation. The climate data is real.**

---

## Two engines, validated against each other

| Engine | File | What it is |
|--------|------|------------|
| Theory (NumPy) | `QRC_theoretical.py` | Exact statevector evolution of the gate-model reservoir |
| Qiskit twin | `QRC_QPU_implementation.py` | The *same* circuit family on a Qiskit statevector / shot simulator (dormant IBM-hardware hook) |

They are the same physics at the same seed, so the Qiskit port must reproduce the NumPy
features to machine precision. It does:

```
validation gate   max|qiskit_statevector - numpy_exact|  = 1.8e-15
SST 3-month forecast:  NumPy NMSE = 1.287856  ==  Qiskit NMSE = 1.287856   (predictions agree to 8e-15)
```

The reservoir is the windowed-restart, frozen-entangler variant of the Fujii–Nakajima
protocol (Fujii & Nakajima, *Phys. Rev. Applied* **8**, 024030, 2017). Per window step,
oldest input first: encode each qubit with `RY(γ·gᵢ·uₖ)`, then apply a frozen entangler
`W = CP(s·π) ring · RY(s·aᵢ) · RZ(s·bᵢ)`; read out `⟨Zᵢ⟩, ⟨Xᵢ⟩, ⟨ZᵢZⱼ⟩` (20 features at
n = 5). `MultiChannelReservoir` extends this to several drivers by encoding each on a
different rotation axis (`RY, RX, RZ`) of every qubit, so the shared entangler builds
genuine cross-channel features; with one channel it reduces **exactly** to the univariate
reservoir.

---

## The data — three real, physically-coupled ENSO drivers

The El Niño–Southern Oscillation is a coupled ocean–atmosphere system, which is precisely
what makes it a fair testbed for a *compound* event. Three real monthly series are pinned
as reproducible CSV caches in `data/`:

| key | driver | source | span (pinned) |
|-----|--------|--------|---------------|
| `sst` | Niño-region sea-surface temperature | NOAA via statsmodels `elnino` | 1950-01 … 2010-12 |
| `soi` | Southern Oscillation Index (Tahiti−Darwin pressure anomaly) | NOAA CPC | 1951-01 … |
| `pdo` | Pacific Decadal Oscillation index | NOAA PSL | 1948-01 … |

`data/enso.csv` is the file from `QRC_main_stack/.../stage1_numpy_core/data`; `soi.csv`
and `pdo.csv` were fetched and pinned by `data_sources.py`. A **compound El Niño** month is
*warm ocean AND collapsed Southern Oscillation together*: `SST anomaly > +thr` **and**
`SOI anomaly < −thr` (base rate ≈ 10 %, 72 of 720 months).

**Leak rule (non-negotiable).** Every fitted statistic — the monthly climatology used to
deseasonalise, the `[0,1]` encoder scaler, the event thresholds, and every detector
operating point — is computed on the **train span only**. The train boundary is derived
from the `(L, H, train_frac)` harness so it cannot drift.

---

## Training the read-out weights `W_out` — step by step

In reservoir computing **only the linear read-out is trained**; the reservoir itself (the random
gains and the frozen entangler / Hamiltonian couplings) is drawn once per seed and never touched
again — exactly the Fujii–Nakajima *Training readout weights* prescription (§II.E). So "training
the QRC" is a single linear least-squares solve for one weight vector `W_out`. Here is precisely
how it is done, on our data.

### What is being solved

At each month `k` the reservoir emits a feature (signal) vector `x_k ∈ ℝ^F` — `F = 20` for the
windowed 5-qubit reservoir: the five `⟨Z_i⟩`, five `⟨X_i⟩`, and ten `⟨Z_iZ_j⟩`. We append a
constant **bias node** `1`, giving `x'_k = [x_k, 1] ∈ ℝ^{F+1}`. The forecast is a **linear** map
of those signals,

    ŷ_{k+H} = x'_k · W_out ,

and `W_out` minimises the regularised squared error over the training months:

    W_out = argmin_w  Σ_{k∈train} (x'_k·w − y_{k+H})²  +  λ‖w‖²
          = (X'ᵀ X' + λ I)^{-1} X'ᵀ y_target          (closed form — ridge regression).

As `λ → 0` this is exactly the paper's **Moore–Penrose pseudoinverse** `W_out = X'⁺ ȳ` (Eq. 13);
`λ > 0` is the numerically-stable generalisation we use, with `λ` chosen by generalised
cross-validation (step 6). Equivalently, `W_out` defines the paper's *optimal trained observable*
`O_trained = Σ_i w_i·(signal_i) + bias` (Eq. 15) — the reservoir supplies a fixed bank of
observables and training just picks the best linear combination of them.

### Step-by-step, on our data (SST shown; SOI/PDO identical)

The flagship 3-month-ahead SST forecast, as implemented by `forecast_univariate` + `RidgeModel`
in `pipeline.py` over features from `WindowedReservoir.feature_matrix` (`QRC_theoretical.py`):

1. **Anomaly + encode (train-only transforms).** Take monthly Niño SST (732 months); subtract the
   per-calendar-month **climatology fit on the train span only** → anomaly `y_k`; scale to `[0,1]`
   with the train-span min/max → `u_k`, the reservoir input. (`data_sources.prepare_univariate`.)
2. **Drive the reservoir → features.** For every month `k ≥ L−1` (`L = 24`) restart from `|0…0⟩`,
   feed the window `u_{k−23 .. k}`, and read the 20 signals → `x_k`. Stacking gives the feature
   matrix `X`, one row per month. (`feature_matrix(u, L)`.)
3. **Form (input, target) pairs.** Sample `k` pairs feature row `x_k` with target `y_{k+H}`,
   `H = 3` months ahead. (`make_samples` / `align`.)
4. **Chronological split.** First 80 % of samples = train, the rest = test (or the expanding
   folds of `forecast_cv_strict`). **Every statistic below uses train rows only** — no leakage.
5. **Standardise, then add bias.** Compute each feature column's mean `μ_j` and std `σ_j` on the
   train rows; replace `x_k → (x_k − μ)/σ` (so one scalar `λ` penalises all signals evenly), then
   append the constant-`1` bias column → design matrix `X'`.
6. **Pick `λ` by GCV (leak-free).** Over `λ ∈ logspace(−8, 4, 25)`, minimise the generalised-CV
   score `‖y − ŷ_λ‖²/T ÷ (1 − df(λ)/T)²`, where the effective degrees of freedom
   `df(λ) = Σ_i s_i²/(s_i²+λ)` and `s_i` are the singular values of `X'`. One SVD of the **train**
   design matrix serves every `λ`. (`ridge_gcv`.)
7. **Solve for `W_out`.** At the chosen `λ`,
   `W_out = V · diag(s/(s²+λ)) · Uᵀ y_target` — the SVD form of `(X'ᵀX'+λI)^{-1}X'ᵀ y_target`.
8. **Predict & score.** For each test month, `ŷ_{k+H} = x'_k · W_out`; report
   `NMSE = Σ(y−ŷ)² / Σ(y−ȳ_train)²` (1.0 = the train-mean predictor).

### Notes

- **Strict cross-validation** (`forecast_cv_strict`) repeats steps 1–8 inside each expanding-window
  fold, re-fitting the climatology, scaler, reservoir features **and** `W_out` on that fold's train
  months only — this was the audit fix (see the audit section).
- **Two regularisation flavours, one read-out.** The ENSO forecasts use standardised features +
  GCV-selected `λ` (above). The paper-faithful NARMA benchmark in `Hamiltonian_QRC` uses raw
  features with a tiny fixed `λ = 10⁻⁶…10⁻¹²` (`ridge_fit`), i.e. the plain pseudoinverse. Both
  are the *same* linear least-squares read-out; they differ only in regularisation/standardisation.
- **Detection uses a separate head, not `W_out`.** The compound-event stage does **not** reuse the
  forecasting weights; it trains a downstream logistic-regression classifier `P(event)` on the same
  reservoir features (`exceedance_head`). `W_out` is strictly the forecasting read-out.

---

## Files

| File | Role |
|------|------|
| `QRC_theoretical.py` | NumPy reservoirs (`WindowedReservoir`, `MultiChannelReservoir`) + ridge read-out + **matrix-level theoretical diagnostics** (`linear_memory_capacity`, `entangler_scrambling`) + validation suite. Run it: `python QRC_theoretical.py`. |
| `QRC_QPU_implementation.py` | Qiskit statevector/shot twin of the univariate reservoir. Run it to see the validation gate. |
| `data_sources.py` | Loads/pins the three drivers; leak-free anomaly + scaler transforms; univariate and aligned-multivariate preparation. |
| `pipeline.py` | Forecast-then-detect harness: ridge point forecast, classical battery (audit-corrected baselines), logistic exceedance head, the joint-vs-bank-vs-ESN compound experiment, **strict per-fold rolling-origin CV** (`forecast_cv_strict`), skill-vs-horizon, the `(gamma, ent_scale)` **operating-point sweep** + `memory_capacity_grid`, **finite-shot sampling** (`probs_cache`, `sampled_feature_matrix`, `forecast_nmse_vs_shots`), and the **horizon study** (`point_forecast_vs_horizon`). |
| `tools/build_notebook.py` | Regenerates and re-executes `QRC_run.ipynb` from source (the notebook is fully reproducible: `$PY tools/build_notebook.py`). |
| `QRC_run.ipynb` | The demonstration notebook (pre-executed, figures embedded). |
| `data/` | `enso.csv`, `soi.csv`, `pdo.csv`. |
| `figures/` | PNGs written by the notebook. |

Each module runs standalone (`python <file>.py`) as its own smoke test.

---

## How to run

The stack (`numpy, scipy, qiskit, qiskit-aer, statsmodels, pandas, matplotlib,
scikit-learn`, plus `ipykernel/nbclient` for the notebook) is already installed in
`QRC_main_stack/QRC_code_stack/.venv`, and a Jupyter kernel **“Python (Quantathon QRC)”**
is registered for it.

```bash
PY=../../QRC_main_stack/QRC_code_stack/.venv/bin/python   # from proof_of_concept/

$PY QRC_theoretical.py          # NumPy validation anchors
$PY QRC_QPU_implementation.py   # numpy↔qiskit validation gate
$PY data_sources.py             # data summary  (add --refresh to re-pin from NOAA)
$PY pipeline.py                 # end-to-end smoke test
```

Open `QRC_run.ipynb` and select the **Python (Quantathon QRC)** kernel to re-run it. It is
already executed, so the results and figures are visible without running anything.

---

## Headline results (honest, including the null)

1. **The engines agree to ~1e-15.** Every quantum number is reproducible on either path.
2. **Beats the mean (robustly), but no edge over the classical baselines.** A single 80/20
   split makes SST look *worse* than the train-mean (NMSE 1.29) — but that is a **split
   artifact**: the 80/20 tail (1999–2010) is a hard decade, and a 70/30 cut gives ~0.67. Scored
   the robust way, by **strict rolling-origin CV** (notebook §3; every fitted statistic —
   climatology, scaler, features, read-out — re-fit per fold, see the audit section below), the
   QRC beats the mean on all three drivers (**0.70 / 0.82 / 0.62 < 1**) — a real signal. What it
   does **not** do at `H=3` is beat trivial **persistence** (0.58 / 0.92 / 0.52; QRC wins only on
   SOI), the **corrected ESN** (which wins on SOI 0.75 and PDO 0.47), or a plain linear
   autoregression. The nonlinearity earns nothing over properly-run classical baselines at
   forecasting, so the value question genuinely lives downstream.
   - **Why 3-month lead, not step-by-step (notebook §3d).** The headline uses `H=3` mainly for
     *decision relevance* (early warning needs actionable lead time). `H=1` is a persistence game
     — copying `y_k` *beats* the QRC under strict CV on SST (0.15 vs 0.49) and PDO (0.19 vs 0.46).
     Persistence decays with lead while the QRC decays gently, so the gap narrows; on **SOI** the
     QRC overtakes persistence by `H=3`, but on the smoother SST/PDO persistence stays ahead even
     at `H=3`. So `H=3` is where the QRC is at least competitive, not where it dominates.
3. **Failure mode 1 is real and large.** Thresholding a *point* forecast misses almost all
   compound events (recall ≈ 0); a probabilistic exceedance head recovers them
   (recall ≈ 0.4–0.7). The distributional second stage is the correct design, not cosmetic.
   - **Theoretical vs sampling (notebook §3c).** The finite-shot forecast converges to the
     exact ceiling as `S → ∞` (the numpy emulator matches the Qiskit AerSimulator path to
     shot-noise level). PDO/SOI close the gap from above at `S^{-1/2}`; SST is non-monotonic
     because its exact read-out is over-fit (NMSE>1) so shot noise mildly *regularises* it.
4. **Failure mode 2 is real and quantified.** Compound-detection skill is meaningful at
   1–3 months (AUC ≈ 0.69) and **collapses toward chance by ~6 months** (AUC ≈ 0.49).
5. **On the quantum-advantage claim, a null that survives a fair sweep.** At the reference
   operating point the **joint** quantum reservoir does **not** beat a **bank** of
   one-channel reservoirs, and the **classical multivariate ESN is the strongest** detector:

   | model (SST+SOI, rolling CV, 52 events) | avg precision | ROC-AUC | mean F1 |
   |---|---|---|---|
   | joint (QRC) | 0.324 | 0.694 | 0.204 |
   | bank (QRC) | 0.357 | 0.731 | 0.258 |
   | ESN (classical) | **0.418** | **0.826** | **0.312** |

   The reference point *was* suboptimal — sweeping `gamma`×`ent_scale` lifts the joint
   reservoir to AP 0.406 — **but the null survives the sweep**: best-joint 0.406 still loses
   to best-bank 0.442, and the ESN dominates on AUC (0.826). A **theoretical** panel computed
   directly from the matrices (memory capacity + entangler scrambling) explains the landscape
   and shows only a moderate link (r ≈ +0.5) to detection skill. Reported plainly, not tuned away.

---

## Implementation audit vs the founding paper — errors found & fixed (2026-07)

Every QRC-relevant file (`QRC_theoretical.py`, `QRC_QPU_implementation.py`, `pipeline.py`,
`data_sources.py`, `QRC_run.ipynb`, plus `Quantathon_stack/Hamiltonian_QRC/{qrc_core, experiments,
sampling_forecast, data_loader}.py` and its notebooks) was traced against **Fujii & Nakajima,
"Harnessing disordered-ensemble quantum dynamics for machine learning"** (arXiv:1602.08159,
PR Applied **8**, 024030, 2017), with special attention to the *Training readout weights* section.
Notebook **§8** re-runs the verification live. What the audit established:

**The core QRC logic is correct.** Injection `ρ→ρ_s⊗Tr₁ρ` with `√(1−s)|0⟩+√s|1⟩`, evolution
`e^{−iHτ}`, V-node temporal multiplexing, frozen random couplings (randomised **once** per seed —
never trained, never re-drawn), washout → chronological train → later-evaluation, and a linear
read-out + constant bias fit by (ridge-regularised) least squares on **train rows only** — all as
the paper prescribes. Verified dynamically: the echo-state property holds (initial-state signal
difference decays 0.56 → ~10⁻⁸ by 200 steps), and the paper's **own NARMA10 benchmark reproduces**
(QR V=5: NMSE 1.7×10⁻³ vs the paper's LR control 6.3×10⁻³, in the paper's ~10⁻³ band, with V↑ ⇒
error↓ exactly as in its Fig. 12).

**Errors found (all fixed; none changed the direction of the honest conclusions — both genuine
bugs had actually *flattered the quantum side*):**

| # | error | impact | fix |
|---|---|---|---|
| 1 | `classical_battery` fed the ESN through `align()`, giving it a state **23 months stale**, and gave ESN/linear-lags the clipped `[0,1]` encoder series instead of raw anomalies | crippled the classical baseline: ESN read 1.29/1.01/1.07 instead of the true 1.35/**0.75**/**0.47** — the honest ESN **beats the QRC on SOI and PDO** | `pipeline.classical_battery` re-indexed + raw-anomaly inputs (reproduces the validated reference battery exactly) |
| 2 | rolling-CV reused the 80%-span climatology/`[0,1]`-scaler for every fold, so folds testing months before the 80% boundary had those months **inside their own deseasonalisation statistics** | leak worth ~+0.04 NMSE in the QRC's favour on SST | replaced by `pipeline.forecast_cv_strict`: climatology, scaler, reservoir features and read-out all re-fit per fold on that fold's train months only; all CV numbers re-derived |
| 3 | compound-event **labels** derive from the 80%-span climatology/thresholds (fixed reference-period definition; shared identically by joint/bank/ESN) | absolute AP slightly optimistic; *rankings* unaffected | robustness check in §8: with label statistics fit strictly **before every CV fold** (35% span), the classical ESN **still wins** (AP 0.51/AUC 0.77) — the null is convention-robust |

**Documented conventions (deviations from the paper that are choices, not bugs):**
- `WindowedReservoir` is **not** the paper's persistent-state protocol — it is the **rewinding
  protocol** of [Mujal et al., npj Quantum Information 9, 16 (2023)](https://www.nature.com/articles/s41534-023-00682-z)
  (restart + fixed input window per output step), the standard way to run QRC on gate hardware
  where measurement destroys the state. The paper-faithful persistent-state reservoir is
  `Hamiltonian_QRC/qrc_core.py::QuantumReservoir`, and the memory cost of rewinding is quantified
  in the memory-capacity plot.
- Our Ising field term is `(h/2)ΣZᵢ` vs the paper's Eq. (16) `hΣZᵢ` (our `h=1` ≡ paper's `h=0.5`),
  and couplings sum over `i<j` vs the paper's `Σᵢⱼ` — hyperparameter conventions of an untrained
  disordered reservoir, now stated in the docstrings.
- Our NMSE is variance-about-train-mean (mean predictor = 1, the stricter RC convention); the
  paper's Eq. (A1) normalises by `Σȳ²`, which reads lower for non-zero-mean targets. The two
  scales must not be compared directly.
- Ridge (λ by GCV on train rows) generalises the paper's Moore–Penrose pseudoinverse; the
  Hamiltonian pipeline's fixed `λ=1e−6…1e−12` is pseudoinverse-equivalent.

## Careful-implementation notes

- **Endianness** (the classic Qiskit trap) is reversed exactly once per pathway in the
  Qiskit twin; the validation gate is what proves nothing else reorders qubits.
- **Multichannel encoding** uses distinct rotation axes because `RY(a)·RY(b) = RY(a+b)` —
  same-axis layers would silently collapse two drivers into one linear combination.
- **Strict rolling-origin CV** is used for *both* the compound detection AND the univariate
  forecast NMSE (`pipeline.forecast_cv_strict`), because a single 80/20 split is fragile here:
  it leaves only ~6 compound months in the test tail, and for the forecast it swings SST H=3
  NMSE from ~0.67 (70/30) to 1.29 (80/20) purely by where the split lands. *Strict* means every
  fitted statistic — the monthly climatology, the `[0,1]` encoder scaler, the reservoir feature
  matrix and the ridge read-out — is re-fit inside each expanding-window fold on that fold's
  train months only (see the audit section: the earlier fixed-climatology CV leaked ~+0.04 NMSE
  in the QRC's favour). Pooling out-of-sample predictions across folds uses every month while
  never letting a fold see its own future — so the reported skill does not hinge on one
  arbitrary cut. (This corrected an earlier single-split reading that had SST "worse than the
  mean".)
- **Why the Hamiltonian-QRC `predict_timeseries` notebook shows much lower NMSE (~0.02–0.09).**
  Verified factor-by-factor in **notebook §7** (both reservoirs on the *same* data, one knob at a
  time, with the full exact-setup table, a compounding "staircase" 1.29 → 0.06, and a
  **per-case prediction-trajectory figure** — `figures/attribution_cases.png` — where the SST
  forecast visibly sharpens from loose to near-perfect as each factor is flipped). The gap is
  **not** one thing: (1) it forecasts **1-step** (`H=1`), near-persistence and far easier than the
  `H=3` headline; (2) it uses `train_frac=0.7` — the single biggest lever, but really single-split
  noise (SST swings 0.67↔1.29), which is why §3 now reports rolling-origin CV; (3) intrinsically
  easier series (30-min solar, daily TAO); (4) architecturally its continuous + 4-virtual-node
  reservoir is genuinely richer, but its edge is **operating-point-dependent** — only ~0.15 at the
  `H=3` headline (1.29→1.14) yet ~0.3 at the easy `H=1` point (0.46→0.15), so "minor" held only for
  the headline. The windowed gate reservoir stays regardless because it is the one that validates
  exactly against the Qiskit twin. (The Hamiltonian's simpler `λ=1e-6` ridge is *not* an advantage —
  it slightly worsens SST in the staircase.)
- **One operating point** (`γ=π/4, s=0.5`) is used everywhere, inherited from the
  univariate study and deliberately **not** re-tuned on any test span or on the compound
  task.
- **Forecast horizon is `H=3` months, not step-by-step — on purpose (notebook §3d).** Two
  reasons. *(1) Product:* the deliverable is compound-event **early warning**, which is
  worthless without actionable lead time — a 1-step (1-month) forecast gives essentially none;
  3 months is the standard operational ENSO lead. *(2) Method:* at `H=1` these slowly-varying
  anomalies are dominated by **persistence** (next month ≈ this month), which under CV *beats*
  the QRC on SST (0.16 vs 0.55) and PDO (0.21 vs 0.61) — a short horizon flatters persistence,
  not the reservoir. Persistence error climbs with lead while the QRC decays gently, so the gap
  narrows; the QRC overtakes persistence by `H=3` on **SOI**, though on the smoother SST/PDO
  persistence stays ahead even there. So `H=3` is chosen mainly for lead-time relevance (and is
  where the QRC is at least competitive), not because it dominates the baseline. `H` is a free
  knob (`baselines`/`data_sources.HORIZON`, or `pipeline.make_samples(H=...)`); §3d shows the
  step-by-step (`H=1`) forecast and the full rolling-CV skill-vs-lead curve.
- **Theoretical vs sampling (§3c).** "Theoretical" means the exact statevector features
  (`S→∞`); "sampling" estimates each feature from `S` shots via `add_shot_noise` /
  `sampled_feature_matrix` — the numpy emulator that matches the Qiskit AerSimulator path to
  shot-noise level. The read-out is fit on the sampled features it predicts from (honest
  end-to-end device numbers), and the sampled forecast converges to the exact ceiling as `S`
  grows.

## Windowed restart vs continuous drive — the reservoir difference explained

§7's Factor 4 shows the reservoir is one lever in the accuracy gap. Here is *what* differs and
*why* it matters, grounded in two measurements (reproducible in §7).

**What differs.**
- **`WindowedReservoir` (this notebook — "windowed").** For every forecast step it **restarts from
  |0…0⟩** and pushes a fixed window of `L=24` inputs through a **frozen scrambling entangler**
  (`RY` encode → `CP`-ring → frozen `RY/RZ`), then reads its 20 features **once, at the end of the
  window**. A shallow gate circuit; no state is carried between windows. This is the established
  **rewinding protocol** for gate-based QRC ([Mujal et al., npj QI 9, 16 (2023)](https://www.nature.com/articles/s41534-023-00682-z)),
  not an ad-hoc shortcut.
- **`QuantumReservoir` (Hamiltonian — "continuous", without restart).** It **never restarts**: it
  keeps one density matrix `ρ` across the whole series, injects each input by **partial trace**
  (the input qubit is traced out and re-prepared with the new value, `ρ → ρ_in(u) ⊗ Tr₀ρ`), evolves
  under a fixed Hamiltonian, and reads `⟨Zᵢ⟩` **immediately**, at `V=4` sub-steps (30 features).

**Why it makes a difference** — it comes down to the read-out's *access to recent inputs*:
1. **Linear memory capacity (Jaeger MC on i.i.d. input):** windowed **MC ≈ 1.25** vs continuous
   **≈ 5.6 (V=1) / 7.4 (V=4)** — see the memory-function plot `figures/memory_capacity.png`, where
   the windowed curve `MF(d)` is flat and near-zero at *every* delay (even `d=1`) while the
   continuous curves start near 1.0 and decay over ~6–8 steps (the area under each curve is its MC).
   The continuous reservoir linearly retains ~5× more of the recent input history. For short-lead
   forecasting of autocorrelated series — where the best answer is close to the recent value — that
   is decisive, and it is exactly why the windowed reservoir cannot even reproduce persistence at
   `H=1` while the continuous one can.
2. **A longer window makes the windowed reservoir *worse*, not better:** CV NMSE (SST, `H=1`) runs
   `L=12 → 0.31`, `L=24 → 0.55`, `L=48 → 0.78`, `L=96 → 0.96` (≈ the mean predictor). So the problem
   is **not** too little history — it is that pushing the most-predictive recent input through more
   frozen-entangler scrambling before the single end-of-window read-out **buries its linear
   signature**. The continuous reservoir reads each input *right after* injecting it, before it
   scrambles.

In one line: **the windowed reservoir scrambles the recent input away before it reads out; the
continuous reservoir reads it while it is still fresh** — hence far more usable short-term memory.
(Virtual nodes are secondary: continuous `V=1`, with *fewer* features, already beats windowed by a
wide margin.)

**Why keep the windowed reservoir anyway.** It is the **exact gate-circuit twin of the Qiskit
implementation** (§1) — shallow restart-per-window circuits a real QPU can run. The continuous
reservoir needs a persistent quantum state across the whole series (mid-circuit reset + long
coherence), far harder on hardware. The windowed design **trades forecast accuracy for
hardware/circuit faithfulness**, deliberately — and the compound-detection deliverable does not hinge
on univariate forecast NMSE. (It *can* be pushed toward higher memory — shorter `L`, lower `γ` per
§5 — but that is a separate tuning question.)

## Limitations & next steps

n = 5 qubits, statevector only (the IBM hardware path is implemented but dormant; no
shot/device-noise results are claimed). The `(gamma, ent_scale)` sweep is a coarse 4×4 grid
scored by CV — reading its max is optimistic, and the encoding-axis choice was not swept —
so while the null is robust to what was tested, the grid is not exhaustive. Event counts are
modest (52 / 27 compound months). The SST cache ends 2010, so the 2015–16 and 2023–24
super-El-Niños are outside the record.

Next: *nested* operating-point selection and an encoding-axis sweep; an explicit
tail-dependence diagnostic (does the joint model reproduce the observed SST–SOI co-exceedance
rate the bank cannot?); other compound pairs (heat + drought, rain + antecedent soil
moisture); and — only if a simulated edge survives — finite-shot and real-hardware runs. The
scaffolding for all of it already lives in `pipeline.py`, `QRC_theoretical.py` and
`data_sources.py`.
