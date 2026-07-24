# HANDOFF_P7 — Baseline battery + structure controls

Written per G10. In-repo (declared deviation, as P0–P6).

## 1. Objective & status vs exit criteria

| Exit criterion | Status | Evidence |
|---|---|---|
| Simple baselines (mean/persistence/seasonal) reused from P2 | ✅ | `models/trivial_baselines.py` |
| Statistical: ridge-AR + ARIMA/SARIMA (AIC + dev) | ✅ | `models/autoregression.py`, `models/statistical.py` |
| Reservoir: ESN size-matched (effective rank) + best-of-class; NVAR | ✅ | `models/esn.py`, `models/nvar.py` |
| Neural: small LSTM/GRU (torch optional, early stop on dev) | ✅ | `models/{lstm,gru}.py` (`AVAILABLE` guard) |
| Variational quantum: windowed VQC (+ QLSTM behind torch) | ✅ | `models/qlstm.py` |
| Haar-random-reservoir control (G7) | ✅ | `quantum/hamiltonians.py::haar_reservoir` |
| One command → teacher-forced table on MG + one climate dev split | ✅ | `scripts/run_baselines.py`, `make baselines` |
| Every closed-loop baseline passes the G5 no-future rule | ✅ | `tests/test_baselines.py` (parameterised) |
| Suite passes | ✅ | **212 passed, 1 skipped** (torch) |

## 2. Artifacts & keys

- `models/autoregression.py::RidgeAR` — ridge-AR on `p` lags via the shared SVD
  readout; `warm`/`step` adapter (most-recent lag prepended). Recovers a known
  AR(2)'s coefficients to ±0.05.
- `models/nvar.py::NVAR` — Gauthier-style delay + unique degree-2 polynomial
  features + ridge; `warm`/`step`.
- `models/esn.py` — leaky ESN; `best_of_class` (spectral radius/leak/input-scale
  grid, selected on **dev NMSE only**) and `size_matched` (grows `N_r` until the
  participation-ratio effective rank of its states reaches the QRC's — **G7**, not
  qubit-count matching). Under a scalar drive the ESN state effective rank
  saturates ≈2, so the size-matched targets are small (measured: MG QRC rank 1.76
  → Nr 20; ENSO QRC rank 1.49 → Nr 2).
- `models/statistical.py::ARIMAModel` — statsmodels ARIMA/SARIMA, order by AIC;
  `warm`/`step` append-and-forecast adapter (recursion flows through one engine).
- `models/{lstm,gru}.py::RNNForecaster` — shared sliding-window RNN, early stopping
  on a dev tail; `AVAILABLE=False` when the optional `torch` extra is absent.
- `models/qlstm.py::WindowedVQC` — a genuine trainable data-re-uploading VQC (exact
  statevector, SciPy L-BFGS over rotation angles **and** the linear read-out;
  train rows sub-sampled to ≤200 for tractability). `QLSTM` (Chen–Yoo, torch)
  behind `TORCH_AVAILABLE`.
- `quantum/hamiltonians.py::haar_reservoir` — Haar eigenvectors, structureless;
  drop-in `Reservoir` for `ExactQRC` with identical injection/readout (G7 control).
- `scripts/run_baselines.py` → `results/metrics/run_baselines.json`; `make baselines`.
- `tests/test_baselines.py` (11 tests): AR-coefficient recovery, NVAR/VQC skill,
  ESN effective-rank matching, ARIMA fit+forecast, Haar-vs-structured distinctness,
  and the **G5 no-future rule parameterised over {ridge_ar, nvar, esn, windowed_vqc}**.

## 3. Key results (teacher-forced one-step NMSE)

- **MG τ=17** (n=1600, n_train=1120): ridge_ar/nvar/esn ≈ 0.000, QRC(FN exact)
  0.0001, windowed_vqc 0.0006, persistence 0.0218, climatology 1.000. As P6 found,
  **one-step is too easy to separate the strong models** — separation lives in the
  autonomous/dynamical regime (P8).
- **ENSO dev** (n=586, n_train=410): everything clusters at CNRMSE ≈ 0.099–0.112
  (nvar 0.0992 best, persistence 0.1124). Honest finding: **univariate one-step
  ENSO is near-persistence for every model class**, matching the predictability
  priors (P9 sets expectations properly).

## 4. Decisions & deviations

- **VQC is exact-statevector + SciPy, not PennyLane** — deterministic, no extra
  framework, and testable; train rows sub-sampled (≤200) to keep the L-BFGS loop
  tractable (~18 s per fit). Faithful to "one fully specified windowed VQC".
- **Haar control implemented as a `Reservoir` with Haar eigenvectors** (H = Q Λ Q†)
  rather than a bespoke class, so it flows through *all* existing FN machinery
  (`ExactQRC`, `StatefulFN`, feature cache) unchanged.
- **LSTM/GRU/QLSTM require the optional `torch` extra** (not installed here);
  their rows are reported as `None`/skipped, never silently dropped (spec §24).
- Climate baselines run on the train-only-anomaly, [0,1]-scaled ENSO series (same
  transform the QRC track uses), for a like-for-like table.

## 5. Open issues / known gaps

- torch not installed in this env ⇒ LSTM/GRU/QLSTM paths are code-complete but
  unexercised; install `.[neural]` to populate those rows.
- Best-of-class ESN grid is intentionally small (3×3×3) for runtime; widen in the
  P8/P9 campaigns where the full multi-origin budget applies.
- ARIMA emits a NumPy-2.5 shape-deprecation warning from statsmodels (upstream,
  benign).

## 6. Entry instructions for Phase 8 (controlled-system campaign + dynamical audits)

1. `make test` green first (212 passed, 1 skipped).
2. Audits FIRST (G8-1..3): `dynamical_systems/lorenz63.py` — integrate β=8/3 AND
   the printed 3/8 at (σ,ρ)=(10,28); Benettin λ_max each; record which gives ≈0.9.
   `dynamical_systems/enso_ode.py` — Vallis (1986) box model; audit Hamhoum Eq. 13
   signs/params; compute λ ourselves; pin the time unit (never reuse 0.05–0.1).
   `dynamical_systems/integration.py` is still a stub (RK4 + Benettin-for-ODE).
3. Campaign: 20–50 origins × {FN, rewind} × baseline battery; horizons in steps
   AND Lyapunov times; dynamical-fidelity suite; shot-slope H≈a+b·ln S vs
   b≈1/(2λΔt) with the signal-to-noise ceiling check as a precondition.
4. `scripts/run_autonomous_forecasts.py` (`make grid-autonomous`) + `HANDOFF_P8.md`.

## 7. Suggested skills

`quantum-reservoir-computing` + `qrc-project-playbook` (min per G10); add
`citation-audit` when Gauthier 2021 / Chen–Yoo QLSTM enter `references.bib`.
