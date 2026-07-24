# HANDOFF_P2 — Exact FN reservoir, features, readout, metrics, trivial baselines

Written per G10. In-repo (declared deviation, as P0/P1).

## 1. Objective & status vs exit criteria

Phase-2 goal: the exact NumPy FN reference reservoir (the ceiling for everything
else) plus the whole feature→readout→metric chain, all physics-free-testable.

| Exit criterion | Status | Evidence |
|---|---|---|
| Exact reservoir `run`/`features` implemented (G8-4 virtual nodes) | ✅ | `quantum/exact_qrc.py` |
| Feature = (1+⟨Z⟩)/2 + bias; observable set is config | ✅ | `exact_qrc.features`, `quantum/observables.py` |
| Readout: SVD ridge path + λ=0 pinv + GCV, multi-target | ✅ | `models/readout.py` |
| Both NMSE conventions labelled (G8-5), capacity, eff-rank | ✅ | `evaluation/metrics.py` |
| Trivial baselines (mean/persistence/seasonal), train-only | ✅ | `models/trivial_baselines.py` |
| align() is the one G3 pairing point | ✅ | `data/windows.py` |
| G3 STM τ_B=0 exactness → C(0)=1.0 machine precision | ✅ | `test_exact_qrc.py::test_g3_stm_tau0_exactness` (NMSE<1e-12) |
| MV+1 ≤ L/5 assert refuses to run | ✅ | `test_exact_qrc.py::test_budget_assert_refuses_short_series` |
| float32 cache vs float64 within 1% NMSE (G4) | ✅ | `test_exact_qrc.py::test_float32_cache_within_one_percent` |
| FN Fig.5 direction (C_STM rises/saturates with V; C_PC needs V) | ✅ | `scripts/validate_exact_qrc.py` (results below) |
| NARMA2 Table-I ordering (QR beats LR) | ✅ | same script |
| Suite passes | ✅ | **121 passed, 11 skipped** (~2s) |

## 2. Artifacts & keys

Modules (`src/qrc_single_time_series/`):
- `quantum/exact_qrc.py` — `ExactQRC(reservoir, V, tau, observable_kind)`.
  - `.features(inputs, x0, sample, include_bias, check_budget)` — GEMM eigenbasis
    path; `sample='virtual'` → M·V (+bias), `sample='postinjection'` → M (+bias,
    the debug v=−1 sample for G3). Enforces MV+1 ≤ L/5.
  - `.run(inputs, x0, debug_postinjection)` — reference generator yielding
    `(k, v, ρ)` at every virtual time (v=−1 first if debug). Materialises ρ; slow.
  - `.from_config(cfg)` builds from `qrc_exact.yaml`. `.feature_dim = M·V + 1`.
  - Efficiency: σ=W†ρW once/step; `_Phi_sub` (V,d,d) + `_Phi_end`; obs pre-rotated
    to the eigenbasis (`_ObsT`); one einsum for all V readouts; advance = one
    basis round trip `W(σ⊙Φ_end)W†`.
- `quantum/observables.py` — `build_observables(N, kind, strings)` for
  `z_local` / `z_and_zz` / `pauli`; `is_diagonal`. **Fast Z-family FWHT path NOT
  implemented** — the eigenbasis einsum handles any observable correctly; FWHT is a
  micro-opt, deferred (note for later if Z-only large-N speed matters).
- `models/readout.py` — `fit(X, Y, lam, rcond, gcv_grid) → Readout`. Centered
  (no bias column; intercept recovered), one economy SVD, ridge path
  W(λ)=V diag(s/(s²+λ))Uᵀy_c, per-target GCV (`lam='gcv'`), λ=0 pinv mode
  (`rank`, `w_norm`, `dof` logged). 1-D Y supported. `Readout.predict(X)`.
- `evaluation/metrics.py` — `mse/rmse/mae`, `nmse_variance` (mean-predictor=1),
  `nmse_fn` (FN Eq.A1 /Σy²), `r2`, `pearson`, `mase`, `anomaly_correlation`,
  `capacity` (=corr²), `effective_rank` (Roy-Vetterli), `nmse_mean_predictor`
  anchor, `persistence_skill`.
- `models/trivial_baselines.py` — `climatological_mean`, `mean_forecast`,
  `persistence_forecast(horizon)`, `seasonal_persistence_forecast(period=12)`.
  All train-only; each `*_forecast` returns `(idx, y_true, y_pred)`.
- `data/windows.py` — `align(features_by_k, y, horizon)`, the ONLY G3 pairing.
  Row k ↔ y_{k+horizon}; horizon≥0 forecasting, horizon<0 for STM delay τ_B.

Tests (real): `test_exact_qrc.py` (11), `test_readout.py` (7), `test_metrics.py` (8).

Result artifact: `results/metrics/validate_exact_qrc.json` (via `make validate-exact`).
Experiment registered in `configs/experiments.yaml::registry.validate_exact_qrc`.

## 3. FN Fig.5 / Table-I reproduction (N=5, seed=7, τ=2.0, J=1, h=0.5)

| V | STM total MC | C(1) | C(3) |
|---|---|---|---|
| 1 | 2.98 | 0.835 | 0.223 |
| 2 | 4.55 | 0.937 | 0.373 |
| 5 | 6.51 | 0.993 | 0.958 |
| 10 | 7.52 | 0.997 | 0.971 |

- Parity/processing capacity: PC(V=1)=0.39, PC(V=10)=1.73 — needs virtual nodes.
- NARMA2 sine: QR NMSE≈0.000 vs LR NMSE=0.066 — QR beats LR (Table-I ordering).
- Qualitative only (random couplings, unstated pinv cutoff — G8). MC rises and
  saturates with V; direction matches FN Fig. 5. **Figures not rendered**
  (matplotlib not installed); JSON holds the numbers, regenerate plots later (G1).

## 4. Effective rank (G7 — the number the size-matched ESN must match)

Feature matrix on a scalar iid drive, N=5 seed=7:

| V | feature dim | effective rank |
|---|---|---|
| 1 | 6 | 1.50 |
| 5 | 26 | 2.00 |
| 10 | 51 | 2.03 |

**Critical for G7**: under a scalar drive the nominal N·V features expose an
effective rank of only ~2, *far* below the feature count. The honesty-battery
size-matched ESN (Phase 7) must be matched to this **effective rank (~2)**, NOT to
NV=51. Recompute per config; it is logged for every feature matrix from here on.

Feature-count / min-L budget (MV+1 ≤ L/5, M=N=5):

| V | feat = 5V+1 | min L |
|---|---|---|
| 1 | 6 | 30 |
| 5 | 26 | 130 |
| 10 | 51 | 255 |

## 5. Decisions & deviations

- **New module `models/trivial_baselines.py`** — the phase file mandates the
  trivial baselines "now" (P2) but no scaffold module owned them
  (`autoregression.py`/`statistical.py` are P7). Single-responsibility, documented.
- **FWHT Z-family fast path deferred** — eigenbasis einsum is correct for any
  observable; FWHT is an optional speed-up, not a correctness item.
- **matplotlib not installed** → reproduction saves JSON + prints; no figures yet.
  `uv pip install --python .venv/bin/python matplotlib==3.11.0` when plots needed.
- Per-target GCV implemented; joint default falls out for univariate Y.
- `qiskit/aer/statsmodels/pennylane` still not installed (P3/P8 on demand).
- No commit made (user hasn't asked).

## 6. Open issues / known gaps

- Endianness Qiskit-wire half (`quantum/endianness.py`) still a stub — lands P3.
- 11 skipped tests remain = P3/P4 placeholders.
- NARMA2 QR NMSE is ~0 (sine input is highly predictable at V=10); ordering holds
  but the margin is not a stress test — a harder input (chaotic/noisy) would be a
  sharper Table-I check if desired later.

## 7. Entry instructions for Phase 3 (Trotter mirror + Qiskit + validation ladder)

1. `make test` green first (121 passed, 11 skipped).
2. **Install qiskit stack**: `uv pip install --python .venv/bin/python qiskit==2.5.0
   qiskit-aer==0.17.2`.
3. Implement `quantum/endianness.py` (the ONE logical↔wire map + counts-key
   reversal, G2) with its two anchor tests (`test_count_to_expectation.py`,
   `test_qiskit_parity.py`): X on Qiskit qubit 0 → counts key '1' in expected slot
   after the single reversal; counts→⟨Z_i⟩=P(0)−P(1) through that one function.
4. `quantum/exact_trotter_qrc.py` (first-order Suzuki-Trotter matrix mirror of the
   FN evolution) then `quantum/{qiskit_qrc,noisy_qiskit_qrc}.py`.
5. Validation ladder (PLAN §7): L1 exact e^{−iHτ} → L2 exact Trotter → L3 Aer
   density-matrix → L4 ideal shots → L5 noisy. N=2/3 matrix-Trotter ≡ Aer-DM to
   1e−10; exact dense vs Trotter converges as Δt→0.
6. Write `handoffs/HANDOFF_P3.md`.

## 8. Suggested skills for the next session

`quantum-reservoir-computing` + `qrc-project-playbook` (min per G10); add
`citation-audit` only when a document ships.
