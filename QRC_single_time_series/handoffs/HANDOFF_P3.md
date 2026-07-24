# HANDOFF_P3 — Trotterised matrix mirror + Qiskit implementations + validation ladder

Written per G10. In-repo (declared deviation, as P0–P2).

## 1. Objective & status vs exit criteria

Phase-3 goal: an exact-matrix Trotter model and ideal/noisy Qiskit models that
agree with the exact reservoir by construction, every error source isolated.

| Exit criterion | Status | Evidence |
|---|---|---|
| qiskit/aer installed (pinned) | ✅ | qiskit 2.5.0, qiskit-aer 0.17.2 in `.venv` |
| Endianness = ONE G2 translation | ✅ | `quantum/endianness.py`; `test_count_to_expectation.py` (7) |
| Schedule consumed by BOTH matrix + circuit | ✅ | `quantum/schedule.py` |
| Angle convention pinned (RXX(2Jτ)=exp(−iJτXX), RZ(2hτ)) | ✅ | `test_qiskit_parity.py` (1e-12) |
| L1 dense vs L2 Trotter = Trotter error only | ✅ | decreases ~1/κ, N∈{3,5,8} |
| L2 vs L3 (Aer DM) agree ≤1e-10 | ✅ | achieved ~**1e-15** |
| L3 vs statevector stochastic-reset avg ≤3·SE | ✅ | `test_L3_vs_statevector_trajectory_within_3SE` |
| L3 vs L4 finite shots (binomial) | ✅ | shot-emulation √S scaling test |
| L4 vs L5 noise attributable | ✅ | `test_noise_models.py` (5) |
| Ladder green N∈{3,5,8}, one command | ✅ | `make validate-ladder` |
| Suite passes | ✅ | **146 passed, 8 skipped** (~15s) |

## 2. Artifacts & keys

Modules (`src/qrc_single_time_series/quantum/`):
- `endianness.py` — **the ONE G2 crossing**. `logical_to_wire(i,N)=N-1-i`,
  `wire_to_logical`, `reverse_key`, `counts_to_z`, `counts_to_probabilities`.
  Circuits built with `logical_to_wire` return counts keys already in logical
  MSB-first order (`key[i]` = logical qubit i). Verified against Aer: X on logical
  qubit j → ⟨Z_j⟩=−1.
- `schedule.py` — `Schedule` (frozen: N, block_time, kappa, order, topology,
  gates=tuple of (name, sites_logical, angle)); `build_block_schedule(reservoir,
  block_time, kappa, order, ordering)`. First-order and 2nd-order (Strang) Trotter;
  `ordering` permutes bonds (Opt-NN hook). Sites are LOGICAL; wires resolved only
  at circuit build.
- `exact_trotter_qrc.py` — `gate_unitary` (closed form; XX/Z eigvals ±1),
  `block_unitary(schedule,N)`, `ExactTrotterQRC` (FN reservoir, matrix Trotter
  blocks, mirrors `ExactQRC.features`). Advances one virtual-node interval τ/V per
  node with κ substeps.
- `qiskit_qrc.py` — `block_circuit(schedule,N,qc)` (RXX/RZ on wires via
  endianness), `block_unitary_logical` (plain `Operator`, already logical basis —
  see §5 gotcha), `_theta(s)=2·arcsin√s`, `shot_emulate(z,S,rng)` (the consistent
  finite-shot model, G8-7), `QiskitReservoir` (Aer density_matrix; virtual nodes
  via `save_density_matrix`, ⟨Z⟩ via `DensityMatrix.expectation_value(Pauli,
  [wire])`).
- `noise_models.py` — `build_synthetic(cfg)` (depol 1q/2q, thermal relaxation,
  readout error; all-zero → ideal), `from_snapshot(path)` (frozen JSON →
  NoiseModel; no live calibration).
- `noisy_qiskit_qrc.py` — `NoisyQiskitReservoir` (QiskitReservoir + NoiseModel;
  deviation attributable to channels; optional shot layer).

Tests: `test_count_to_expectation.py` (7), `test_qiskit_parity.py` (13),
`test_noise_models.py` (5). Result: `results/metrics/validate_qiskit_qrc.json`.
Experiment registered `configs/experiments.yaml::registry.validate_qiskit_qrc`.

## 3. Achieved ladder parity (V=3, τ=2.0, κ=2, seed=7)

| N | Trotter err κ=1 | κ=4 | κ=16 | L2 vs L3 | L3 vs L4 shot-std (S=1024) | L4 vs L5 noise |
|---|---|---|---|---|---|---|
| 3 | 5.17e-2 | 1.44e-2 | 3.77e-3 | 6.7e-16 | 0.015 | 0.081 |
| 5 | 6.52e-2 | 1.30e-2 | 3.41e-3 | 8.9e-16 | 0.016 | 0.169 |
| 8 | 7.31e-2 | 8.35e-3 | 2.31e-3 | 2.0e-15 | 0.015 | 0.208 |

First-order Trotter error falls ~1/κ (confirmed). L2 vs L3 is machine precision —
the shared Schedule makes matrix and circuit identical up to floating point. Noise
deviation (depol 1e-3/7e-3, readout 0.015) grows with N as expected.

## 4. Pinned versions & memory

qiskit **2.5.0**, qiskit-aer **0.17.2** (match `requirements-lock.txt`). Dense DM
cost 16·4^N: N=8 → 1 MB (trivial), N=10 → 16 MB (FN cap). FN stateful track capped
N≤10 as designed. Ladder ran N=8 comfortably.

## 5. Decisions & deviations (READ before P4/P6)

- **`block_unitary_logical` uses plain `Operator(qc)`, NOT `reverse_bits()`.** Since
  `block_circuit` already maps logical i → wire N-1-i, the plain Operator is
  already in the project's logical MSB-first basis (matches matrix `block_unitary`
  to 2e-16). An initial `reverse_bits` double-reversed (0.03 error). The DM-based
  feature path was correct throughout (routes ordering through qiskit +
  endianness) — only the matrix-comparison helper was fixed. Don't re-add reversal.
- **Readout error at the DM level is a no-op on saved density matrices** (it is a
  measurement-time channel). Documented in `test_readout_error_pulls_z_toward_zero`.
  On the FN track the finite-shot layer (`shot_emulate`) carries the sampling
  spread; a literal readout-flip belongs to a true-sampling / backend-snapshot path
  (P10). Do NOT expect DM features to move under `readout_p`.
- **Stinespring dilation / exact rewind statevector mode DEFERRED.** The P3
  soundness review describes it as `rewind_qrc.py`'s exact mode, but `rewind_qrc.py`
  is a P4/P6 module (stub). The ladder "rewind-style blocks" are covered here via
  NN-TFI schedules (`topology='nn'`, `build_block_schedule` on `nn_tfi`); full
  dilation (N=12→21 qubits pure statevector) lands when `rewind_qrc.py` is built.
  Memory measurements for dilation therefore NOT yet recorded — flag for P6.
- **backend snapshot not exported** — `scripts/export_backend_noise_model.py` still
  a stub; no frozen snapshot under `results/backend_snapshots/`. `from_snapshot`
  works but has no file to read yet; synthetic noise is the only live channel.
- No commit made (user hasn't asked).

## 6. Open issues / known gaps

- 8 skipped tests remain = P4 placeholders (autonomous, accumulated error, failure
  classification, feedback scaling, prediction horizon, recursive feedback, rewind
  predicted window, no-future access).
- `QiskitReservoir`/`NoisyQiskitReservoir` read the Z-family only (raise otherwise);
  ZZ/general observables via DM are a small extension when needed.
- Full multi-step Aer DM circuits rebuild the whole circuit per `features` call
  (fine for N≤8 test lengths; for long P8/P10 runs consider incremental `initialize`
  or the matrix path, which is exact and faster).

## 7. Entry instructions for Phase 4 (autonomous rollout — physics-free)

1. `make test` green first (146 passed, 8 skipped).
2. Build `evaluation/autonomous.py::rollout(model, warmup, n_steps, domain_policy,
   telemetry)` with the `warm(history)->state` / `step(state,u)->(state,ŷ)` model
   protocol. **Build and test it against a toy AR(1) model BEFORE any quantum
   object** (phase file mandate).
3. Domain policies A–D (clip / smooth / terminate / wider-encoding), all
   telemetered, none silent. `evaluation/accumulated_error.py` (CNRMSE/CRMSE/CNMAE,
   σ_train train-only). `evaluation/prediction_horizon.py` (H_error with the
   for-every-k prefix rule — a trajectory cannot re-validate after failing;
   H_skill/H_reliable/H_effective, report ALL components). `evaluation/failure_modes.py`
   (registry, preregistered thresholds in `configs/preregistration.yaml::failure_rules`).
   `evaluation/statistics.py` (block bootstrap + Diebold-Mariano Harvey correction).
4. Turn green the G5 trio + `test_accumulated_error`, `test_effective_prediction_time`
   (incl. the dip-back-under-ε curve that must NOT re-validate),
   `test_failure_classification`, `test_feedback_scaling`, `test_autonomous_no_future_access`,
   `test_recursive_feedback`, `test_prediction_horizon`.
5. Exit at spec acceptance items 8–13 green with the toy model; write `HANDOFF_P4.md`.

## 8. Suggested skills for the next session

`quantum-reservoir-computing` + `qrc-project-playbook` (min per G10); add
`citation-audit` only when a document ships.
