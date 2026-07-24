# PLAN.md — QRC_single_time_series

Derived from `IMPLEMENTATION_PHASES.md` (13 phases P0–P12) and the master
specification (33 sections). This is the required §6 first deliverable; the
phase file is the authoritative build order, this file is the standing plan.

Status: **Phase 3 complete** (see `handoffs/HANDOFF_P0.md`, `HANDOFF_P1.md`,
`HANDOFF_P2.md`, `HANDOFF_P3.md`). Phases 4–12 pending user confirmation, one at a time.

---

## 1. Research objectives

1. Implement Quantum Reservoir Computing (QRC) in three faithful forms — exact
   matrix (NumPy/SciPy), ideal gate-based Qiskit, and noisy Aer simulator — that
   agree by construction, and quantify each error source between them.
2. Forecast three observational climate indices (ENSO/PDO/SOI in `data/raw/`) as
   independent univariate problems, in **two regimes kept strictly separate**:
   (A) teacher-forced one-step, and (B) fully autonomous closed-loop, where after
   a forecast origin the model consumes only its own predictions.
3. Answer the central question: **for how many steps/months can a trained QRC
   forecast autonomously before leaving a pre-registered error band**, and how
   that horizon depends on shots, noise, launch month, and dataset.
4. Separate five questions that are usually conflated: scientific forecast skill;
   autonomous dynamical stability; quantum-vs-classical computational value;
   operational usefulness for renewable-energy decisions; product feasibility.
5. Make no claim of quantum advantage, commercial readiness, or renewable-energy
   value without a dedicated experiment supporting it. Negative results ship.

## 2. Hypotheses (falsifiable; each mapped to a phase/gate)

- H1 (mechanism): the exact FN reservoir reproduces the Mackey–Glass
  teacher-forced→autonomous transition — non-chaotic tracking (τ_MG=16), finite
  valid time then attractor-like divergence (τ_MG=17). *Gate, Phase 5.*
- H2 (skill): on the untouched test span, the QRC beats persistence at h=1–3
  months on ≥1 index (pre-registered Gate-1 criterion). *Phase 9.*
- H3 (parity, not speed-up): the QRC is at best competitive with an
  effective-rank-matched ESN / NVAR / small LSTM-GRU; any edge is
  parameter-efficiency, verified across seeds/origins. *Phases 7, 9, 10.*
- H4 (shot compounding): autonomous valid time grows like a + (1/2λ)·ln(S) for
  chaotic systems; deviations are attributed, not forced. *Phase 8/10.*
- H5 (noise≠free lunch): any "noise helps" effect is recovered by a matched
  classical ridge/SVD-truncation without noise. *Phase 5 (setup), Phase 10.*
- H6 (feasibility): closed-loop rewind on real hardware costs one QPU round-trip
  per step; usefulness is latency-vs-decision-deadline, per use case. *Phase 11.*

## 3. Paper-by-paper implementation mapping

**Paper A — Fujii & Nakajima, PR Applied 8, 024030 (2017).** Sequential quantum
input replacement `ρ→ρ_s⊗Tr_inj ρ`; disordered fully-connected transverse-field
Ising `H=Σ_{i<j}J_ij X_iX_j + Σ_i h Z_i`, `J_ij~U[−J/2,J/2]` frozen per seed;
local Z observables; V virtual nodes (read at `t+(v+1)τ/V`, v=0..V−1 — errata
G8-4); linear readout; teacher forcing; autonomous closed loop; Mackey–Glass.
→ `quantum/{pauli,hamiltonians,input_channels,exact_qrc,fujii_nakajima_qrc}.py`,
`dynamical_systems/mackey_glass.py`, `evaluation/autonomous.py`.

**Paper B — Hamhoum et al., arXiv:2510.13634v2.** Injection + memory qubits;
rewind-window feature extraction (reset + re-encode over a length-`t_w` window);
nearest-neighbour TFI, Opt-NN gate ordering; first-order Suzuki–Trotter
(RXX(2Jτ/κ), RZ(2hτ/κ)); local Z; finite-shot; ridge readout. Its target is
**next-step** `y(t)=f_QRC(u_{t−1})≈u_t`; its "ENSO" is a **generated ODE** (the
Vallis 1986 box model), **never** conflated with the observational `enso.csv`.
→ `quantum/{schedule,rewind_qrc,qiskit_qrc,noisy_qiskit_qrc}.py`,
`dynamical_systems/enso_ode.py`.

## 4. Mathematical definitions (implemented in P1–P2)

Injection state `ρ_s = ½(I + 2√(s(1−s))X + (1−2s)Z)`, `s∈[0,1]`. Step
`x(kτ)=U_τ S_k x((k−1)τ)`, `U_τ=e^{−iHτ}`. Virtual-node signals
`x'_{n,v}=(1+⟨Z_n(t+(v+1)τ/V)⟩)/2`, plus bias. Readout `ŷ=W_out r`,
`W_out=Y R⁺` (pinv) or ridge `W_out=YRᵀ(RRᵀ+βI)⁻¹`, β on dev only. NMSE in
**both** conventions: variance-normalised (mean-predictor=1) and FN Eq. A1
(`/Σȳ²`) — labelled everywhere (errata G8-5).

## 5. Task taxonomy (kept in separate modules; spec §4.2)

Same-variable autonomous forecasting (MG, Lorenz, Vallis-ENSO, enso/pdo/soi) vs
input→output memory/capacity tasks (STM, PC, NARMA) — the latter are **not**
forced through the closed-loop interface. Two tracks never share a table:
**A controlled synthetic** (MG, Lorenz-63, Vallis ENSO ODE) and **B
observational** (enso/pdo/soi).

## 6. Data workflow (Phase 0 done)

Raw CSVs pinned + checksummed (`data/manifests/`), identities resolved (ENSO =
raw Niño 1+2 SST; SOI = CPC Tahiti−Darwin anomaly; PDO = Mantua index). One
chronological split in `data/loaders.py` (final 20% untouched test; rolling
origin inside the first 80%). Anomaly = subtract **train-years-only** monthly
climatology; scaler fit train-only; transforms are ablation arms. EDA in P9.

## 7. Validation stages (the ladder; P1→P3)

L1 exact dense `e^{−iHτ}` → L2 exact matrix Trotter → L3 Aer density-matrix →
L4 ideal finite shots → L5 noisy → (L6 optional real QPU, reported separately).
Each step isolates one error source. N=2/3 matrix-Trotter ≡ Aer-DM to 1e−10.

## 8. Model architecture

FN stateful reservoir (persistent ρ, N≤10 dense cap) and Hamhoum rewind
reservoir (re-init per window; exact via **Stinespring dilation** to a pure
statevector on `N+(t_w−1)d_inj` qubits — unlocks N=12–15 exactly). Linear ridge
readout for both; observable set is config.

## 9. Autonomous feedback procedure (P4)

`evaluation/autonomous.py::rollout(model, warmup, n_steps, domain_policy)` with
a no-future guard (G5). Stateful-FN keeps ρ across steps; rewind shifts a
provenance-masked buffer until it is all predictions. Domain policies (clip /
smooth / terminate / wide-encoding) all telemetered; clipping never hidden.

## 10. Noise and shot experiments (P3, P10)

Synthetic channels (readout, 1q/2q depolarising, thermal relaxation) + a frozen
backend snapshot (no live calibration). Shot grid {128…8192}×seeds. Consistent
shot-emulation formula across all models. Signal-to-noise ceiling check gates
every noise study. Three-arm adjudication: noise vs matched-ridge vs
SVD-truncation.

## 11. Classical and quantum baselines (P7)

mean, persistence, seasonal persistence, ridge-AR, ARIMA/SARIMA, ESN
(effective-rank size-matched **and** best-of-class), NVAR, LSTM, GRU, QLSTM +
one specified windowed VQC; Haar-random-reservoir control. All recursive
baselines feed their own predictions back through the same engine.

## 12. Statistical evaluation (P4+)

Teacher-forced: MSE/RMSE/MAE, both NMSEs, R², Pearson, anomaly correlation,
MASE, persistence skill. Autonomous: instantaneous/cumulative normalised error,
`H_error/H_skill/H_reliable/H_effective`, survival curve, saturation rate,
failure-mode frequencies, phase/amplitude error. Block bootstrap over origins;
Diebold–Mariano (Harvey correction); effective sample size beside every CI.

## 13. Lyapunov analysis (P6, P8)

Benettin (Jacobian QR) + Rosenstein (scalar) cross-validated on Lorenz;
learned rewind-map Jacobian spectrum; valid time in Lyapunov units for
controlled systems only. Observational indices: divergence estimates only if
robust; horizons reported in calendar months. Lorenz β=8/3 vs printed 3/8 and
the Vallis ODE audited before use (errata G8-1/2/3).

## 14. Failure-mode classification (P4)

Registry with pre-registered dev-tunable thresholds: blow-up, fixed point, mean
collapse, variance collapse/explosion, spurious limit cycle, boundary
saturation, stochastic instability. Multiple labels allowed; first-failure time
recorded.

## 15. Computational budget (see IMPLEMENTATION_PHASES.md §3)

Dense ρ: N=7 trivial, N=10 fine (FN cap), N=12 borderline, N=15 infeasible.
Rewind dilated statevector: N=12→21 qubits (32 MB), N=15→24 qubits (268 MB) —
feasible & exact. Big noisy grids are configurable, never in `make all`.

## 16. Hardware feasibility (P11)

Per-step sequential-QPU latency ledger; strategy comparison (fresh circuits,
pre-transpiled templates, dynamic circuits, local sim, offline-features +
classical deploy, classical surrogate, periodic refresh); accuracy-vs-latency
Pareto. Batched open-loop runtime is **not** used as closed-loop evidence.

## 17. Renewable-energy product pathway (P12)

`PRODUCT_STRATEGY.md`: users, decisions, and ONE fully designed downstream
experiment (arms A–E: no-index / true-index / persistence-index /
classical-index / QRC-index) with the required operational dataset named.
Execution is out of scope; the design + data requirement is the deliverable.
Go/no-go Gates 1–7 evaluated from stored results only.

## 18. Risks and limitations (see LIMITATIONS.md)

Short observational records (hundreds–~2000 monthly points) → origin overlap,
block bootstrap and effective-sample-size mandatory. Dense-ρ memory wall (N≤10).
Spring-predictability barrier for ENSO; >12-month autonomous skill triggers a
leakage hunt. Shot-based rewind map is stochastic (no Lyapunov spectrum).

## 19. Objective acceptance criteria

The 28 spec §32 items, mapped to phases/tests in `IMPLEMENTATION_PHASES.md §4`.
Gate-1 success criterion is frozen in `configs/preregistration.yaml`.

## 20. Global invariants (bind every phase)

Layout G1; qubit-order contract G2 (`quantum/endianness.py`, one reversal);
alignment contract G3 (`data/windows.py::align`); config-hash artifact keys G4;
no-future guard G5; frozen preregistration G6; honesty battery G7 (size-matched
ESN at effective rank, Haar control, mean-predictor line, parity-not-speed-up);
errata register G8; citation discipline G9; per-phase handoff G10.
