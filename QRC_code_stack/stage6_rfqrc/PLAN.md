# Stage 6 plan - recurrence-free QRC with virtual-node readout

Status: BUILT (Phase 4, 2026-07-14); Part X Phases 0-3 acceptance
green, Phase 4 pilot-validated. Papers verified live before
implementation: arXiv:2405.03390 = PRR 6 043082 (2024); QMI 2025
(s42484-025-00261-9); MFE ODEs transcribed verbatim from the original
Moehlis-Faisst-Eckhardt NJP 6, 56 (2004) author-hosted PDF, protected by
two structural anchors (laminar fixed point exactly stationary; quadratic
terms conserve energy to 1e-15 - a typo in any coefficient breaks it).

Built: mfe_model.py, rfqrc_reservoir.py, rfqrc_baselines.py,
rfqrc_metrics.py, exp_lorenz63_check.py, exp_mfe_extremes.py; 14 anchors
in stage0_anchors/test_rfqrc_anchors.py (full Part X anchor table).
Phase-1 acceptance: exact vs 1e6-shot truncation agree at 8.9e-4
(bound 3.5e-3) on 4 qubits.

Implementation notes vs the original plan below:
- entangler "ham_trotter:<family>" deferred to stage 7 (raises
  NotImplementedError); brickwork_zz + haar_control implemented.
- Encoding covers slots-vs-qubits in BOTH directions after a pilot-caught
  bug: with 9 channels on 6 qubits the original loop silently dropped
  channels 7-9 (a blind reservoir). Fix documented in _encode.
- Closed-loop rollouts warm the leak integrator on true history before
  closing the loop (second pilot-caught bug; ESN path was already warm).
- The pre-registered criteria are evaluated ONLY by exp_mfe_extremes.py
  --full (n 8-11, five seeds, full sweep; hours-days, writes
  mfe_results.json incrementally). --check pilots are pipeline gates and
  their comparisons are NOT citable (ESN untuned at pilot scale -
  rubric #2).

The original plan follows for reference.

## Architecture being implemented (Part X, sec. 1)

- (a) RF-QRC: |psi_k> = V Phi(u_k) Phi(u_k) |0>^n; classical leaky
  integrator r_k = (1-eps) r_{k-1} + eps phi(u_k); ridge read-out.
  Constant circuit depth in the memory horizon; per-step circuits
  independent (batchable through the stage-5 compiler).
- (b) Virtual-node readout: features at N_v intermediate evolution times;
  exact path = statevector snapshots at layer barriers (free); shots path
  = N_v truncated circuits (reuse-fused realisation deferred to stage 7).
- (c) Extreme-event battery: PH (Racca-Magri convention), F-score vs
  prediction-time offset, VPT, effective rank; baselines gain NVAR, QELM
  ablation (eps=1), and this repo's own recurrent reservoirs.

## Files (Part IX tree)

| File | Role | Seeds it extends |
|---|---|---|
| `rfqrc_reservoir.py` | step-circuit factory (shared with stage 5's windowed comparator), exact snapshot path, shots path via existing Aer runner, leak, washout | refactor of `stage5_qubit_reuse/qrc_experiment.py`'s circuit builder |
| `rfqrc_baselines.py` | NVAR, QELM ablation, + imports stage-1 battery (mean/persistence/linear-lags/ESN/Haar) | `stage1_numpy_core/baselines.py` |
| `rfqrc_metrics.py` | PH ladder, event F-score binned by offset, VPT, effective rank / participation ratio | new |
| `mfe_model.py` | nine-mode MFE shear-flow integrator (dt=0.25, Lambda=0.0163) | new |
| `exp_mfe_extremes.py` | Phase-4 anchor study, full sweep grid | new |
| `exp_lorenz63_check.py` | side check; classical reservoir expected to WIN at matched DoF - declared in advance | new |

Config surface: the `RFQRCConfig` dataclass exactly as printed in Part X
(n_qubits=10, window_m=1, n_uploads=2, entangler="brickwork_zz",
gamma=pi/4, tau=1.0, disorder_W=0, n_virtual_nodes=4,
readout="full_probs", leak_eps=0.3, shots=None, denoise="none",
ridge_beta=(1e-6,1e-9,1e-12), seed=7).

## Anchors to append to stage 0 (`test_rfqrc_anchors.py`, Part X table)

leak impulse response r_d = eps(1-eps)^d exact; leak memory function vs
analytic form; 2-qubit hand-solved virtual-node circuit; snapshot-vs-
truncation TVD < 1e-12; endianness on a known basis state; PH ladder on a
synthetic ramp; F-score degenerate cases; NVAR exact recovery of a known
quadratic map to 1e-10; Haar control distinctness; NMSE(mean) = 1.000.

## Acceptance (phase gates, Part X)

- Phase 0: untouched suites green, new anchors green with sabotage check.
- Phase 1: exact vs 1e6-shot truncation agree within shot noise (4 qubits).
- Phase 2: ridge anchor; ONE evaluation harness applies any head to all
  models (fairness rule).
- Phase 3: battery end-to-end on a synthetic series, every number printed;
  feature-count matching: ESN nodes = NVAR features = N_v * F_step.
- Phase 4: MFE study - published qualitative finding reproduced OR its
  failure reported; Lorenz-63 check run and reported either way.

## Pre-registered success criteria (frozen, Part X sec. 6 - do not edit)

MFE: median PH (best config, n=11) >= ESN plateau + 1.0 LT, >= Haar + 0.5
LT, >= NVAR at matched feature count. Any clause failing => the null IS
the result. Claims banned regardless of outcome: quantum advantage,
speed-up, hardware implications from Aer.

## Known failure modes kept in view

Effective-rank collapse despite N_v (the rank-vs-N_v curve is a primary
figure whatever it shows); NVAR dominance (if it wins, that is the
headline); the additive-across-delays limitation of the leak (no
cross-delay products - remedies: window m<=3 ablation, nonlinear head
under the fairness rule).
