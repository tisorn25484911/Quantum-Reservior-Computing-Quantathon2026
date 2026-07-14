# HANDOFF - qrc-stack repository build

Last updated: 2026-07-14, end of Phase 5.

## Resume here (next session)

State: stages 0-7 GREEN under `./.venv/bin/python run_tests.py` (exit 0;
slowest gates: stage 6 ~4 min, stage 7 ~1 min). Stage 8 is PLAN.md only.
Git not initialised (open user decision, with the docs/-PDF question).

Phase 6 scope: read `../LaTeX_logbook/part11.tex` IN FULL, convert Part
XI Phases 8-12 into a detailed structural plan for stage8_product/
(file-by-file responsibilities, interfaces, promotion gates). Plan-only
unless the user upgrades scope. Then Phase 7: stage reports in docs/,
citation audit, final handoff.

Carried forward: stage-7 solar study awaits a user data-download
decision (SURFRAD/NSRDB); stage-6 --full study needs ESN sweep +
windowed/recurrent-ring comparators; stage-3 full noise grid; stage-1
namespace merge.

How to run things:
    ./.venv/bin/python run_tests.py              # full ladder
    ./.venv/bin/python run_tests.py --stage 7    # one stage
    cd stage7_integration && \
        PYTHONPATH=../stage5_qubit_reuse:../stage6_rfqrc \
        ../.venv/bin/python exp_rfqrc_reuse.py    # full scan

Suggested skills for the next session: /qrc-project-playbook,
/quantum-reservoir-computing; /arxiv + /citation-audit for Phase 7;
/task-clarifying-questions if part11's product scope is ambiguous.

Working directory: `Quantathon2026/QRC_main_stack/QRC_code_stack/`
Design source: `../LaTeX_logbook/part9.tex` (stage ladder, rules R1-R6,
repo tree) and `part10.tex` (RF-QRC theory + phased plan). Part XI
(stage 8) not yet read in detail.

## Phase plan (agreed workflow: confirm with user after each phase)

| Phase | Scope | Status |
|---|---|---|
| 1 | Inventory sources, resolve duplicates, create stage skeleton, copy stages 0-5 seed files, PLAN.md for new-build stages, README/MOVE_MAP/driver/pyproject | DONE |
| 2 | Environment + import repair; stages 0-2 green under `run_tests.py`; reproduce stage-1 regression numbers (NARMA-10, memory capacity, ENSO from handbook Part VII) | DONE |
| 3 | Stages 3-5: extract `noise_models.py`, build `exp_tfim1d_noisy.py`, merge the two Hamiltonian files + `exp_family_scan.py` (<r> anchors 0.386/0.531), extract `test_reuse_anchors.py`, `exp_reuse_compression.py`; stages 3-5 green | DONE |
| 4 | Stage 6 RF-QRC: expand PLAN.md to signatures, then implement Part X Phases 0-4 (anchors -> core -> read-out -> metrics/battery -> MFE + Lorenz-63) | DONE (this handoff) |
| 5 | Stage 7: `qreuse_batch.py`, batch-and-compress validation, solar data plan/vetting, `exp_solar_ramps.py`; vn-fusion prototype only if IR extension stays small | DONE (this handoff; solar = plan-level, vn-fusion deferred as anticipated) |
| 6 | Stage 8: read part11.tex fully, produce detailed structural plan (plan-only unless user upgrades scope) | pending |
| 7 | Docs: stage reports in docs/, citation audit of anything cited, final handoff | pending |

## What Phase 1 did

- Created the stage-ladder skeleton exactly per Part IX Listing (repo tree).
- Copied all verified seed code for stages 0-5. Full provenance table:
  `docs/MOVE_MAP.md`. Sources untouched (copies, not moves).
- Canonical-source decision: `LaTeX_logbook/code/` (2026-07-13) beats
  `QRC_Climate_Prediction_complete/code/` (2026-07-05) for the five files
  they share; climate contributes its unique `baselines.py`,
  `qrc_qiskit.py`, `forecast.py`, `tests/`. The `qubit_reuse_protocol`
  root-vs-`qreuse_project` duplicates were byte-identical (verified with
  cmp) - `qreuse_project/` used.
- Third seed source discovered and used: `~/Desktop/qrc-edge-of-chaos`
  supplies `tasks.py` (stage 1) and `hamiltonians.py` with the <r>
  level-spacing anchors (stage 4).
- PLAN.md written for stages 3, 4 (merge+scan), 6, 7, 8 - per user rule,
  completely-new components are planned in detail, not built.
- Root `run_tests.py` driver (thin orchestrator; plan-only stages SKIP).

## What Phase 2 did (all verified by running, 2026-07-14)

- MAJOR CORRECTION to Phase 1's assumption: logbook and climate
  `qrc_core.py`/`datasets.py`/etc. are DIFFERENT MODULE FAMILIES sharing
  names, not versions. Restructured: logbook family at `stage1_numpy_core/`
  root (the R2 exact reference), climate ENSO family in
  `stage1_numpy_core/enso_app/`. See MOVE_MAP correction note.
- Environment: system python3.13 bare; conda base BROKEN (numpy 2.4.6 vs
  scipy/pandas compiled for numpy 1.x - scipy.linalg hard-fails; do NOT
  use). Created repo-local `.venv` from /opt/anaconda3/bin/python (3.11.8):
  numpy 2.4.6, scipy 1.17.1, qiskit 2.5.0, aer 0.17.2, pandas,
  statsmodels, scikit-learn, pytest, qiskit-ibm-runtime, cirq-core,
  pennylane. Driver: `./.venv/bin/python run_tests.py`.
- Import repair: `stage0_anchors/conftest.py` (adds enso_app + stage2 to
  path; the tests' own old-layout inserts are harmless no-ops, files
  otherwise unchanged). One 1-line shim in `enso_app/forecast.py`.
- Copied the pinned ENSO cache so no network fetch is needed.

## Green results (regression targets reproduced)

- Stage 0: 37/37 tests pass.
- Stage 1: `qrc_core.py` validation suite all-pass. `experiments.py`
  results.npy matches the logbook reference EXACTLY on all 12 scalars
  (narma_qrc 0.283, enso_qrc 0.596 = Part X's "recurrent ~0.60", etc.).
  `enso_app/forecast.py` full run matches source project's metrics.json to
  1e-13 (NMSE 1.288, battery rows identical).
- Stage 2: qiskit gate max|qiskit_exact - numpy_ref| = 1.43e-15; cirq port
  NMSE 0.2774; pennylane port ENSO 3-mo NMSE 0.7465 (= Part X's "windowed
  ~0.75"). Driver runs qiskit+cirq; pennylane manual (slow).
- Stage 5 (bonus, ahead of schedule): `run_stage5_tests.py --quick`
  passes as copied - reuse anchors + equivalence gates green.
- `run_tests.py` full ladder: stages 0,1,2,5 PASS; 3,4,6,7,8 SKIP (planned).

## What Phase 3 did (all verified by running, 2026-07-14)

- Stage 3 BUILT: `noise_models.py` (consolidates the two source noise
  constructions + new thermal_model; self-test green) and
  `exp_tfim1d_noisy.py` (degradation curves on stage 5's verified
  brickwork circuit; all three gates pass). See stage3 PLAN.md header for
  deviations from the original plan.
- Stage 4 BUILT: `hamiltonians.py` rewritten numpy-only (user decision;
  qutip dropped); `dense_from_terms()` bridges benchmark PauliTerm specs
  to spectral diagnostics. `exp_family_scan.py` gates green: RMT anchors
  GOE 0.5335/Poisson 0.3853 vs 0.5307/0.3863 targets; crossover W=1 ->
  0.5147 (GOE side), W=12 -> 0.3789 (Poisson side). Physics caveat: W=0 is
  contaminated by reflection symmetry - scan grids start at W>0.
- Stage 5 FULL suite green (22.7 s): 12 logical -> 4 physical, feature
  corr 0.9924, width scan constant physical=4 for logical 8-40, ENSO-ring
  honest zero compression. Seven-family compression benchmark green
  (chains 10->5, TFIM-2D 12->7, SK/SYK4 zero compression; artifacts in
  stage4/: benchmark_results.json + 2 png). Part IX's
  `exp_reuse_compression.py` is SATISFIED by benchmark_hamiltonians.py -
  no wrapper written (documented decision).
- Stage 0 grew from 37 to 47 tests: test_noise_anchors.py (incl.
  sabotaged-Kraus red test), test_family_anchors.py (incl.
  uncorrelated-spectrum sabotage), test_reuse_anchors.py (runs the stage-5
  quick suite as single source of truth). All 47 pass.
- Driver: stages 3 and 4 wired in; full ladder = 0,1,2,3,4,5 PASS,
  6,7,8 SKIP (planned).

## What Phase 4 did (2026-07-14)

- Papers verified live BEFORE implementing (playbook rule):
  arXiv:2405.03390 = PRR 6 043082 (2024); QMI 2025 optimal-training paper;
  MFE nine-mode ODEs transcribed verbatim from the ORIGINAL
  Moehlis-Faisst-Eckhardt NJP 6, 56 (2004) author-hosted PDF (IOP page
  403s; use sites.me.ucsb.edu/~moehlis). Two structural anchors protect
  the transcription: laminar fixed point exactly stationary; quadratic
  terms conserve energy to 1e-15.
- Stage 6 BUILT: mfe_model.py, rfqrc_reservoir.py (RFQRCConfig exactly
  per Part X listing; exact snapshots + shots-mode truncated circuits;
  R3 honoured - counts_to_features/probs_to_features single reversal),
  rfqrc_baselines.py (NVAR/ESN/ridge/GCV, fairness-rule harness),
  rfqrc_metrics.py (PH Racca-Magri ladder, event F-score vs offset, VPT;
  effective_rank imported from stage 5), exp_lorenz63_check.py,
  exp_mfe_extremes.py (--check pilot / default medium / --full
  pre-registered study).
- Stage 0 grew 47 -> 61 tests (test_rfqrc_anchors.py: the full Part X
  anchor table, 14 tests). Phase-1 acceptance: exact vs 1e6 shots agree
  at 8.9e-4 on 4 qubits.
- TWO PILOT-CAUGHT BUGS, both fixed and documented in the code: (i) with
  more input channels than qubits the encoder silently dropped channels
  (blind reservoir); (ii) closed-loop rollouts did not warm the leak
  integrator on true history before closing the loop.
- HONEST PILOT FINDINGS (reduced scale; NOT citable comparisons):
  Lorenz-63 pilot: rfqrc VPT 5.4 LT > nvar 3.7 > esn 0.6 - the DECLARED
  expectation (classical wins) was violated at pilot scale, but the ESN
  ran untuned defaults (rubric #2) - full run must sweep ESN
  hyperparameters. MFE pilot: NVAR dominated (PH 1.25 LT vs 0 for all
  others at one scale) and effective rank collapsed (4.4 of 1024
  nominal) - EXACTLY the two failure modes Part X kept in view. The
  rank-vs-N_v curve is a primary figure of the full study.
- Pre-registered criteria remain FROZEN and unevaluated: they are tested
  once, by `exp_mfe_extremes.py --full` (n 8-11, N_v {1,2,4,8}, eps/tau
  log grids, gamma set, m {1,2,3}, 5 seeds; hours-days; writes
  mfe_results.json incrementally). Nothing in Phase 4 constitutes a
  performance claim.

## What Phase 5 did (2026-07-14)

- `stage7_integration/qreuse_batch.py`: batch_circuits / compile_batch /
  compiled_width_of_batch / unbatch_counts over the UNTOUCHED stage-5
  compiler (frozen imports only). Endianness: unbatch only PARTITIONS
  count keys; counts_to_features stays the single semantic reversal.
- `stage7_integration/exp_rfqrc_reuse.py`: ALL promotion gates PASS --
  TVD(direct,reuse) 0.0385 vs floor 0.0357 (self-calibrated gate);
  feature interchangeability 0.0405 within 6-sigma shot bound 0.0938;
  width: logical 12 -> physical 4 (B=3, n=4); constant physical=4 for
  B in {2,4}; THE STRUCTURAL CLAIM measured live: windowed protocol
  physical grows 2->6 as T goes 2->6 while the rfqrc batch stays 4
  regardless of T. Topology: haar (all-to-all) block = no within-block
  compression, cross-block reuse still compresses 16->4; haar blocks
  need transpile to (u,cx) before the frozen parser.
- Stage 0 grew 61 -> 64 tests (test_batch_anchors.py: hand-built
  unbatch partition; batched-vs-standalone block distributions to 1e-12;
  compiled-batch legality).
- Solar-ramp study NOT built (plan-level in stage7 PLAN.md): needs a
  user decision on data download (SURFRAD via NOAA / NSRDB via API key)
  plus the provider-page vetting pass. vn-fusion prototype deferred
  (declared default).

## Still open

- Namespace merge of the two stage-1 families deferred (works as-is).
- Stage-3 full grid (`exp_tfim1d_noisy.py` without --check) not yet run.
- Stage-6 full study (`exp_mfe_extremes.py --full`) not launched: long
  compute; needs an ESN hyperparameter sweep added for fairness (rubric
  #2) before its comparisons are citable - flagged in the module
  docstring. Also missing from the full-run comparator set: the windowed
  protocol and the stage-1 recurrent ring (Part X Phase 4 asks for both).
- Stage-7 solar study: awaiting user decision on data download; then
  solar_ramps_data.py (vetting-gated) + exp_solar_ramps.py per PLAN.md.
- Phase 6 next: read part11.tex fully -> detailed stage-8 structural
  plan (plan-only unless user upgrades scope). Then Phase 7: docs,
  stage reports, citation audit.
- exp_mfe_extremes --check takes ~4 min (n=9 exact sim dominates the
  driver's stage-6 gate; acceptable but the slowest gate in the ladder).

## Open decisions for the user (carry to next session)

1. Should `docs/` also hold a copy of (or pointer to) the compiled
   handbook PDF, per Part IX tree's `docs/ # this handbook`?
2. Stage-4 merge may drop the qutip dependency if a numpy path suffices -
   preference?
3. git init for QRC_code_stack (it is not a repo yet)?

## Suggested skills for the next session

- `/qrc-project-playbook` - build order, anchors, gotchas (endianness,
  encoding-gain sweep) - load before touching stage code.
- `/quantum-reservoir-computing` - domain physics for stages 4/6 design.
- `/arxiv` + `/citation-audit` - only when Phase 4+ needs paper checks
  (Ahmed 2024/2025 RF-QRC papers, NVAR/Gauthier 2021, arXiv:2508.12383).
- `/task-clarifying-questions` - if any phase's acceptance criteria feel
  underspecified, ask before building.
