# MOVE_MAP - provenance of every file in qrc-stack

Date: 2026-07-14. Copies, not moves: sources left untouched.
Canonical-source rule used throughout: where the same file existed in two
projects, the newer copy shipped with the handbook (`LaTeX_logbook/code/`,
2026-07-13) wins; the climate project contributes only its unique files.

Source roots (all under `~/Desktop` unless noted):

- `LB` = `Quantathon2026/QRC_main_stack/LaTeX_logbook/code/` (handbook code, canonical)
- `CL` = `Quantathon2026/QRC_Climate_Prediction_complete/`
- `QR` = `Quantathon2026/qubit_reuse_protocol/qreuse_project/`
  (root-level duplicates in `qubit_reuse_protocol/` verified byte-identical)
- `EC` = `qrc-edge-of-chaos/src/qrc/`

## Stage 0 - anchors

| Destination | Source | Note |
|---|---|---|
| `test_core_anchors.py` | `CL/tests/test_anchors.py` | renamed per Part IX tree |
| `test_baselines.py` | `CL/tests/test_baselines.py` | |
| `test_datasets.py` | `CL/tests/test_datasets.py` | |
| `test_ent_scale.py` | `CL/tests/test_ent_scale.py` | |
| `test_forecast.py` | `CL/tests/test_forecast.py` | |
| `test_qiskit.py` | `CL/tests/test_qiskit.py` | |

`test_reuse_anchors.py`: NOT yet extracted - the reuse anchor suite lives
inside `stage5_qubit_reuse/run_stage5_tests.py` (its anchor section).
Extraction is a Phase-3 refactor. `test_rfqrc_anchors.py`: Phase-4 plan
(Part X Table of anchors).

All six copied tests still contain `sys.path.insert(..., parents[1]/"code")`
pointing at the OLD layout - import repair is Phase 2, so the suite is NOT
claimed green yet.

## Stage 1 - NumPy core

CORRECTION (Phase 2): the LB and CL files sharing names are NOT
older/newer versions of one module - they are two different module
families. LB `qrc_core.py` = Fujii-Nakajima recurrent density-matrix
reservoir (`QuantumReservoir`, narma10, memory_function,
run_validation_suite). CL `qrc_core.py` = windowed CZ-ring ENSO
application (`WindowedReservoir`, ridge_gcv, SEED). Part IX stage 1 needs
both (the ENSO reservoir "stays here as the long-memory comparator"), so
the climate family lives in the `enso_app/` subdirectory to avoid the
name collision. A true namespace merge is deferred (tracked in HANDOFF).

Stage-1 root (logbook family - the R2 exact reference):

| Destination | Source | Note |
|---|---|---|
| `qrc_core.py` | `LB/qrc_core.py` | `QuantumReservoir`; validation suite green 2026-07-14 |
| `datasets.py` | `LB/datasets.py` | solar_surrogate, load_surrogate, enso_real |
| `eda.py` | `LB/eda.py` | |
| `experiments.py` | `LB/experiments.py` | Part VII figures; results.npy reproduced exactly 2026-07-14 |
| `figstyle.py` | `LB/figstyle.py` | writes to `../figures/` |
| `tasks.py` | `EC/tasks.py` | self-contained (numpy only) |

`enso_app/` (climate family - windowed CZ-ring ENSO application):

| Destination | Source | Note |
|---|---|---|
| `enso_app/qrc_core.py` | `CL/code/qrc_core.py` | `WindowedReservoir` |
| `enso_app/datasets.py` | `CL/code/datasets.py` | real NOAA ENSO pipeline; cache at `../data/raw/enso.csv` (copied) |
| `enso_app/eda.py` | `CL/code/eda.py` | |
| `enso_app/experiments.py` | `CL/code/experiments.py` | Phase-5 operating-point sweep |
| `enso_app/figstyle.py` | `CL/code/figstyle.py` | |
| `enso_app/baselines.py` | `CL/code/baselines.py` | full battery + eval harness |
| `enso_app/forecast.py` | `CL/forecast.py` | MODIFIED (1-line layout shim): old `REPO/"code"` path insert now points at `stage2_circuits_exact/`; metrics.json reproduced to 1e-13 2026-07-14 |
| `data/raw/enso.csv` | `CL/data/raw/enso.csv` | pinned cache, avoids re-download |

## Stage 2 - circuits, exact

| Destination | Source | Note |
|---|---|---|
| `qiskit_qrc_hw.py` | `LB/qiskit_qrc_hw.py` | |
| `cirq_qrc_minimal.py` | `LB/cirq_qrc_minimal.py` | |
| `pennylane_qrc_hybrid.py` | `LB/pennylane_qrc_hybrid.py` | |
| `qrc_qiskit.py` | `CL/code/qrc_qiskit.py` | windowed-restart Qiskit port; owns `counts_to_features` (R3 single endianness point); exact/sampled/noisy modes |

## Stage 3 - noisy simple Hamiltonian

No files copied. `PLAN.md` in the stage directory: `noise_models.py` is an
EXTRACTION of the depolarizing/thermal-relaxation code already inside
`qrc_qiskit.py` and `qiskit_qrc_hw.py`; `exp_tfim1d_noisy.py` is new.

## Stage 4 - Hamiltonian battery

| Destination | Source | Note |
|---|---|---|
| `benchmark_hamiltonians.py` | `QR/benchmark_hamiltonians.py` | seven families as Pauli strings + Trotter |
| `hamiltonians.py` | `EC/hamiltonians.py` | level-spacing <r> diagnostics (0.386/0.531 anchors); depends on qutip |

Two overlapping definitions of the family battery - merging them into the
single `hamiltonians.py` of the Part IX tree is a Phase-3 task.
`exp_family_scan.py`: planned (see stage `PLAN.md`).

## Stage 5 - qubit reuse

| Destination | Source | Note |
|---|---|---|
| `qreuse_ir.py` | `QR/qreuse_ir.py` | unchanged |
| `qreuse_analysis.py` | `QR/qreuse_analysis.py` | unchanged |
| `qreuse_scheduler.py` | `QR/qreuse_scheduler.py` | unchanged |
| `qreuse_compiler.py` | `QR/qreuse_compiler.py` | unchanged |
| `qreuse_validation.py` | `QR/qreuse_validation.py` | unchanged (self-calibrated TVD gate) |
| `qrc_experiment.py` | `QR/qrc_experiment.py` | windowed protocol; step-circuit factory refactor due in Phase 4/5 |
| `run_stage5_tests.py` | `QR/run_tests.py` | renamed; becomes the stage-5 exit script |

NOT copied: `QR/qrc_reuse_additions/apply_qubit_reuse_additions.py`
(a one-shot patch script, already applied to its target) and
`QR/skill_update/` (skill docs, not code).

## Stages 6-8

Nothing copied - nothing exists. Each directory holds a `PLAN.md`; per the
project instruction, completely-new components are planned in detail, not
built, until their phase arrives.

## Deliberately left behind

- `CL/build_walkthrough.py`, `CL/notebooks/walkthrough.ipynb` - pedagogy for
  the old layout; regenerate after Phase 2 if wanted.
- `CL/CLAUDE.md`, `CL/HANDOFF.md`, `QR/HANDOFF.md`, `QR/REFLECTION.md` -
  historical project docs; superseded by the handbook + this repo's docs.
- `qrc-edge-of-chaos/` everything except `tasks.py`/`hamiltonians.py` - that
  project's PennyLane/QuTiP pipeline is a different architecture family;
  revisit only if Stage 4 needs its dynamics-analysis scripts.
