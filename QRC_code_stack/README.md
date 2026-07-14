# qrc-stack

Staged repository for the QRC programme of the LaTeX handbook
(`../LaTeX_logbook/`, Parts IX-XI). One question per stage, one directory per
stage; work proceeds strictly up the ladder (Part IX, sec. "Promotion
protocol").

## Repository law (Part IX, R1-R6)

- **R1 Anchors first.** No stage accepts code before its validation anchors
  exist and fail meaningfully when sabotaged.
- **R2 The exact reference defines the ceiling.** `stage1_numpy_core/qrc_core.py`
  is the semantic definition; framework ports validate against it at
  identical seeds.
- **R3 One endianness point.** `counts_to_features` performs the single
  `key[::-1]` reversal (in `stage2_circuits_exact/qrc_qiskit.py`); nothing
  else may reorder bits.
- **R4 Printed configuration, fixed seed.** `SEED = 7` project-wide; every
  experiment prints its config block; no result enters a table unless a
  script printed it.
- **R5 Negative controls are first-class.** SK / SYK4 zero-compression rows,
  Haar-random reservoir, mean-predictor NMSE = 1.0 line ship in the same
  harness as headline configurations.
- **R6 Promotion gates, not vibes.** A stage is done when its exit script
  passes with exit code 0; downstream imports only from done stages.

## Stage ladder

| # | Directory | Question | Status (2026-07-14) |
|---|-----------|----------|--------|
| 0 | `stage0_anchors/` | do our instruments work? | GREEN - 77 tests (core/noise/family/reuse/RF-QRC/batch/product anchors) |
| 1 | `stage1_numpy_core/` | exact small-scale QRC (NumPy) | GREEN - Part VII regression numbers reproduced exactly |
| 2 | `stage2_circuits_exact/` | same reservoir as circuits (exact) | GREEN - qiskit gate 1.4e-15 at seed 7; cirq + pennylane ports run |
| 3 | `stage3_noisy_simple/` | noisy simple Hamiltonian | GREEN - noise anchors + degradation gates; full grid pending |
| 4 | `stage4_hamiltonian_battery/` | complex Hamiltonians | GREEN - RMT anchors + crossover gate; compression benchmark ALL PASS |
| 5 | `stage5_qubit_reuse/` | qubit reuse on those | GREEN - full suite ALL PASS (22.7 s) |
| 6 | `stage6_rfqrc/` | independent RF-QRC | GREEN - anchors + pilots; pre-registered study pending (--full) |
| 7 | `stage7_integration/` | RF-QRC x reuse | GREEN (batch+compress gates) - solar study plan-level |
| 8 | `stage8_product/` | product pipeline | GREEN - six-layer skeleton + walk-forward demo on labelled surrogate |

Full provenance of every copied file: `docs/MOVE_MAP.md`. How to run
everything, including data acquisition: `RUNBOOK.md`.

## Running

```
python run_tests.py            # anchors-first driver; exit 0 iff all green
python run_tests.py --stage 0  # single stage
```

Dependencies: numpy, scipy, matplotlib, statsmodels (ENSO data),
scikit-learn; qiskit + qiskit-aer (stages 2-5); qutip
(stage 4 `hamiltonians.py` diagnostics); cirq and pennylane (stage 2
optional ports). Pinned in `pyproject.toml`.

## Claim discipline

Everything in this repository is **simulation** unless a stage report says
otherwise; surrogate data is labelled wherever a number appears; "quantum
advantage" claims are banned outright (Part X, sec. "Pre-registered success
criteria").
