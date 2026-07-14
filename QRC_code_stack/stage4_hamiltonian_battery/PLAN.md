# Stage 4 plan - Hamiltonian battery merge + family scan

Status: BUILT AND GREEN (Phase 3, 2026-07-14). `hamiltonians.py` is now
the numpy-only merged module (qutip dropped per project decision):
dense_from_terms() bridges the benchmark PauliTerm specs to the spectral
diagnostics, one Hamiltonian definition serving both consumers.
`exp_family_scan.py` gates on the RMT anchors (GOE 0.5307 / Poisson
0.3863, 3-sigma tolerance) plus the disorder crossover (W=1 GOE side,
W=12 Poisson side) - all pass. Anchors in
stage0_anchors/test_family_anchors.py (incl. uncorrelated-spectrum
sabotage). Full seven-family compression benchmark run 2026-07-14: ALL
CHECKS PASSED (chains 10->5, TFIM-2D 12->7, SK/SYK4 honest zero
compression); artifacts benchmark_results.json + two figures. Known
physics caveat, printed by the scan: clean uniform chains keep
parity/reflection symmetry, so the per-family <r> table is descriptive
only; the gated crossover uses disorder to break symmetries. The original
plan follows for reference.

## What exists (copied, verified in their source projects)

- `benchmark_hamiltonians.py` (reuse project): the seven families as
  arbitrary Pauli strings + first-order Trotter circuits - TFIM-1D,
  mixed-field Ising, XXZ, XY, TFIM-2D (3x4), SK all-to-all, JW-mapped SYK4.
- `hamiltonians.py` (edge-of-chaos project): dense-matrix family builders
  with the disorder dial W and level-spacing diagnostics; carries the
  <r> anchors 0.386 (Poisson) / 0.531 (GOE). Depends on qutip.

## Planned merge (Phase 3)

Target: single `hamiltonians.py` per the Part IX tree.

- One family registry: `FAMILIES: dict[str, FamilySpec]` where a spec
  yields BOTH the Pauli-string form (feeds Trotter circuits, stage 5) and
  the dense form (feeds <r> diagnostics). Keep the Pauli-string
  representation primary; dense built from it (removes the qutip
  dependency if a numpy kron path is enough - decide during merge).
- `mean_r(H, W) -> float` level-spacing ratio, with the two anchors as
  stage-0 tests: Poisson limit 0.386, GOE limit 0.531 (tolerance from the
  source project's tests; sabotage: shuffle eigenvalues -> anchor red).
- Disorder dial `W` threaded through every family constructor.

### `exp_family_scan.py` (new)

Scan <r> versus W per family, printed table + one figure; marks each
family's integrable/chaotic side. Config block printed (R4); seed 7.

## Exit criterion

<r> diagnostics hit the 0.386/0.531 anchors on the analytic limits; the
family scan runs end-to-end and prints every number that enters the
figure.
