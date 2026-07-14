# Stage 3 plan - noisy simple Hamiltonian (TFIM-1D)

Status: BUILT AND GREEN (Phase 3, 2026-07-14). `noise_models.py`
(extraction + thermal variant) and `exp_tfim1d_noisy.py` exist; anchors in
stage0_anchors/test_noise_anchors.py (incl. sabotaged-Kraus red test);
gates pass (`python noise_models.py`; `python exp_tfim1d_noisy.py
--check` with stage5 on PYTHONPATH). Deviations from the original plan:
the experiment reuses stage 5's verified brickwork-ZZ restart circuit
(a Trotterised TFIM-1D-with-drive step) and its delayed-memory task
instead of a new circuit family; `transpile_ring` was not extracted
(plain transpile to the rz/sx/x/cx basis suffices at this stage). The
original plan follows for reference.

## Question answered

How do the Stage-1/2 results degrade under gate-local noise on the
simplest Hamiltonian family?

## Files

### `noise_models.py` (extraction, not new code)

The depolarizing + readout noise construction already exists inside
`stage2_circuits_exact/qrc_qiskit.py` (its `noisy` mode) and
`qiskit_qrc_hw.py`. Extract into one module so stages 3-5 share a single
noise definition:

- `depolarizing_model(p1, p2, p_ro) -> NoiseModel` - 1q/2q depolarizing +
  readout error, attached to the ECR/RZ/SX/X basis.
- `thermal_model(t1, t2, gate_times) -> NoiseModel` - thermal relaxation
  (new; parameters defaulted to published IBM-device medians, values cited
  in the stage report).
- `transpile_ring(circ, n) -> QuantumCircuit` - ring coupling map +
  native-basis transpilation, moved from `qrc_qiskit.py` unchanged.

Anchor (append to stage 0): a depolarizing channel with p=0 reproduces the
noiseless counts distribution (TVD < shot floor); Kraus completeness
sum K_i^dag K_i = I to 1e-12 (sabotage: break one Kraus normalisation ->
suite must turn red).

### `exp_tfim1d_noisy.py` (new experiment script)

- Config block printed at runtime (R4): n=5 ring, gamma=pi/4, seed 7,
  shots in {256, 1024, 4096, 16384}, p2 swept log-spaced 1e-4..1e-1.
- Output: NMSE-vs-noise-strength and NMSE-vs-shots degradation curves for
  the ENSO task, exact ceiling line always plotted (playbook: measure the
  analytic ceiling before any shot study).
- Table rows: exact / sampled / noisy per shot count, mean line and
  persistence included (R5).

## Exit criterion (promotion gate)

`python exp_tfim1d_noisy.py --check` exits 0 iff (i) the p->0, S->inf cell
agrees with the Stage-1 exact NMSE within shot noise, and (ii) curves are
monotone-degrading outside error bars.

## Documentation caveat (carried from the reuse project, measured there)

Gate-local noise models CANNOT see the 3-5x depth inflation qubit reuse
introduces. Stage-5 noise claims must never be extrapolated from Stage-3
models alone. This caveat ships in the stage report verbatim.
