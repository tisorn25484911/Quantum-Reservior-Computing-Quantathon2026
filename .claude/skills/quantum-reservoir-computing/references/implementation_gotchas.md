# QRC implementation gotchas and mechanistic findings

Non-obvious lessons from building and debugging a QRC codebase from scratch. Each
of these produces a reservoir that *looks* plausible (runs, returns bounded
features, trains without error) but learns nothing — and each cost real debugging
time to localize. Read this before writing or debugging QRC simulation code.

The unifying theme: **validate every component against a known exact answer
before using it in an experiment.** A silent bug in injection bookkeeping or a
non-trace-preserving channel does not raise an error; it just quietly destroys the
result. The anchors below are cheap to encode as unit tests and catch these.

## Validation anchors (build these as tests first)

| Component | Anchor | Expected |
|---|---|---|
| State re-injection | known product state, trace out one qubit, re-inject | rebuilt state, norm diff = 0 |
| Chaos diagnostic ⟨r⟩ | Atas et al. reference values | Poisson 0.3863, GOE 0.5307 |
| Ridge read-out | known affine map + intercept | exact recovery |
| Kraus instrument | Σ K†K = I | machine precision |
| Process-matched channel | exact Bayesian filter of an HMM | NMSE ≈ 1e-25 |
| Effective rank | i.i.d. vs rank-one features | ≈ n vs ≈ 1 |
| Metrics | perfect / constant-mean predictor | NMSE = 0 / 1 |

## Library API pitfalls (versions matter)

- **QuTiP 5 `rand_unitary` takes dimensions positionally.** QuTiP 4 used
  `rand_unitary(dim, dims=[[2,2],[2,2]])`; in QuTiP 5 there is no `dims` kwarg —
  the first positional argument is the flat subsystem list:
  `qt.rand_unitary([2,2,2,2], seed=...)`. A nested list double-wraps the dimension
  object and every later operator product fails with an incompatible-dimensions
  TypeError. Tensor identity is `qt.qeye([2,2,2])`.
- **PennyLane mid-circuit reset allocates a wire on `default.mixed`.**
  `qml.measure(0, reset=True)` allocates an auxiliary wire; on a device declared
  with exactly n wires this raises `WireError`. Fix: don't force the reset through
  the circuit API. Compile the layer to an explicit unitary with
  `qml.matrix(layer_fn, wire_order=range(n))()` and do injection/partial-trace/
  read-out in NumPy density-matrix form. The simulated object is still exactly the
  native-gate circuit.
- **QuTiP `mesolve` expectation values are complex.** Cast with `.real`
  explicitly, or a `ComplexWarning` fires and downstream code may misbehave when
  the true value is ~0 with round-off imaginary part.

## The chaos dial that does NOT work at small sizes

To sweep integrable→chaotic at N ≈ 8–10 (the largest exact-diagonalization size
inside a sweep), the naive knobs fail:
- **Longitudinal/transverse field ratio on an all-to-all Ising model:** no
  crossover — ⟨r⟩ stays ~0.47–0.51 across the whole range. The all-to-all point
  has extensive symmetry (permutation, parity); *unresolved symmetry sectors
  corrupt the ⟨r⟩ statistics*.
- **Tilted chain + weak disorder, sweeping h_z:** the chain is simply chaotic for
  every h_z > 0 at this size. The longitudinal field is a switch, not a dial.
- **What works: on-site disorder strength W as the dial** (thermal→MBL crossover),
  on a nearest-neighbour chain. Disorder both drives the crossover *and* breaks
  the symmetries that pollute the statistics, so no sector resolution is needed.
  Validated at N=10, ~8 realizations: ⟨r⟩ goes 0.534 (W=1, GOE) → 0.391 (W≈6–8,
  Poisson). Compute ⟨r⟩ over the central 50% of the spectrum and drop
  near-degeneracies (gap < 1e-12).

**Lesson:** thermodynamic-limit "integrable vs chaotic" knobs do not transfer
naively to small sizes; symmetry resolution is the hidden cost, and disorder pays
it for free.

## State re-injection bookkeeping (silent index errors)

Replacing one qubit of an entangled n-qubit state = partial-trace that site out,
tensor the fresh single-qubit state back *at the correct position*. After
`tensor(ρ_in, ρ_rest)` the re-inserted qubit is at index 0, so you must permute it
back to its original slot (`Qobj.permute` with the argsort of the reordering).
Index errors are silent — the state stays normalized and physical, just wrong.
Validate: build `|0⟩⟨0| ⊗ |1⟩⟨1| ⊗ |+⟩⟨+|`, trace out site 1, re-inject
`|+⟩⟨+|` at site 1, check norm diff = 0 and single-site ⟨Z⟩ = (+1, 0, 0). The same
check validates the NumPy reshape/transpose helpers used by the gate reservoir.

## Gate-reservoir memory mechanisms that erase the signal

The hardware-style reservoir must be *exactly* a native-gate circuit *and* have
real memory. Two designs that fail:
- **Classical state proxy** (carry per-wire ⟨Z_i⟩ as re-prepared angles): a rank-n
  classical bottleneck that discards all inter-wire correlations. NARMA-2 NMSE
  ≈ 0.98 (nothing).
- **Statevector threading with a global "leak" toward |0…0⟩ + renormalize:** a
  nonphysical uniform contraction that erases precisely the linear input trace
  NARMA needs. NMSE stays 0.93–1.05 across all leak fractions. (Diagnostics show
  feature std is healthy and read-out endianness is correct — so the plumbing is
  fine; the *mechanism* is wrong.)
- **What works:** compile one Trotter layer to its unitary, then do the
  Fujii–Nakajima protocol (reset only the input qubit) in NumPy density-matrix
  form, reading at intermediate Trotter depths for temporal multiplexing.

**Lesson:** the contraction mechanism is not interchangeable. Input-qubit reset
removes only stale information; a global leak destroys task-relevant correlations.

## Channel/QHMM: the trace-preservation trap

Realizing a *given* HMM as a QHMM channel: the obvious constructions fail
Σ K†K = I. One Kraus per symbol, `K_x[j,i] = √(T_ij E_ix)`, has non-cancelling
off-diagonal terms; one Kraus per (symbol, destination) fails identically — the
square root couples the source index to the data. **Correct:** one Kraus per
(symbol, source, destination) triple, `K_{x,i,j} = √(T_ij E_ix)|j⟩⟨i|`, giving a
diagonal `K†K = T_ij E_ix |i⟩⟨i|` that sums to I by row-stochasticity. This
decohering realization keeps the memory state diagonal and equal to the exact
Bayesian filter belief — validate to NMSE ≈ 1e-25.

**Contraction-time matching.** Generic random instruments fail on memory tasks and
get *worse* with larger memory — the opposite of reservoir intuition — because the
symbol-averaged channel contracts distances by ~0.34/step (a ~1-step horizon),
while a sticky process with second eigenvalue |λ₂| = 0.8 needs memory over
~(1−0.8)⁻¹ = 5 steps. Match the channel's contraction time to the process
correlation time.

## Field encoding: the polarized-fixed-point requirement

Hamiltonian-parameter (field) encoding without re-injection needs a *polarized*
dissipative fixed point:
- **Dephasing fails:** the driven system relaxes to the maximally mixed state,
  ⟨Z_i⟩ → 0 for all sites — input-independent, signal erased. Features become
  identically ~0 within one interval; NMSE pinned at ~1.0.
- **Amplitude damping works:** fixed point |0…0⟩ (⟨Z_i⟩ = +1) is polarized, so the
  input-modulated Hamiltonian imprints recoverable structure on the transient
  relaxation. With amplitude damping + a *transverse* (x-axis) drive, NARMA-2
  reaches NMSE ≈ 0.19–0.22 at N=4–5. Longitudinal (z-axis) drive commutes with the
  read-out basis and carries far less memory.

## The signal-to-noise ceiling check (before any shot-noise study)

A shot-noise / mitigation study can silently measure *signal absence* instead of
the phenomenon of interest. Symptom: the naive noisy-feature NMSE is pinned near a
value that does not improve with more shots, and no ridge penalty helps. **Always
measure the noiseless (analytic) ceiling first.** If the analytic error is already
near the noisy error, the reservoir's per-feature signal is too weak and the study
is measuring nothing — pick a stronger-signal reservoir (e.g. a light-damping,
`sqrt`-encoding configuration with analytic NMSE ~0.08) before studying shots.
Shot noise follows σ ∝ S^{−1/2}: every halving of error costs ~4× the shots.

## Report effective rank, not just feature count

Under a scalar drive, most features are strongly correlated: a nominal 16-feature
reservoir often exposes only ~1–3 *effectively independent* directions
(participation ratio of the feature-covariance spectrum). This reframes "size
scaling" — bigger reservoirs are largely adding correlated features, not usable
dimensions — and is a direct, cheap probe of the exponential-concentration problem.

## Two process/meta lessons

- **Compute conclusions in-script; never pre-write them.** A hand-written "reading
  guide" once asserted a reservoir was "feature-poor" when its own figure showed
  the opposite. Derive the summary (best/worst, most-fragile, most-costly) from the
  run's results dictionary.
- **Ship what you tested.** Run the full test suite from the *packaged* copy, not
  the working tree — a partial copy failure can silently nest or truncate the tree.
- **Shot-noise model used consistently across reservoirs for fair comparison:**
  z = ⟨Z⟩ ∈ [−1,1], p = (1+z)/2, binomial std error σ(z) = 2√(p(1−p)/S); emulate as
  clip(z + ξ·σ(z), −1, 1), ξ ~ N(0,1), seeded, per-feature per-step.
