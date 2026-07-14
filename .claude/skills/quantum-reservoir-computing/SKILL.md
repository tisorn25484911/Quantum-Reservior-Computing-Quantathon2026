---
name: quantum-reservoir-computing
description: Expert knowledge for quantum reservoir computing (QRC) — designing, implementing, and benchmarking quantum reservoirs — and for searching the QRC research literature. Use this whenever the task touches QRC in any form, such as encoding data into a reservoir (state/amplitude vs Hamiltonian-parameter vs drive vs channel), designing the reservoir Hamiltonian (spin models, edge of chaos, disorder tuning), picking an implementation modality (Trotterized/digital vs analog vs digital-analog), choosing a platform, designing the readout, or benchmarking against classical baselines. Also use it when searching or synthesizing QRC literature — it captures the key venues, the anchor papers, how to query the field's sub-axes, and why cross-paper numbers are incomparable. Trigger even when the user only says reservoir computing, QRC, quantum machine learning on temporal data, echo state, or names QRC concepts like NARMA, memory capacity, virtual nodes, or edge of chaos without explicitly asking for a skill.
---

# Quantum Reservoir Computing: design, implementation, and literature research

This skill captures a working understanding of quantum reservoir computing (QRC):
the design surface, the mechanistic findings that decide whether a reservoir
actually works, the standards for honest benchmarking, and — separately — how to
search the QRC literature efficiently. It is written to let you (or a future
instance) reason about a QRC task from first principles rather than pattern-match,
and to avoid the specific mistakes that silently produce a reservoir that looks
plausible but learns nothing.

Two reference files hold the deep detail; read them when the task calls for it:
- `references/key_papers.md` — an annotated catalog of anchor papers with arXiv
  IDs, grouped by topic. Read it when you need a citation, want the
  state-of-the-art on a sub-topic, or are building a bibliography.
- `references/implementation_gotchas.md` — the hard-won, non-obvious lessons from
  actually building and debugging a QRC codebase (API pitfalls, the constructions
  that fail trace-preservation, the fixed-point requirement for field encoding,
  the chaos-dial that doesn't work at small sizes, and more). Read it before
  writing or debugging any QRC simulation code.

## The three-layer anatomy — and why only two layers are "designed"

A quantum reservoir is a quantum system (a network of interacting qubits or
bosonic modes) driven into non-equilibrium dynamics by an input sequence. It
factorizes into three layers:

1. **Input / encoding layer** — how each input is written into the system.
2. **Reservoir layer** — the *fixed*, untrained dynamics (the Hamiltonian, the
   dissipation, the time-scale) that map the input history into a
   high-dimensional feature space of measured observables.
3. **Read-out layer** — a *linear* map (ridge regression), the only trained part.

The one-linear-solve training is the whole point: it sidesteps gradients and
barren plateaus entirely. Consequently **all design effort lives in layers 1 and
2**, plus the measurement scheme that feeds the read-out. When you help with a QRC
task, you are almost always choosing an encoding, choosing/tuning a Hamiltonian,
choosing an implementation modality, or choosing a read-out — the four axes below.

## Read this first: the honesty discipline

QRC is a field where it is very easy to over-claim, and much of the published
literature quietly stacks the deck. At the system sizes anyone can simulate
exactly (roughly N ≤ 10–12 qubits) **there is no quantum speed-up to
demonstrate.** Enforce these rules in any QRC work you touch:

- **Always compare against a size-matched classical baseline.** An echo-state
  network (ESN) whose node count equals the quantum feature count, so both
  read-outs train the same number of weights. Quantum reservoirs frequently lose
  to a good ESN on linear-memory tasks — report that.
- **Always include a structureless quantum control.** A Haar-random reservoir
  (identical injection and read-out, but featureless random unitaries instead of
  structured Hamiltonian evolution) fixes the floor. Any claimed "edge of chaos"
  or "quantumness" benefit must beat this control, or the structure is doing no
  work.
- **Frame claims as parity and parameter efficiency, never speed-up.** The
  defensible statements are "matches a size-matched ESN with fewer trained
  parameters" or "reaches comparable accuracy" — not "outperforms classical".
- **Separate results that embed prior knowledge.** If a reservoir is handed the
  generating process (e.g. a QHMM whose channel *is* the target process), report
  it as a validation anchor / reference line, clearly starred, never as a fair
  untrained competitor.
- **Put a "worse than the mean predictor" line on every NMSE axis.** With
  variance-normalized NMSE, a value ≥ 1 means the model is worse than predicting
  the mean. Many broken reservoirs sit at exactly 1.0; making that line visible
  turns a silent failure into an obvious one.

## Axis I — How the dataset enters the reservoir (encoding)

This is the single most consequential choice. There is a clean dichotomy plus
several sub-families. Full formulas are in `references/key_papers.md`; the
decision logic:

- **State / amplitude encoding (Fujii–Nakajima, 2017).** Write each scalar input
  into the *state* of one qubit, then inject by *partial trace + re-prepare*
  (`ρ → U(ρ_in(u) ⊗ Tr_1[ρ])U†`). Uses entanglement explicitly; costs
  high-fidelity state preparation. The partial-trace injection is realized on
  hardware as **mid-circuit measurement + reset** — that reset is also what gives
  memory unbounded by the qubit coherence time (NISQRC).
- **Hamiltonian-parameter encoding (McCaul 2025; Settino 2025).** The input
  *modulates a parameter of the Hamiltonian* (e.g. a transverse field `h_k`).
  **No state prep, no feedback, no tomography** — the chief practical attraction.
  A closed unitary driven this way has no observable fading memory, so either add
  weak dissipation (to restore the echo-state property) or augment the read-out
  with classical delay embeddings. Expressivity has a clean form: the output is a
  truncated Fourier series whose frequencies are set by the encoding.
  **Critical, non-obvious requirement:** without re-injection to re-pump
  polarization, this encoding needs a *polarized* dissipative fixed point.
  Amplitude damping (fixed point |0…0⟩) works; pure dephasing (fixed point
  maximally mixed, ⟨Z⟩→0) erases the signal completely. See
  `references/implementation_gotchas.md`.
- **Drive-amplitude encoding (bosonic / circuit-QED).** Input rides on the
  amplitude of a coherent drive; measure the cavity in the Fock basis. Natural for
  superconducting and photonic hardware.
- **Continuous-variable encoding.** Input across Gaussian/squeezed optical modes.
- **Channel encoding (QHMM route).** The input (a symbol, or a continuously
  parametrized ancilla angle) selects an input-driven CPTP map on a small memory;
  contraction of the channel supplies the echo-state property. A *process-matched*
  channel can realize a given hidden Markov model exactly and recover its Bayesian
  filter — the maximally parameter-efficient reference point.
- **Rydberg's three parameter-encoding knobs (Kornjača 2024, 108 qubits).** Global
  detuning (capacity *independent* of atom number — robust but capacity-limited),
  local detunings, and atomic positions (both scale expressivity with size, but
  are more prone to the concentration problem).

**Guidance.** State encoding and parameter encoding are the two poles; build a
controlled comparison by holding the dynamics and read-out fixed and varying only
the encoding. Parameter encoding is usually the better bet for near-term hardware
(no arbitrary state prep), provided you get the dissipative fixed point right.

## Axis II — What to develop in the Hamiltonian layer

**The model zoo** (details and citations in `references/key_papers.md`):
transverse-/mixed-field Ising (the workhorse), Sachdev–Ye–Kitaev (SYK, the
edge-of-chaos anchor), Bose–Hubbard lattices, Rydberg arrays (analog flagship),
Jaynes–Cummings / dispersive-JC qubit-boson systems, and dual-unitary circuits.

**The headline design principle — the edge of many-body quantum chaos, which has
TWO edges.** The quantum statement is spectral (no phase-space trajectory): the
level-spacing ratio ⟨r⟩ interpolates between Poisson (≈ 0.386, integrable /
localized) and Wigner–Dyson/GOE (≈ 0.531, chaotic).
- **Parametric edge** — the integrable-to-chaotic transition. Task performance is
  typically U-shaped in the tuning parameter, peaking in the ergodic-but-not-yet-
  localized window.
- **Temporal edge** — set by the Thouless time. Sweep the input interval Δt_in at
  fixed chaotic coupling: short intervals under-mix (weak nonlinear processing),
  long intervals scramble past the Thouless time (memory capacity collapses toward
  zero), and the optimum sits in between.

Reproduce *both* edges — they are the concrete, falsifiable form of the principle.

**Other design principles worth applying:**
- **Ergodicity constrained by symmetry.** Ergodic dynamics gives good state
  separation, but *unresolved symmetry sectors corrupt the level statistics* and,
  at small sizes, are the reason a "clean" chaos dial is hard to find. Break the
  symmetries (disorder does this for free) or block-diagonalize by sector.
- **Disorder is *not* universally required.** In homogeneous Bose–Hubbard
  lattices the optimal regime can be chaotic *or* weakly-interacting depending on
  the task. The all-to-all-random recipe is one good design, not the only one.
- **Fading memory has a quantum order parameter:** the spectral radius of the
  Pauli transfer matrix, which tracks the MBL/spin-glass transition — the quantum
  analogue of the ESN spectral radius.
- **Where memory is sourced:** weak Lindblad dissipation (exponential fading set
  by the damping rate), deterministic reset (coherence-time-unlimited), or
  non-Markovian engineering (can exceed the Markovian memory bound). Add explicit
  feedback when the task horizon exceeds the natural fading time.

## Axis III — How many ways to realize the dynamics (modalities)

Three modalities; the digital/analog choice is the live debate.
- **Digital (Trotterized).** Decompose `e^{−iHt}` into native gates (IsingZZ +
  single-qubit rotations). The default for circuit-model experiments; costs Trotter
  error and depth, and discretizes a continuous-time input into one gate per step.
- **Analog.** Evolve under a tunable Hamiltonian with no gate decomposition — the
  dynamics *is* the computation. Avoids Trotter error; natural for continuous-time
  signals. The modality of the large-scale Rydberg and oscillator-qubit demos.
- **Digital-analog (DAQC).** Fixed analog entangling blocks interleaved with
  single-qubit gates. More robust to control error than two-qubit gates; ~10×
  depth reduction for structured Hamiltonians. A natural fit for QRC — the fixed
  reservoir Hamiltonian is the analog block, input/basis changes the digital layer.

**Platform map.** Superconducting circuit-QED (digital or oscillator-analog);
neutral-atom / Rydberg (analog flagship); trapped ions (digital, strong
mid-circuit measurement); NMR (ensemble analog, FID signal is a natural temporal
multiplexer); photonic / continuous-variable (Gaussian/squeezed states);
Jaynes–Cummings cavities (bosonic analog).

## The read-out layer co-determines performance

- **Temporal multiplexing (virtual nodes):** sample one observable at several
  times per interval — equivalent to measuring Heisenberg-evolved operators
  {O(t_i)}. Nearly free extra features; almost always worth it.
- **Spatial multiplexing:** an ensemble of reservoir copies (NMR-native).
- **Measurement backaction** (projective read-out disturbs the propagated state)
  is handled by: the ensemble picture (neglect backaction — valid for NMR),
  rewinding/restart, weak/non-demolition measurement, or deterministic reset.
- **Observables:** local ⟨Z_i⟩ and two-point correlations for qubits; Fock-basis
  populations and higher quadratures for bosonic systems. Polynomial read-outs
  extend achievable decision boundaries for hard classification.
- **Diagnostics to report:** memory capacity (linear memory), effective rank
  (participation ratio of the feature covariance — usable feature directions,
  which is often *far* below the nominal count under a scalar drive), and channel
  contractivity / PTM spectral radius (fading memory).

## Open problems (and honest caveats)

- **Exponential concentration** — as the reservoir grows, measured features
  concentrate and predictions become input-agnostic. The quantum-specific failure
  mode with no classical analogue, and the central scaling threat. Probe it with
  effective rank.
- **Coherence-time-unlimited memory** (deterministic reset) and **non-Markovian
  memory enhancement** are active fronts worth benchmarking against a dissipative
  baseline.
- **Fair benchmarking is genuinely hard:** different papers use different
  observables and sampling rates even for the same Mackey–Glass series, so
  cross-paper numbers are essentially incomparable — never quote them as if on a
  common scale.

## Searching the QRC literature — the research methodology

QRC moves fast (major results in 2024–2026) and much of it is on arXiv before or
instead of journals. Search deliberately, not with one broad query.

**Where to look (in rough priority):**
- **arXiv** (quant-ph, cond-mat) — the primary and most current source; most key
  results appear here first.
- **APS journals** — Physical Review Letters, Physical Review Applied, PRX
  Quantum, Physical Review Research. PRApplied and PRX Quantum are the field's
  center of gravity for QRC specifically.
- **Nature Communications, npj Quantum Information, Communications Physics** —
  high-impact experimental and theory results.
- Prefer original papers over reviews for specifics; prefer the most recent
  version of an arXiv preprint.

**Search by sub-axis, not with one blanket query.** The field decomposes exactly
as this skill does, so issue *separate* searches per axis and synthesize:
- encoding: `quantum reservoir computing input encoding state versus Hamiltonian parameter`
- Hamiltonian design: `quantum reservoir computing edge of chaos disorder coupling topology`
- modality: `quantum reservoir computing analog versus digital Trotterization`
- platform: `quantum reservoir computing Rydberg / trapped ion / photonic / NMR`
- readout: `quantum reservoir computing temporal spatial multiplexing measurement backaction`
- theory: `quantum reservoir computing fading memory echo state property expressivity`

A single combined query returns shallow results for all of these; per-axis queries
go deep. Use short queries (3–6 content words), start broad then narrow, and add
the current year only when you want the newest work (a stale year term suppresses
recent results).

**Anchor papers to orient any search** (full list in `references/key_papers.md`):
Fujii & Nakajima 2017 (state encoding, the founding proposal); Kobayashi & Motome
2026, arXiv:2506.17547 (edge of many-body chaos, two edges); Kornjača et al. 2024,
arXiv:2407.02553 (108-qubit analog Rydberg, the scaling data point); McCaul et al.
2025, arXiv:2505.22575 (minimal Hamiltonian-encoding reservoir); Hu et al. 2024,
Nat. Commun. (NISQRC, coherence-time barrier); Bravo et al. 2022, PRX Quantum
(Rydberg reservoir theory). To find the state-of-the-art, search recent years and
follow forward citations from these anchors.

**Verification discipline.** When you pull an arXiv preprint into a bibliography,
verify the author list and venue against the actual arXiv page — snippets often
truncate authors, and recent preprints may have placeholder fields. Do not invent
attributions; if unsure of an author list, say so.

## Working on QRC code

Before writing or debugging any QRC simulation, read
`references/implementation_gotchas.md`. It records the library API pitfalls
(QuTiP 5, PennyLane mid-circuit reset), the injection-bookkeeping validation
anchor, the constructions that silently fail (non-trace-preserving Kraus sets, the
chaos dial that doesn't work at small sizes, the gate-reservoir memory mechanisms
that erase the signal), the field-encoding fixed-point requirement, and the
signal-to-noise ceiling check that must precede any shot-noise study. These are
mistakes that produce plausible-looking reservoirs that learn nothing, and each
one cost real debugging time to find.
