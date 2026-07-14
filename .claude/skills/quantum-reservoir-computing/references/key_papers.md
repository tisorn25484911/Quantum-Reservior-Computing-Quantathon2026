# Key QRC papers — an annotated catalog

Grouped by topic. arXiv IDs are given where known; verify author lists and venues
against the arXiv page before citing (snippets truncate authors; recent preprints
may carry placeholder fields). Years reflect publication/preprint dates known as
of the last update and should be re-checked for the current state of the art.

## Founding proposals and encoding

- **Fujii & Nakajima, "Harnessing Disordered-Ensemble Quantum Dynamics for
  Machine Learning," Phys. Rev. Applied 8, 024030 (2017).** The founding QRC
  proposal. State/amplitude encoding: write input into one qubit as
  `ρ₁(u) = √((1+u)/2)|0⟩ + √((1−u)/2)|1⟩`, inject by partial trace + re-prepare,
  read Heisenberg-evolved observables via temporal multiplexing (virtual nodes).
- **McCaul et al., "Minimal quantum reservoirs with Hamiltonian encoding," Chaos
  35, 093135 (2025), arXiv:2505.22575.** Encode the input into *Hamiltonian
  parameters* rather than a state — no state prep, feedback, or tomography.
  Closed-unitary version has no fading memory; uses delay embeddings.
- **Settino et al., "Memory-augmented hybrid quantum reservoir computing," Phys.
  Rev. Applied 24, 024019 (2025), arXiv:2409.09886.** Input-dependent-field
  Hamiltonian encoding with retained memory:
  `H_k = Σ J_ij σ_i^x σ_j^x + h Σ σ_i^z + h_k Σ σ_i^x`.
- **"Expressivity of quantum reservoir computers," arXiv:2501.15528 (2025).**
  Under Hamiltonian encoding the output is a truncated Fourier series; accessible
  frequencies are fixed by the encoding, coefficients by the reservoir/measurement.

## Edge of chaos and Hamiltonian design

- **Kobayashi & Motome, "Edge of Many-Body Quantum Chaos in Quantum Reservoir
  Computing," Phys. Rev. Lett. 136, 040602 (2026), arXiv:2506.17547.** THE
  edge-of-chaos anchor. Mixes SYK₄ (chaotic) and SYK₂ (integrable), each rescaled
  by its spectral norm. Identifies TWO edges: temporal (Thouless time) and
  parametric (integrable→chaotic). Establishes the edge as a design guideline.
- **Martínez-Peña et al., "Dynamical Phase Transitions in Quantum Reservoir
  Computing," Phys. Rev. Lett. 127, 100502 (2021).** Links reservoir performance
  to the ergodic/MBL phase of a transverse-field Ising model with disorder.
- **"Quantum reservoir computing in atomic lattices," Chaos, Solitons & Fractals
  (2025), ScienceDirect S0960077925003029.** Counter-evidence to the
  disorder-is-necessary folklore: homogeneous Bose–Hubbard couplings suffice, and
  the optimal regime (chaotic vs weak-interaction) is task-dependent.
- **Dual-unitary circuits** as minimal models of many-body chaos — entangling
  power as the tuning dial (see arXiv listings on dual-unitary reservoir dynamics).

## Fading memory, non-Markovianity, feedback

- **"Connection between coherence influx and QRC fading memory," arXiv:2409.12693
  (2024).** The Pauli-transfer-matrix spectral radius characterizes fading memory
  and tracks the dynamical phase transition; coherent-environment interaction is
  needed for a non-stationary echo-state property.
- **Sannia et al., "Non-Markovianity and memory enhancement in quantum reservoir
  computing" (2025).** A structured (non-Markovian) environment can push memory
  capacity beyond the Markovian bound.
- **Kobayashi, Fujii & Yamamoto, "Feedback-Driven Quantum Reservoir Computing for
  Time-Series Analysis," PRX Quantum 5, 040325 (2024).** Explicit feedback extends
  the effective memory beyond the reservoir's natural fading time.
- **Götting et al., "Connection between Memory Performance and Optical Absorption
  in Quantum Reservoir Computing," Phys. Rev. Lett. 135, 240403 (2025).** Ties
  memory performance to a measurable physical quantity in photonic reservoirs.

## Platforms and large-scale / experimental demonstrations

- **Kornjača et al., "Large-scale quantum reservoir learning with an analog
  quantum computer," arXiv:2407.02553 (2024).** 108-qubit neutral-atom (QuEra)
  analog QRC. Three Rydberg encodings — global detuning (capacity independent of
  atom number), local detunings, atomic positions. The key scaling data point.
- **Bravo et al., "Quantum Reservoir Computing Using Arrays of Rydberg Atoms,"
  PRX Quantum 3, 030325 (2022).** Foundational Rydberg-reservoir theory.
  `H = Σ Ω σ_x − Σ Δ n + Σ V_jk n_j n_k`, `V_jk = C₆/‖r_j−r_k‖⁶`.
- **Senanian et al., "Microwave signal processing using an analog quantum
  reservoir computer," Nat. Commun. 15, 7490 (2024).** Oscillator-qubit
  superconducting *analog* reservoir; continuous-time input, no Trotter
  discretization.
- **Carles et al., "Experimental quantum reservoir computing with a circuit
  quantum electrodynamics system," Phys. Rev. Applied (2026),
  arXiv:2506.22016.** Drive-amplitude encoding; single transmon + resonator,
  Fock-basis read-out.
- **Nakajima et al., "Boosting Computational Power through Spatial Multiplexing in
  Quantum Reservoir Computing," Phys. Rev. Applied 11, 034021 (2019).** The
  spatial-multiplexing (ensemble) technique; NMR-native.
- **Nokkala et al., "Gaussian states of continuous-variable quantum systems
  provide universal and versatile reservoir computing," Commun. Phys. 4, 53
  (2021).** Universality for continuous-variable optical reservoirs.
- **Paparelle et al., "Experimental memory control in continuous variable optical
  quantum reservoir computing," arXiv:2506.07279 (2025).**
- **"Quantum reservoir computing in Jaynes–Cummings models," Phys. Rev. Research
  (2026).** Bosonic qubit-cavity reservoir; nonlinear memory and forecasting.

## Read-out, measurement, coherence-time barrier

- **Hu et al., "Overcoming the coherence time barrier in quantum machine learning
  on temporal data," Nat. Commun. 15, 7491 (2024).** NISQRC: mid-circuit
  measurement + deterministic reset processes signals of arbitrary duration,
  unbounded by qubit coherence time; a Volterra-series theory of the read-out.
- **Mujal et al., "Time-series quantum reservoir computing with weak and
  projective measurements," npj Quantum Inf. 9, 16 (2023).** How weak vs
  projective read-out and backaction shape performance; rewinding protocols.

## Digital-analog computation

- **Parra-Rodriguez et al., "Digital-Analog Quantum Computation," Phys. Rev. A
  101, 022305 (2020).** The DAQC paradigm: analog entangling blocks + single-qubit
  gates; robustness and depth advantages over pure-digital for structured
  Hamiltonians.

## How to extend this catalog

To find newer work: search arXiv (quant-ph, cond-mat) and the APS/Nature venues
per-axis (see the search methodology in SKILL.md), and follow forward citations
from the anchors above — especially Kobayashi & Motome 2506.17547, Kornjača
2407.02553, and Hu 2024. Re-verify any arXiv ID and author list against the live
page before adding it to a bibliography.
