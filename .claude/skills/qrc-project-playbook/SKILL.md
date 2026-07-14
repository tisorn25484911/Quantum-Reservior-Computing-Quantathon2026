---
name: qrc-project-playbook
description: End-to-end workflow playbook for quantum reservoir computing (QRC) applied projects, especially renewable-energy and climate forecasting. Use when planning a QRC project or startup idea, searching/synthesising QRC literature, finding and vetting time-series datasets (solar, wind, load, grid, battery, weather, ENSO/climate), turning an application idea into a concrete pipeline, critically reviewing a QRC idea/paper/benchmark for overclaiming, or scaffolding a QRC codebase in NumPy/Cirq/Qiskit/PennyLane. Trigger on phrases like "QRC project", "quantum reservoir startup", "find a dataset for", "review this QRC idea/result", "is this quantum advantage claim sound", "set up a QRC repo", or any QRC-for-applications planning. Complements the quantum-reservoir-computing skill (domain physics); this one is process and judgement.
---

# QRC Project Playbook

Process skill for taking QRC application work from idea to defensible result.
Load `quantum-reservoir-computing` (domain physics: encodings, Hamiltonians,
edge of chaos, benchmarking axes) alongside this for any technical design.
Deep worked reference: the review + handbook produced 2026-07 (user's
`qrc_review` project: 8-part LaTeX doc + verified code in
`code/qrc_core.py`, `cirq_qrc_minimal.py`, `qiskit_qrc_hw.py`,
`pennylane_qrc_hybrid.py`).

## 1. Literature-search playbook

Search per AXIS, not per topic — QRC papers self-describe inconsistently.

| Axis | Query stems (combine 2) | What to extract |
|---|---|---|
| Encoding | "quantum reservoir" + input encoding / Hamiltonian parameter / re-uploading / detuning | encoding knob, input dimensionality handled |
| Dynamics | + edge of chaos / dynamical phase / localization / scrambling / Thouless | operating-point recipe, chaos diagnostic used |
| Memory/ESP | + echo state property / fading memory / coherence influx / convergence | contraction mechanism named? |
| Measurement | + weak measurement / mid-circuit / feedback / restart / back-action | protocol; shots per feature |
| Noise/shots | + shot noise / sampling / exponential concentration / eigentask | ceiling-vs-sampled separation? |
| Platform | + Rydberg / superconducting / photonic / trapped ion / NMR | H(hardware) vs S(simulation) label |
| Application | + forecasting / solar / wind / load / climate / chaotic | baselines used (see rubric §4) |

Venues that carry the field: arXiv quant-ph (primary; IDs are YYMM.NNNNN —
June 2026 = 2606.*), PRX Quantum, PR Applied, PR Research, PRL/PRE, npj QI,
Quantum, Nat. Commun., Commun. Phys., Mach. Learn.: Sci. Technol., Chaos.
Applications side: Applied Energy, Int. J. Forecasting, Solar Energy.

Use `/arxiv`, `/semantic-scholar`, `/alphaxiv`, `/deepxiv` skills for
retrieval; `/citation-audit` before anything ships. Hard rules learned the
expensive way:
- NEVER cite a paper whose title+authors+venue were not verified against a
  live page this session. Training-memory bibliography is ~90% right, which
  means several wrong entries per document.
- One search verifying a target paper usually verifies 5–10 more from its
  reference list — harvest them.
- If unverifiable after one focused search: drop it and cite a verified
  neighbour, or state the claim as this-document's own demonstration.
- Anchor set (verified 2026-07, safe to reuse): Fujii & Nakajima PRApplied 8
  024030 (2017); Nakajima+ PRApplied 11 034021 (2019); Chen & Nurdin QIP 18
  198 (2019); Chen+ PRApplied 14 024065 (2020); Martínez-Peña+ PRL 127
  100502 (2021) and PRE 107 035306 (2023); Mujal+ AQT 4 2100027 (2021), npj
  QI 9 16 (2023); Hu+ PRX 13 041020 (2023), Nat Commun 15 7491 (2024);
  Kobayashi+ PRX Quantum 5 040325 (2024); Kobayashi/Tran/Nakajima PRE 110
  024207 (2024), arXiv:2409.12693; Sannia+ Quantum 8 1291 (2024),
  arXiv:2505.10062; Xiong+ arXiv:2505.10080; Kobayashi & Motome PRL 136
  040602 (2026) [temporal+parametric edges]; Kornjača+ arXiv:2407.02553
  [108-qubit Rydberg]; Senanian+ Nat Commun 15 7490 (2024); Bravo+ PRX
  Quantum 3 030325 (2022); Settino+ PRApplied 24 024019 (2025); Ahmed+ PRR 6
  043082 (2024), QMI 7 31 (2025); Pfeffer+ PRR 4 033176 (2022), PRR 5 043242
  (2023). Full BibTeX: `references/verified_refs.md`.

## 2. Dataset discovery and vetting

Registry with access notes: `references/dataset_registry.md`. Shortlist by
domain: solar → NSRDB (30-min gridded) or SURFRAD (1-min ground truth);
wind → WIND Toolkit (modelled — say so); load → GEFCom2014, UCI
ElectricityLoadDiagrams, ENTSO-E/OPSD; storage → Severson 2019, NASA PCoE;
weather → Jena 10-min, ERA5; climate → NOAA Niño SST (statsmodels
`elnino` ships it), CMIP6.

Vetting checklist (all must pass before any model touches the data):
1. Provenance + licence verified from the provider page; version pinned.
2. Length ≥ a few thousand steps at the modelling resolution → supports
   chronological train/val/test.
3. Missing data located; policy = MASK, never silently interpolate; gaps >
   autocorrelation time split the series.
4. Target transform identified from physics: solar → clear-sky index k_t
   (never raw GHI), climate → anomaly vs training-years climatology, load →
   residual after calendar model. Baselines apply to the TRANSFORMED target.
5. Leakage audit: any statistic (scaling, climatology, clear-sky fit)
   computed on training span only.
6. A living classical benchmark literature exists (else "good" is
   undefined).
QRC-fit triage: inputs low-dimensional (≤ a few channels — reduce fields
classically first), difficulty from memory+nonlinearity not dimensionality,
T in the thousands (shot budgets). Fail → recommend classical or redesign.
Surrogate policy: building a statistical surrogate is legitimate for
self-contained/licence-clean work IF (a) constructed to a named archetype's
EDA template, (b) labelled "surrogate/synthetic" in every figure caption and
table where a number appears, (c) real-data pathway stated.

## 3. Idea → pipeline planner (incl. startup-style ideas)

Intake questions (ask, or infer and state assumptions): What variable, what
horizon, what decision does the forecast feed? What does an error cost?
What's the incumbent (there is always an incumbent — often persistence or a
calendar model)? Data access confirmed? Success criterion the user would
accept BEFORE seeing results?

Then instantiate the nine-stage pipeline (each stage gets one decision,
written down): 1 acquisition (source, licence, version) → 2 QC (mask policy)
→ 3 target transform → 4 encoding (scale to encoding's natural domain with
train stats; scalar replacement vs multi-channel rotations) → 5 reservoir
(architecture from quantum-reservoir-computing skill; SWEEP JΔt — never
default it; seed recorded) → 6 read-out (ridge; λ by chronological val block,
or GCV when the block is thin/unrepresentative) → 7 uncertainty (pinball
quantiles for energy work) → 8 baseline battery (mean=1.0 line, persistence,
linear lags, size-matched ESN at equal feature count, Haar-random quantum
control) → 9 interpretation (regime-conditional errors; claims scoped
theory/simulation/hardware and surrogate/real).

Posture defaults: HYBRID read-out (quantum features ⊕ cheap classical
features) with quantum-only and classical-only ablation columns mandatory —
the delta IS the quantum contribution. Spend no quantum resources on what a
lookup table solves (calendar, solar geometry). Pre-register the success
criterion: "beat {baseline} on {metric} on {split} by {margin}" in writing
before test data is touched. For startup framing add: shots-and-wallclock
cost per prediction next to accuracy (a model needing 1e9 shots to match a
laptop has answered the business question), and name which pipeline stage
the venture actually owns.

## 4. Idea / paper / result review rubric

Score each; any single ✗ on 1–4 invalidates headline claims.

| # | Check | Red flags |
|---|---|---|
| 1 | Baseline battery complete | no persistence on near-persistent series; no size-matched ESN; no mean line |
| 2 | Matched read-out capacity | quantum's 30 features vs classical's 5; unequal tuning effort |
| 3 | Chronological splits | shuffled CV on time series; climatology/scaler fit on full record |
| 4 | Target transform sound | raw GHI "accuracy"; raw SST; load without calendar ablation |
| 5 | Structure controls | no Haar control (tuned-dynamics vs "quantum"); no simulable control (Gaussian/Clifford) when advantage implied |
| 6 | Shots/noise honesty | exact-sim results implying hardware; S unreported; no analytic ceiling |
| 7 | Operating point swept | single arbitrary JΔt / γ; "we set Δt=1" with no sweep |
| 8 | Scaling, not points | one configuration crowned; no curves vs n, S, T |
| 9 | Concentration awareness | n>10 claims from n=4 trends; deep scrambling + local obs |
| 10 | Claim scoping | "advantage" absent cost axis; surrogate data implied real; sim implied HW |

Verdict template: strongest element / fatal or fixable flaws (numbered
against rubric) / what result WOULD support the claim / salvage path.
Calibration from own experiments: solar 1-step is a 4-way statistical tie
with persistence (a "QRC-only" table there looks publishable and is
meaningless); recurrent-vs-windowed on ENSO ≈ 0.60 vs 0.75 NMSE (recurrence
is worth more than any tuning); noisy hybrid can score BELOW its classical
ablation (small T + noisy columns) — a correct pipeline reports that.

## 5. Codebase scaffolding

Layout: `code/` (core, datasets, eda, experiments, per-framework files),
`figures/` (regenerated by scripts only), doc dir. One config block per
script (n, V, J, h, Δt, seeds) printed at runtime.

Build ORDER (non-negotiable): (1) validation anchors FIRST —
inject/rebuild identity, propagator unitarity, ρ Hermitian/trace-1/PSD,
ridge recovers a known linear map to 1e-10, NMSE(mean)=1.000,
shot-noise σ ∝ S^(-1/2); (2) exact NumPy reference reservoir (defines the
ceiling); (3) baselines + eval harness (chronological splits, NMSE, battery);
(4) EDA before any modelling; (5) experiments; (6) framework ports validated
AGAINST the reference at identical seeds; (7) figures from scripts, stats
printed not hardcoded.

Framework choice: architecture research → NumPy/Cirq exact expectations;
hardware-bound → Qiskit from day one (transpile to CouplingMap + basis
ecr/rz/sx/x, NoiseModel, batched jobs, dynamic circuits); hybrid/autodiff →
PennyLane (broadcasting; tune encoding gain by ALTERNATING ridge-refit ↔
gradient step with read-out frozen — don't differentiate through the solve).

Gotchas that cost hours (all hit in practice):
- Qubit order: fix input qubit as leftmost tensor factor so np.kron
  re-insertion needs no permutation; write the rebuild anchor to catch it.
- Qiskit counts keys are LITTLE-ENDIAN → reverse key once, one place.
- Mid-circuit demos need per-step ClassicalRegisters (one register =
  overwritten history).
- PennyLane qml.measure(reset=True) allocates an extra wire on
  default.mixed → use compiled-unitary/NumPy for density-matrix studies.
- Thin chronological val tails mis-select λ on regime-switching series →
  GCV via SVD.
- Listings/LaTeX: em-dashes in .py break lstinputlisting under utf8 →
  ASCII-only code comments.
- Encoding scale: sweep γ ∈ {π/4, π/2, π}; π/2 vs π was 2.5× NMSE.
- Always sweep JΔt log-spaced over ≥2 decades; optimum was 4× better than
  plausible-looking defaults.

Definition of done: anchors pass; every table number printed by a script;
surrogate labels everywhere applicable; baseline battery in every results
table; claims scoped; a stranger can rebuild PDF + figures with two
commands.
