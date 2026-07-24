# plan.md — Improve QRC accuracy per Fujii & Nakajima (2017)

**Paper:** K. Fujii & K. Nakajima, *"Harnessing disordered ensemble quantum
dynamics for machine learning,"* Phys. Rev. Applied **8**, 024030 (2017),
arXiv:1602.08159. This is the founding QRC paper the repo's reservoir
(`Hamiltonian_QRC/qrc_core.py`) already implements.

**Goal:** apply the paper's accuracy levers to the existing pipeline, honestly —
every accuracy number still ships with persistence + a size-matched ESN, over
multiple seeds (project invariant). No quantum-advantage claim at ≤8 qubits.

---

## 0. Direct answer to the question asked

> *Is using the V subintervals to receive NV virtual nodes a good idea?*

**Yes — it is the paper's single most important accuracy lever, and it is not
optional for nonlinearity.** The paper's Fig. 5(b) shows the parity-check
(nonlinear) capacity is **exactly zero at V=1**: with no virtual nodes the
reservoir has *no* nonlinear computational capacity at all. Virtual nodes
"spatialize the real-time dynamics during the interval τ" and are what let the
linear read-out see nonlinear functions of the input. Memory (STM) capacity
saturates around **V≈10**; nonlinear (PC) capacity keeps rising with V.

The mechanism is **already implemented** in `qrc_core.py` (`U_sub =
exp(-iH·dt/V)`, ⟨Z_i⟩ read at each of V subintervals → NV nodes). What is
sub-optimal is the **value**: the default is `virtual_nodes=4`, below the
paper's V≈10 saturation and well below its powerful configs (V=10–50). So the
improvement is to raise V and prove the effect, not to add the mechanism.

## 1. Gap analysis — paper vs. current code

| Lever | Paper's guidance | Current code | Action |
|---|---|---|---|
| Virtual nodes **V** | ≈10 saturates STM; more for PC; powerful configs 10–50 | `virtual_nodes=4` | **Raise default to 10**; validate PC(V=1)≈0 |
| Reservoir **integrability** | disordered fully-connected (nonintegrable) → high STM+PC; 1D-NN chain integrable → poor PC | `ising` (nonintegrable) ✓, `xxz_hx` (1D-NN XXZ = integrable) | Keep `ising` as the accuracy default; **document** why `xxz_hx` is weaker |
| **τΔ** (coupling×interval) | non-monotonic; powerful config τΔ=2 | `dt=2.0, J=1.0` ⇒ τΔ=2 ✓ | Keep; expose a small τ sweep |
| **h/Δ** (transverse field) | 0.5 | `hx=1.0` ⇒ h/Δ=1.0 | **Lower default hx to 0.5** |
| Qubits **N** | 5–7 (7 ≈ ESN-500) | 5 | Keep 5 default; make 6–7 easy (2ᴺ×2ᴺ ρ is cheap to N=8) |
| **Closed-loop training** (Mackey-Glass = free-run) | teacher-force + **add small noise [−σ,σ] to reservoir states in training**; perturb on loop close | none | **Implement training-noise regularization**; wire into the free-run + rollout paths |
| Observables | ⟨Z_i⟩ only (NV nodes) | ⟨Z_i⟩×V **plus** ⟨Z_iZ_j⟩ | Keep the extra ZZ (an enhancement); no change |

## 2. Changes to implement (ordered by value)

### 2.1 Training-noise regularization for closed-loop robustness  *(highest value)*
The free-run mode we built flat-lines because a ridge read-out fed back through a
damped reservoir contracts to a fixed point. The paper's fix for exactly this
(its Mackey-Glass closed-loop task): add a slight noise to the reservoir states
during teacher-forced training so the read-out learns to correct back toward the
trajectory instead of overfitting exact clean states.

- `Anomaly_Forecast/forecast.py::fit_readout`: add `train_noise: float = 0.0`
  and an optional `rng`. When `>0`, fit the ridge weights on **noise-perturbed**
  training features (Gaussian, zero-mean, σ=`train_noise`), but compute the
  reported residuals on the **clean** features (the residual pool feeds the
  Step-3 bootstrap and must stay honest).
- Wire it through: `webapp/app/domain/freerun.py` (default on for the closed-loop
  free-run, e.g. σ≈1e-2 of the unit-scaled feature range) and expose it as a
  form field; also make it available to `anomaly.run_rollout_study`.
- **Verify:** free-run NMSE on `got_sst` (train 30%) with vs. without training
  noise, multiple seeds, against the seasonal-average + persistence + ESN
  baselines. Report honestly whether it helps; keep it off (σ=0) if it does not.

### 2.2 Raise virtual nodes to the paper's regime
- Bump the accuracy-path defaults `virtual_nodes=4 → 10` where it does not blow
  up cost (forecast/free-run/anomaly). Keep it a knob.
- **Validate the paper's central claim** with a new benchmark
  `Hamiltonian_QRC/capacity_vs_V.py`: STM capacity and **parity-check (PC)
  capacity** vs V ∈ {1,2,5,10,25}, 5-qubit `ising`, multiple random couplings.
  Must reproduce **PC(V=1) ≈ 0** and PC rising with V. This is the evidence that
  answers §0 with numbers, not assertion.

### 2.3 Paper-optimal field ratio
- Lower the default transverse field `hx=1.0 → 0.5` (h/Δ=0.5) in the accuracy
  paths, and confirm on a benchmark task (NMSE and/or capacity) that it is at
  least parity — do not regress. Keep τΔ=2 (already optimal).

### 2.4 Document the integrability caveat
- Note in `qrc_core.py` / `forecast.py` and the free-run UI that `xxz_hx` (1D-NN,
  integrable) has structurally poor nonlinear capacity per the paper, so `ising`
  is the accuracy default — this is *why* `ising` won on `got_sst`.

## 3. Verification & honesty guardrails

- Every before/after accuracy claim uses the same held-out split, with
  **persistence + size-matched ESN** on the same axis, over **≥3 seeds** (ising
  couplings and ESN weights are random; report mean±spread).
- Reproduce the paper's qualitative result (PC(V=1)≈0, PC↑ with V) as the
  validation anchor for the virtual-node change.
- If a change does not help on our data, **say so and leave the default alone.**
  The point is an honest, paper-grounded improvement, not a flattering number.
- No quantum-advantage claim: ceiling is parity with the size-matched ESN.

## 4. Acceptance criteria

1. `fit_readout(train_noise=σ)` works; free-run can be run with/without it and the
   effect is measured on `got_sst` (≥3 seeds, all baselines shown).
2. `capacity_vs_V.py` reproduces PC(V=1)≈0 and PC increasing with V.
3. Accuracy-path defaults updated (V=10, hx=0.5) only where verified non-regressive.
4. Integrability caveat documented in code + UI.
5. Webapp still imports and the free-run page runs end-to-end with the new knob.
6. Findings written up; `presentation/` untouched unless a number there changes.

## 5. Out of scope (named, not done)

- Hardware / real quantum device; this stays exact classical simulation.
- Full Step-3 stochastic ensemble calibration (separate `Anomaly_Forecast/plan.md`
  item) — training-noise here is complementary but not that.
- Retuning the deck's headline numbers unless a measured result changes.
