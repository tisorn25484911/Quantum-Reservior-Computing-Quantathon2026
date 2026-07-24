# HANDOFF_P6 — Rewind QRC (single channel) + learned-map Lyapunov machinery

Written per G10. In-repo (declared deviation, as P0–P5).

## 1. Objective & status vs exit criteria

| Exit criterion | Status | Evidence |
|---|---|---|
| Rewind reservoir (Hamhoum Alg. 1) implemented | ✅ | `quantum/rewind_qrc.py` |
| Exact mode = Stinespring dilation (P3-deferred) | ✅ | DM == dilated to **3e-16**; `test_rewind_qrc.py` |
| Rewind one-step beats the copy baseline | ✅ | MG17 & logistic, `test_rewind_beats_copy_*` |
| Jacobian & scalar λ̂ agree on a known map (logistic r=3.9) | ✅ | Benettin **0.5028** vs Rosenstein **0.5018** (true ~0.494) |
| Provenance buffer wired through the real model | ✅ | `test_rewind_predicted_window.py::test_real_rewind_model_driven_through_buffer` |
| FN-recurrent vs rewind pre-registered comparison | ✅ | recorded (§4) |
| Suite passes | ✅ | **201 passed, 0 skipped** (~17s) |

## 2. Artifacts & keys

- `quantum/rewind_qrc.py` — `RewindReservoir(reservoir, t_w, tau, kappa, order,
  observable_kind, ordering)`: |+⟩^⊗N init, per-step injection reset + Ry encoding,
  one NN-TFI (κ=1) Trotter block per window step (reuses P3 `schedule`/
  `block_unitary`), local-Z readout.
  - `window_features(window)` — density-matrix exact (N≤10).
  - `window_features_dilated(window)` — **Stinespring dilation**: each reset →
    fresh |0⟩ ancilla + SWAP into site 0, PURE statevector on `n_dilated_qubits()`
    = N+(t_w−1). Equals the DM local-Z to 3e-16.
  - `features_series(inputs)` — sliding-window feature matrix.
  - `copy_baseline_mse(inputs)` — the trivial persistence baseline.
- `evaluation/lyapunov.py` (P6 additions) — `jacobian(fmap, x, eps)` (central diff),
  `benettin_spectrum` / `benettin_largest` (QR product), `benettin_largest_interior`
  (verified-interior trajectory, stops at boundary — never differentiates through a
  clip, spec 20.3), `delay_map_from_prediction(predict_fn, clip=None)` (G(window) =
  [window[1:], F(window)]).
- Tests: `test_rewind_qrc.py` (11) + one added to `test_rewind_predicted_window.py`.

No new make target / script (P6 is analysis + machinery; test-backed). No experiment
registered.

## 3. Learned-map Lyapunov — the key finding (honest negative)

Machinery validated on the logistic map (r=3.9): Benettin **0.5028**, Rosenstein
**0.5018**, true ~0.494 — Jacobian/QR and the scalar estimator agree.

Applied to the trained rewind F-map on MG17 (interior Benettin, 85 interior steps):
**λ̂_Fmap = −0.0037** while the true series is chaotic (**Rosenstein +0.0055**). The
stateless windowed rewind map, driven autonomously, **contracts** — it does NOT
reproduce the chaotic positive exponent. On logistic the learned F-map leaves the
interior immediately (0 interior steps: autonomously unstable). This is a genuine
dynamical result, not a bug: **recurrence (stateful FN, which DID reproduce the
MG τ17 chaotic divergence in P5) carries the dynamics; windowing does not.** Matches
the pre-registered calibration prior that recurrence is worth more than tuning.

## 4. FN-recurrent vs rewind (pre-registered comparison)

On MG17 one-step (teacher-forced, N=5 seed=7): FN-recurrent NMSE ≈ 1e-4, rewind NMSE
≈ 0 — **both near-perfect, both crush the copy baseline (NMSE 0.0218)**. One-step is
too easy to separate them. The separation is in AUTONOMOUS dynamical fidelity (§3):
recurrence reproduces the attractor's positive exponent; the rewind F-map contracts.
Recorded verdict: **recurrence ≥ windowing for dynamical faithfulness on this task**,
consistent with the pre-registration.

## 5. Dilation memory (the P3-deferred measurement)

`n_dilated_qubits = N + (t_w − 1)`. Pure-statevector sizes (complex128):
N=12,t_w=10 → 21 qubits → **32 MB**; N=15,t_w=10 → 24 qubits → **256 MB** — feasible,
matching the P3 soundness review. **Caveat**: the current dilation *gate application*
builds dense 2^nq × 2^nq operators (fine for the small-N verification tests, but
4^nq is infeasible at nq=21). To actually reach N=12–15, the dilated path needs
sparse / tensor-contraction gate application (or Aer statevector with reset→ancilla+
swap). The dilation PRINCIPLE and exactness are verified; the memory-efficient
executor is a P8/P10 optimisation when large-N rewind runs are needed. Flagged.

## 6. Decisions & deviations

- **Input clip inside `window_features`** (u→[0,1]) matches the physical encoder and
  makes the F-map defined everywhere; Lyapunov uses interior trajectories so the clip
  is inactive where it differentiates (spec 20.3).
- **Two synthetic univariate tasks = MG17 + logistic** (both beat copy). The specific
  Vallis-ENSO-ODE and Lorenz reduced-to-univariate tasks are deferred to **P8**, where
  `dynamical_systems/{enso_ode,lorenz63}.py` are built. Documented deviation.
- Opt-NN gate ordering supported via `schedule(ordering=...)` but not swept here
  (P8/P10 Jτ + ordering sweeps).
- Shots ⇒ stochastic F-map ⇒ Lyapunov spectrum undefined (spec 20.3): only the exact
  (expectation) mode is used for the spectrum; shot runs would report divergence
  rates, labelled — deferred to the noise campaign (P10).
- No commit made by me yet — committing this phase per the user's per-phase workflow.

## 7. Open issues / known gaps

- Memory-efficient dilated executor for N≥12 (see §5).
- Vallis/Lorenz univariate rewind tasks (P8).
- `integration.py` still a stub (RK4 + Benettin-for-ODE) — P8.

## 8. Entry instructions for Phase 7 (baseline battery + structure controls)

1. `make test` green first (201 passed).
2. Implement `models/{esn,nvar,lstm,gru,qlstm,autoregression,statistical}.py` and the
   Haar-random-reservoir control. **G7**: the size-matched ESN matches the QRC's
   **effective rank** (~2 under scalar drive, per P2 §4 note), NOT the nominal
   feature count. `statistical.py` needs statsmodels: `uv pip install --python
   .venv/bin/python statsmodels==0.14.6`.
3. All recursive baselines feed their own predictions back through the SAME rollout
   engine (spec §24). Wire them as `warm/step` adapters (P4 protocol).
4. `scripts/run_baselines.py` + `make baselines`; write `HANDOFF_P7.md`.

## 9. Suggested skills

`quantum-reservoir-computing` + `qrc-project-playbook` (min per G10); add
`citation-audit` only when a document ships.
