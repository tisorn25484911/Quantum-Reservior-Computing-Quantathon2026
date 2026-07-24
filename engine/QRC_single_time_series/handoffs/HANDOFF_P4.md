# HANDOFF_P4 — Autonomous rollout engine, effective-horizon metrics, failure taxonomy

Written per G10. In-repo (declared deviation, as P0–P3). Entire phase is
physics-free and unit-tested against a toy AR(1); no quantum object involved.

## 1. Objective & status vs exit criteria

Phase-4 goal: spec §§16,17,19,21 as a fully unit-tested layer, buildable and
testable before any reservoir exists.

| Exit criterion (spec acceptance 8–13) | Status | Evidence |
|---|---|---|
| G5 no-future guard (mutate future → identical) | ✅ | `test_autonomous_no_future_access.py` |
| Recursive feedback identity (own predictions fed back) | ✅ | `test_recursive_feedback.py` |
| Rewind window provenance mask | ✅ | `test_rewind_predicted_window.py` |
| Feedback scaling round-trip identity in-range | ✅ | `test_feedback_scaling.py` |
| CNRMSE/CRMSE/CNMAE on hand-computed sequences | ✅ | `test_accumulated_error.py` |
| H_error prefix rule (no re-validation after failure) | ✅ | `test_effective_prediction_time.py` |
| H_skill / H_reliable / H_effective (all components reported) | ✅ | `test_prediction_horizon.py` |
| Failure taxonomy on constructed pathologies | ✅ | `test_failure_classification.py` |
| Suite passes | ✅ | **182 passed, 0 skipped** (~13s) — every placeholder now real |

## 2. Artifacts & keys

Modules (`src/qrc_single_time_series/`):
- `evaluation/autonomous.py` —
  - `rollout(model, history, n_steps, policy)` — closed loop; reads ONLY
    `history`; returns `{predictions, terminated_at, telemetry}`. Model protocol
    `warm(history)->state`, `step(state,u)->(state,yhat)`. First step consumes the
    last real value, then feeds predictions back.
  - `autonomous_from_series(model, series, origin, n_steps, policy)` — **G5 made
    structural**: copies `series[:origin]` so the future is physically inaccessible.
  - `DomainPolicy` / `make_policy(kind, lo, hi)` — kinds `hard_clip`,
    `smooth_bounded` (logistic into [lo,hi]), `terminate` (stop on violation),
    `wide_encoding`. Telemetry: n, n_clipped, clip_fraction, first_clip_t,
    max_violation, boundary_dwell(+fraction), terminated_at. None silent.
  - `make_origins(n_total, warmup, n_steps, n_origins, spacing)` +
    `multi_origin_horizons(model_factory, series, origins, ...)` — spec §21 driver;
    per-origin H_effective, ties rollout+error+horizon; fresh model per origin.
- `evaluation/accumulated_error.py` — `sigma_train` (train-only, zero-guard),
  `instantaneous_ne`, `crmse`/`cnrmse`/`cnmae` (running-average arrays indexed by
  h-1; they dip after a spike — motivates the prefix rule).
- `evaluation/prediction_horizon.py` — `h_error(ne, eps, K)` (first run of K
  consecutive breaches; **first sustained run wins, later dips do not
  re-validate**), `h_skill(cnrmse_model, cnrmse_base, K)`, `h_reliable(first_fail,
  n)`, `h_effective(...)` (min + **every component returned**, spec 17.6),
  `survival_curve`, `h_product(p)`.
- `evaluation/failure_modes.py` — `FailureRules` (`.from_config` reads
  `preregistration.yaml::failure_rules`), `TrainStats.from_series`,
  `classify(traj, stats, rules, clip_fraction, boundary_dwell_fraction, ...,
  ensemble)` → `{label: first_failure_step}`. Modes: blowup, fixed_point,
  mean_collapse, variance_collapse, variance_explosion, spurious_cycle, saturation,
  stochastic_instability. Multiple labels allowed; window=24.
- `evaluation/statistics.py` — `stationary_bootstrap_ci(values, block_length,
  n_resamples, alpha, statistic)` (Politis-Romano geometric blocks over origins),
  `diebold_mariano(loss_a, loss_b, h)` (Harvey-Leybourne-Newbold small-sample
  correction; positive DM ⇒ A worse).
- `models/recursive_forecaster.py` — `AR1(phi, c)` toy (the horizon-test fixture),
  `RewindBuffer(length)` (values + predicted mask; `push`, `all_predicted`,
  `provenance_fraction`), `EncodedRecursive(encode, decode, feature_fn, readout)`
  (the shape stateful-FN / rewind reservoirs plug into later).
- `data/preprocessing.py` — added `invert_scaler(bounds)` (decode s∈[0,1] → physical;
  round-trip identity in-range).

Tests: the 8 P4 files (36 tests). No new script/result artifact this phase.

## 3. Frozen failure-rule thresholds (regression fixture)

From `configs/preregistration.yaml::failure_rules` (G6, unchanged):
blowup 1.5×train-range · fixed_point std 0.05·σ · mean_collapse 0.10·σ (AND low
std) · var_collapse 0.25×var · var_explosion 4.0×var · spurious_cycle peak >0.60
of power · window 24 · saturation: clip_fraction>0.20 or boundary_dwell>0.30.

Toy-model horizon fixture (regression anchor): `AR1(phi).rollout(y0, n)` gives the
exact geometric `y0·phi^k`; `AR1(phi,c)` converges to the fixed point `c/(1-phi)`.

## 4. Decisions & deviations

- **H_error semantics**: horizon = count of leading steps before the FIRST run of K
  breaches begins (`_first_sustained_breach` returns the run's first index). Dip-back
  curves are explicitly tested to NOT re-validate.
- **mean_collapse requires low variance too** (flat line near μ), else a mean-
  crossing oscillation would false-trigger. Documented in code.
- **Stateful-FN / rewind reservoir adapters are stubs-of-shape only** — `AR1`,
  `RewindBuffer`, `EncodedRecursive` satisfy the protocol; the actual reservoir
  `warm/step` (retain ρ across steps; rewind provenance buffer feeding the reservoir)
  are wired in P5 (Mackey-Glass gate) when the quantum model plugs in.
- **Readout error / DM caveat carried from P3** still stands for the quantum path.
- No commit made (user hasn't asked).

## 5. Open issues / known gaps

- `multi_origin_horizons` uses a `None`-failure `h_reliable` placeholder (failure
  classification not yet wired into the driver) — connect `failure_modes.classify`
  → first-failure step → `h_reliable` in P5/P8 when real trajectories exist.
- `EncodedRecursive.feature_fn`/`readout` are user-supplied; the reservoir versions
  arrive with P5.
- Block length for the bootstrap is passed in; the decorrelation-time estimator
  (first ACF zero-crossing / 1/e, train-only, per G6) is computed in P8/P9 EDA.

## 6. Entry instructions for Phase 5 (Mackey–Glass autonomous reproduction — GATE)

1. `make test` green first (182 passed, 0 skipped).
2. Implement `dynamical_systems/mackey_glass.py` (τ_MG=16 non-chaotic, τ_MG=17
   chaotic; audited integrator). Wire the **stateful-FN adapter** (retain ρ across
   autonomous steps — spec §16.1, never reinitialise) into `EncodedRecursive`-shape
   via `ExactQRC` features + trained `Readout`, plugged into `rollout`.
3. Reproduce the teacher-forced → autonomous transition (H1): non-chaotic tracking
   at τ_MG=16; finite valid time then attractor-like divergence at τ_MG=17. Use the
   P4 horizon/failure/telemetry layer as-is.
4. This is a GATE — record the reproduction against the pre-registered bands; write
   `HANDOFF_P5.md`.

## 7. Suggested skills for the next session

`quantum-reservoir-computing` + `qrc-project-playbook` (min per G10); add
`citation-audit` only when a document ships.
