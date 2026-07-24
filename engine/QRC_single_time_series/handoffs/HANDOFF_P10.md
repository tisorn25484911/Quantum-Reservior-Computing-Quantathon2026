# HANDOFF_P10 — Shot budgets, noise models, feature-space analysis

Written per G10. In-repo (declared deviation, as P0–P9).

## 1. Objective & status vs exit criteria

| Exit criterion | Status | Evidence |
|---|---|---|
| Consistent shot-emulation formula across models | ✅ | `quantum/mitigation.emulate_shots` |
| Feature-space suite (spec §27) | ✅ | `evaluation/diagnostics.py` |
| Horizon/NMSE-vs-shots curves with exact ceiling first | ✅ | `compare_shot_budgets.py` |
| Three-arm adjudication (noise vs ridge vs SVD-truncation) | ✅ | `compare_noise_models.py` + `mitigation.three_arm_adjudication` |
| SVD-truncation mitigation arm (Ahmed–Tennie–Magri) | ✅ | `mitigation.svd_truncate_fit` |
| Signal-to-noise ceiling check gates the study | ✅ | exact NMSE reported first in both scripts |
| Suite passes | ✅ | **230 passed, 1 skipped** (~47 s) |

## 2. Artifacts & keys

- `evaluation/diagnostics.py` — `effective_rank_hhi` (1/HHI, Hamhoum Eq. 17 on
  standardised features), `condition_number`, `explained_variance`, `feature_error`
  (bias/var/RMSE/corr), `cosine_similarity_rows`, `covariance_distance`,
  `readout_coefficient_stability`, `feature_space_report`.
- `quantum/mitigation.py` — `emulate_shots` (the ONE shot formula; bias column kept
  exact), `svd_truncate_fit`, `three_arm_adjudication`.
- Scripts: `run_noisy_qrc.py` (`make feature-space`) →
  `results/metrics/noisy_feature_space.json`; `compare_shot_budgets.py`
  (`make grid-shots`) → `results/metrics/shot_budgets.json`; `compare_noise_models.py`
  (`make grid-noise`) → `results/metrics/noise_adjudication.json`.
- Tests: `tests/test_noise_feature_space.py` (7).

## 3. Key results

**Shots (one-step test NMSE, mean over 6 seed draws).** MG τ17: monotone toward the
exact ceiling — 128→0.0062, 8192→0.0021, exact 0.0001 (`shots_improve_toward_exact`
True). ENSO: **flat** (0.375–0.384 across the grid, exact 0.382) — the config already
sits at its low-signal ceiling, so shots buy nothing (`shots_improve_toward_exact`
False; a small NEGATIVE low-shot penalty −0.007 = a within-noise "noise regularises"
hint on ENSO only).

**Three-arm adjudication (H5).** MG: ridge (arm B, 0.0008) crushes training-on-noise
(arm A, 0.002–0.006) at every shot count — noise never helps. ENSO: arm A wins over
ridge/truncation at 3 of 4 shot counts but only by <1% and within the seed std, and
it LOSES at 512 shots → **not robust**. **Verdict: finite-shot noise does NOT robustly
beat a matched ridge / SVD-truncation on held-out error — the conditioning benefit is
recovered classically without noise. H5 holds.** (`adjudication_verdict` in the JSON.)

**Feature-space dashboard (spec §27).** As shots rise, RMSE-vs-exact falls
(0.042→0.005), row cosine → 1.0000, and the condition number climbs. Notable
diagnostic: **low shots INFLATE the effective rank** (MG 1.59 vs exact 1.18; ENSO
2.56 vs 1.22) — shot noise fills otherwise-empty singular directions, the mechanism
behind any apparent "noise adds capacity". It is spurious capacity (noise, not
signal), which is exactly why the three-arm test rejects it.

## 4. Decisions & deviations

- **Shot noise is emulated at the feature level** (the spec-§3 consistent formula),
  not by per-window Aer sampling — orders of magnitude cheaper and identical in
  distribution for the FN expectation-mode features (G8-7: FN has no per-shot
  back-action). The Aer noisy path (P3 `noisy_qiskit_qrc.py`) remains available for a
  hardware-shaped cross-check but is not needed for the decision-relevant curves.
- **"Noise helps" verdict tightened to REQUIRE robustness** — beating both classical
  arms by more than the seed std at EVERY shot count — after the lenient
  "any-shot-wins" rule flagged an ENSO within-noise coin-flip. Both the strict
  (`noise_robustly_helps…`) and lenient (`noise_helps_at_any_shot`) flags are stored.
- **Deployment reads the trained readout on EXACT test features** in the shot study:
  the question is whether training on noise conditions the readout, not whether test
  features are noisy — matches the "noise as regulariser" claim precisely.

## 5. Open issues / known gaps

- Full synthetic-channel (depol/thermal) and frozen-backend-snapshot **Aer** noisy
  runs are wired (P3 `noise_models.py`) but not swept here; the feature-level shot
  study is the decision path. A hardware-snapshot cross-check is a nice-to-have.
- Closed-loop horizon-vs-shots remains dominated by the P8 finding (shots collapse
  the small-N loop); the one-step curves here are the informative shot signal.
- The J·τ / input-gain sweeps (preregistered) still pending a full run — would set
  the stronger-signal config the closed-loop shot study wants.

## 6. Entry instructions for Phase 11 (sequential QPU feasibility)

1. `make test` green (230 passed, 1 skipped).
2. `hardware/latency_model.py` + `sequential_execution.py` (both stubs): per-step
   ledger (construct/bind/transpile/queue/network/execute/return/readout/buffer).
   Local components wall-clocked on Aer; queue/network ESTIMATED from a parameterised
   assumptions file beside the backend snapshot (never live), with Heron-class
   execution-time anchors recorded as assumptions.
3. Strategy comparison A–G (fresh circuits / pre-transpiled templates / dynamic
   circuits / local sim / offline features + classical deploy / classical surrogate
   of the feature map / periodic refresh). The surrogate arm doubles as the Gate-7
   cost comparator.
4. Totals for 1/12/H_effective/100/1000 steps + one backtest; accuracy-vs-latency
   Pareto. **Batched open-loop runtime is NOT closed-loop evidence** (item 24).
5. `hardware/` modules + a script; write `HANDOFF_P11.md`.

## 7. Suggested skills

`quantum-reservoir-computing` + `qrc-project-playbook` (min per G10); add
`citation-audit` when Ahmed–Tennie–Magri (QMI 2025) / Domingo et al. (2023) enter
`references.bib`.
