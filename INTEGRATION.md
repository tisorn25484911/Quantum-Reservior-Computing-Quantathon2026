# Integration plan — combining the two repos into one stronger project

Two repos, built in parallel, that are two halves of one project. This is the
concrete plan to combine them over the remaining days. Written 2026-07-24.

- **APPLIED** (this repo) — real Gulf-of-Thailand data, marine-heatwave labels, the
  Forecast→Detect→Warn pipeline with end-to-end results, the report and deck.
- **ENGINE** ([`Quantathon2026-QRC_detection`](https://github.com/tisorn25484911/Quantathon2026-QRC_detection)
  → `QRC_single_time_series/`) — a rigorously tested QRC core (phases 0–2 green,
  216 tests), a baseline battery, and an evaluation framework, on generic climate
  indices (ENSO/PDO/SOI). Product track designed, not executed.

## The thesis for combining

The ENGINE repo's own `PRODUCT_STRATEGY.md` says *"climate-index forecast accuracy
alone is NOT a product; a second-stage experiment linking predicted indices to an
operational target is required."* **The APPLIED repo is that second stage.**
Conversely, the APPLIED repo's results need the ENGINE's rigor (validated core,
baseline battery, horizon taxonomy) to survive scrutiny. Each supplies exactly
what the other lacks.

## Division of labour (who owns what)

| Concern | Owner | Why |
|---|---|---|
| Validated QRC core (exact / Trotter / Qiskit / noisy) | ENGINE | 216 passing tests, NARMA2 6-orders proof |
| Baseline battery (ESN, NVAR, LSTM, GRU, NVAR, statistical) | ENGINE | already scaffolded; APPLIED had only 1 ESN |
| Autonomous-horizon taxonomy (H_skill/H_reliable/H_effective) | ENGINE | rigorous answer to "which range" |
| Uncertainty / calibration | shared | APPLIED `stochastic.py` == ENGINE `evaluation/uncertainty.py` |
| Real driver data + provenance (Gulf SST, ONI, rain, SPEI, HadISST) | APPLIED | acquired + loaders |
| Anomaly **labels** (marine heatwaves, ENSO episodes) | APPLIED | Hobday 2016, validated |
| Forecast→Detect→Warn composition | APPLIED | the operational second stage |
| Product framing, report, pitch deck | APPLIED | the narrative |

## Reconcile the duplication (do this first — you are building two of each)

1. **QRC engine.** ENGINE `quantum/exact_qrc.py` (heavily tested) vs APPLIED
   `Hamiltonian_QRC/qrc_core.py` + `forecast.py` (has the closed-loop rollout +
   detection glue the ENGINE lacks). **Decision: ENGINE core is canonical;** port
   APPLIED's `rollout`/`_QRCDriver` on top of it as the forecasting layer. Keep
   `qrc_core.py` as a validated fallback until the port passes the same anchors.
2. **Uncertainty.** Fold APPLIED `stochastic.py` (residual bootstrap + coverage
   gate + ensemble alarm) into ENGINE `evaluation/uncertainty.py` — same idea,
   one implementation.
3. **Horizon.** Replace APPLIED's ad-hoc "useful lead" with ENGINE
   `evaluation/prediction_horizon.py` (H_skill / H_reliable / H_effective).
4. **Baselines.** Use ENGINE's battery everywhere APPLIED currently shows only
   ESN. **Started already:** `nvar_baseline.py` here reproduces the NVAR control —
   result: QRC is at *parity* with NVAR too (mean NMSE 0.633 vs 0.653), so the
   "parity with classical, no quantum advantage" claim now holds against two
   independent strong controls, not one.

## Merge mechanics (pick one)

- **Option A — monorepo (recommended).** Add the ENGINE as
  `git subtree add --prefix engine <engine-repo> main`. APPLIED code imports
  `engine.qrc_single_time_series...`. One repo, one venv, both histories kept.
- **Option B — submodule.** `git submodule add` the ENGINE under `engine/`.
  Lighter, but contributors must `--recurse-submodules`.
- **Option C — package.** ENGINE stays its own repo, publishes an installable
  package; APPLIED depends on it by version. Cleanest long-term, most setup now.

For a few-days deadline, **Option A** keeps everyone moving with the least friction.

## Sequenced plan for the remaining days

**Day 1 — wire the engine in (no science yet). ✅ DONE 2026-07-24.**
- ENGINE vendored under `engine/` via `git subtree add --prefix engine ... --squash`
  (5 MB). Its Phase 1–2 core test suite runs **green in this repo** under the shared
  `.venv` (qiskit/noise tests skipped — Phase-3 deps, not yet installed).
- Integration anchor `Quantathon_stack/Anomaly_Forecast/engine_integration.py`:
  drives **their** exact FN reservoir with **our** `got_sst_mhwi` via our loader →
  1-step held-out **NMSE 0.0849** (same ballpark as our `qrc_core.py`'s 0.077).
  So the applied pipeline can now stand on the engine's validated core. Convention
  differences (their V=10/z_local vs our V=4/z_and_zz) mean features are not
  bit-identical — a later task is to align configs and assert ~1e-10, or simply
  adopt the engine's config as canonical.

**Day 2 — re-run the headline on the canonical engine. ✅ DONE 2026-07-24.**
- `engine_step1.py` runs Step-1 on `got_sst_mhwi` through the ENGINE's validated
  exact FN reservoir (a fast stateful stepper built on their per-step ops), scored
  on identical origins against persistence + ESN + NVAR (their model battery is
  still stubs — Phase 7 — so the classical field is ours for now).
- **Result:** engine-QRC mean NMSE **0.623** (ESN 0.657, NVAR 0.659, persistence
  0.797). Taxonomy: **H_skill = 24, beats persistence at all 24, classical never
  beats it** (QRC edges best-classical at 5/24 short leads). Cross-check: our
  `qrc_core` gives 0.633 — **consistent**, so the two engines tell the same story.
  Verdict: *parity with classical, no quantum advantage — confirmed on the
  validated core.*

**Day 3 — compose + rigor gates.**
- Re-run Detect→Warn (`compose.py`, `stochastic.py`) on the canonical engine;
  fold uncertainty into the ENGINE module. Add the ENGINE's Mackey–Glass
  autonomous gate as the "the engine is faithful" checkpoint.

**Day 4 — one narrative.**
- Single report + deck: APPLIED's Gulf marine-heatwave story as the product, with
  the ENGINE's rigor (tests, baseline battery, horizon taxonomy, calibration) as
  the credibility backbone. Update `COMBINED_PLAN.md` to the merged state.

## Guardrails (unchanged, now enforced by both sides)

- No quantum-advantage claim at ≤8 qubits / exact sim; parity vs the *battery* is
  the ceiling. Now tested against ESN **and** NVAR — both parity.
- Every number multi-seed, chronological splits, train-only fits, baselines on the
  same rows. Both repos already do this; the merge must not regress it.
- The Gulf marine-heatwave result is the product; the climate-index work is the
  engine's proving ground. Keep synthetic and observational data in separate
  tables (ENGINE spec §2).
