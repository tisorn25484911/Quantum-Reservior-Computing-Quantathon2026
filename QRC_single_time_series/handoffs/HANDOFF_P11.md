# HANDOFF_P11 — Sequential QPU feasibility for the rewind architecture

Written per G11. In-repo (declared deviation, as P0–P10).

## 1. Objective & status vs exit criteria

| Exit criterion | Status | Evidence |
|---|---|---|
| Per-step latency ledger (construct…buffer), measured + estimated split | ✅ | `hardware/latency_model.py`, `measured_local_s` + `assumptions` in the JSON |
| Remote timing from a frozen assumptions file beside the snapshot (never live) | ✅ | `results/backend_snapshots/latency_assumptions.yaml`, loaded by `LatencyAssumptions` |
| Heron-class execution anchors recorded as ASSUMPTIONS, not facts | ✅ | `sources:` in the yaml; `estimated: true` flag on every remote ledger |
| Strategy comparison A–G (incl. surrogate arm F = Gate-7 comparator) | ✅ | `STRATEGIES`, `compare_strategies`, `fit_surrogate` |
| Totals for 1/12/H_eff/100/1000 + one backtest | ✅ | `sequential_totals` → `totals` in the JSON |
| Accuracy-vs-latency Pareto (spec fig. 40) | ✅ | `pareto_points` → `results/figures/latency_pareto.png` |
| Batched open-loop NOT used as closed-loop evidence (item 24) | ✅ | totals are strictly `n × (one round trip)`; test asserts linearity |
| Per-mode verdicts with §26.4 practicality criterion | ✅ | `deployment_verdicts` → `verdicts` in the JSON |
| Suite passes | ✅ | **243 passed, 1 skipped** (~54 s) |

## 2. Artifacts & keys

- `hardware/latency_model.py` — `LatencyAssumptions` (loads the frozen yaml;
  `execute_s = sampler_overhead + shots·per_shot`), `build_rewind_step_circuit`
  (`readout="save_dm"` for Aer, `"measure"` for transpile timing — Aer saves don't
  translate to a device basis), `measure_local_components` (wall-clocks
  construct/bind/transpile/readout/local-execute/buffer on THIS machine), `STRATEGIES`
  (A–G component maps), `step_ledger`, `total_time`.
- `hardware/sequential_execution.py` — `fit_surrogate` (MLP window→feature map with
  its OWN ridge readout, on real ENSO; the Gate-7 comparator), `accuracy_by_strategy`,
  `compare_strategies`, `sequential_totals`, `pareto_points`, `deployment_verdicts`
  (`DECISION_DEADLINES_S`: monthly/daily/intraday).
- `results/backend_snapshots/latency_assumptions.yaml` — the ONE place remote timing
  lives (queue 60 s, net 0.25+0.25 s, per-shot 4e-4 s, sampler overhead 2 s, refresh
  R=12); every value labelled an assumption.
- Script: `scripts/estimate_sequential_latency.py` (`make sequential-latency`) →
  `results/metrics/sequential_latency.json` + `results/figures/latency_pareto.png`.
- Tests: `tests/test_sequential_latency.py` (13).

## 3. Key results

**Measured local components (N=5, t_w=10, Aer, this machine).** construct 0.47 ms,
bind 0.055 ms, transpile 3.45 ms, readout 0.80 ms, local-execute 3.68 ms, buffer
<0.001 ms; transpiled depth 44. **All local build costs are sub-10 ms — noise next to
the remote round trip.**

**Per-step + sequential totals (H_eff=4 from ENSO median 4.5; backtest = 20 origins ×
4 = 80 steps).** Remote round trip is dominated by the queue ASSUMPTION (60 s):

| strategy | per-step | 12-step | H_eff | 100 | 1000 | backtest |
|---|--:|--:|--:|--:|--:|--:|
| A fresh circuits | 62.914 s | 755 s | 252 s | 6291 s | 62 914 s | 5033 s |
| B pre-transpiled template | 62.911 s | 755 s | 252 s | 6291 s | 62 910 s | 5033 s |
| C dynamic circuits | 62.910 s | 755 s | 252 s | 6291 s | 62 910 s | 5033 s |
| D local noisy sim | 0.0084 s | 0.1 s | 0.03 s | 0.8 s | 8 s | 0.7 s |
| E offline feats + classical | 0.0008 s | — | — | 0.1 s | 0.8 s | 0.1 s |
| F classical surrogate | 0.0009 s | — | — | 0.1 s | 0.9 s | 0.1 s |
| G periodic refresh (R=12) | 5.243 s | 63 s | 21 s | 524 s | 5243 s | 419 s |

A/B/C are within 4 ms of each other: **once a real queue is in the loop, the
construct/transpile savings of pre-transpiled templates (B) and dynamic circuits (C)
are unmeasurable** — the honest engineering conclusion is that the round trip, not the
client-side build, is the cost.

**Surrogate (arm F, Gate-7 comparator, ENSO).** QRC one-step held-out NMSE **0.361**;
the classical MLP surrogate (window→features + its own ridge) reaches **0.403** at
**~69 µs/step** and no QPU. A laptop model lands within ~4 NMSE points of the quantum
feature map essentially for free — the playbook's "does the business question need the
QPU?" answer for this problem size is *not on latency/cost grounds*.

**Accuracy-vs-latency Pareto (`latency_pareto.png`).** Only **D (local sim, QRC
accuracy, 8 ms)** and **F (surrogate, 0.9 ms)** are Pareto-optimal. Every remote-QPU
strategy (A/B/C at ~63 s, G at ~5 s) is strictly dominated by the local simulator at
equal accuracy. **E is carried but flagged NOT closed-loop capable** (it only owns
offline features; it cannot get quantum features for its own predicted inputs).

**Per-mode verdicts (§26.4, at the H_eff horizon).**
- *research sim (D):* practical, fully local, seconds — the study path.
- *delayed batch (B, queued):* practical for overnight/daily reporting; a whole-horizon
  forecast finishes well inside a daily window.
- *operational (B, live):* **feasible for monthly/daily indices** (minutes ≪ cycle),
  **infeasible for a sub-minute loop** (queue alone busts a 60 s deadline). For the
  project's monthly-index use case the binding constraint is **QPU access/cost, not
  per-forecast latency** — exactly the honest framing §26.4 asks for, not an automatic
  "operationally infeasible".

## 4. Decisions & deviations

- **Remote timing is an explicit assumptions file, never a live call.** Queue/network/
  execute live only in `latency_assumptions.yaml` beside the (empty, frozen) snapshot
  dir; the ledger tags them `estimated: true` and the script prints "(all ASSUMPTIONS)".
  Re-pricing the study = edit that one file.
- **Transpile is timed on a hardware-shaped circuit** (terminal Z measurement, no Aer
  `save_density_matrix`) because save instructions don't translate to a device basis;
  the density-matrix variant is still used for the local readout/execute measurement.
- **The surrogate gets its OWN readout**, not the QRC readout fed dirty features. On the
  clean, near-perfectly-predictable Mackey-Glass one-step task the QRC readout is
  ill-conditioned and amplifies tiny feature errors (surrogate NMSE ~21 — spurious), so
  the comparator was moved to real ENSO with a refit readout: a fair classical
  replacement pipeline, which is what a deployment would actually ship.
- **Strategy G is amortised, not simulated step-by-step:** per-step =
  (one B round trip + (R−1) surrogate steps)/R. Its reported accuracy is the QRC
  ceiling (its refresh-point best case); closed-loop drift toward the surrogate between
  refreshes is noted, not modelled as a curve.
- **H_eff and the backtest size are read from the stored climate campaign**
  (ENSO median 4.5 → 4; 20 origins), in-script, never hand-typed (gotchas meta rule);
  a documented fallback (4, 20) applies if the campaign JSON is absent.

## 5. Open issues / known gaps

- The queue figure (60 s) is a single assumed median; real fair-share waits span
  seconds→hours and are **not** modelled as a distribution. The conclusions are
  monotone in it (a larger queue only widens the local-vs-remote gap), so a sensitivity
  sweep is a nice-to-have, not decision-relevant.
- No real backend snapshot exists (`results/backend_snapshots/` holds only the
  assumptions file). Transpile timing uses an assumed line coupling map; a frozen Heron
  snapshot would tighten depth/transpile numbers but not change the ranking.
- Strategy C's dynamic-circuit win is asserted structurally (still one round trip per
  forecast step); it is gated behind `requires_dynamic` and would need a snapshot that
  confirms mid-circuit feedforward support to price precisely.
- G's between-refresh accuracy is reported at the ceiling; a closed-loop
  accuracy-vs-R curve (surrogate drift) is the natural follow-up if G is pursued.

## 6. Entry instructions for Phase 12 (product track, final report, acceptance sweep)

1. `make test` green (243 passed, 1 skipped); `make sequential-latency` regenerates the
   ledger + Pareto.
2. `PRODUCT_STRATEGY.md`: playbook intake as testable claims; users/decisions (§28.1–2);
   ONE designed downstream experiment with arms A–E and the named operational dataset
   (§28.3); execution out of scope.
3. Probabilistic layer (§28.5): ensembles + block bootstrap; coverage/width/PIT/CRPS;
   the "measurement randomness ≠ future uncertainty" caveat.
4. Gates 1–7 in-script from stored results only; **Gate-7 comparator is the P11 arm F**
   — surrogate NMSE 0.403 vs QRC 0.361 at ~69 µs and no QPU sits next to accuracy in the
   go/no-go (a model matching a laptop has answered the business question).
5. `scripts/generate_report.py` assembles §33 with each conclusion labelled
   observed/statistical/theoretical/inferred/speculative; sixteen §33 questions each
   with an evidence pointer (Q16 recommendation should cite this Pareto: local
   sim / classical surrogate dominate the sequential QPU at this problem size).
6. `citation-audit` over `references.bib`; finalise `LIMITATIONS.md`; acceptance table
   checked item-by-item with test names. Write `HANDOFF_P12.md` (terminal).

## 7. Suggested skills

`quantum-reservoir-computing` + `qrc-project-playbook` (min per G11). For P12 add
`citation-audit` (references.bib) and lean on the playbook's Gate-7 / definition-of-done
framing.
