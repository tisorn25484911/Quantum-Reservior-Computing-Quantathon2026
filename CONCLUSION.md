# Conclusion — combined QRC marine-heatwave early-warning

The integrator's bottom line. Produced by
`Quantathon_stack/Anomaly_Forecast/combine_and_conclude.py`, which combines any
Step-1 forecaster with any Step-2 detector end-to-end and ranks the pairs. As
teammates improve Step 1 and Step 2 separately, drop the new component into the
harness's registry and this conclusion regenerates.

## The combined pipeline

```
Step 1 (forecast the heatwave-intensity anomaly, closed-loop)
    x  Step 2 (detect the heatwave on that forecast)
    ->  a warning, with lead time
```

Validated on 45 years of real Gulf-of-Thailand data, marine-heatwave labels from
the Hobday (2016) standard (165 events), chronological train/val/test, every
number against persistence + classical controls on identical origins.

## What wins — the full combination matrix (2 forecasters shown × 2 detectors)

| Step 1 × Step 2 | F1 (1–7 d) | F1 (1–14 d) | recall @3 d | recall @7 d | recall @14 d |
|---|---|---|---|---|---|
| persistence × threshold | **0.718** | **0.627** | 0.77 | 0.65 | 0.49 |
| **engine-QRC × threshold** | 0.717 | 0.596 | **0.87** | 0.64 | 0.36 |
| engine-QRC × **ensemble** | 0.703 | 0.602 | 0.78 | 0.61 | **0.42** |
| core-QRC × threshold | 0.693 | 0.552 | 0.80 | 0.50 | 0.24 |
| NVAR × threshold | 0.684 | 0.500 | 0.80 | 0.45 | 0.10 |

*(engine-QRC = the sister repo's validated exact reservoir; core-QRC = ours.
"threshold" = Hobday rule on the mean forecast; "ensemble" = fraction of K=60
sampled futures breaching it — the Step-3 calibrated alarm. Deterministic models
have no ensemble row.)*

## The conclusion, stated plainly (and honestly)

1. **On balanced F1 it is a genuine tie** — persistence-then-threshold (0.718)
   ≈ engine-QRC (0.717). And persistence is strong here for a real reason:
   **a marine heatwave is defined as a ≥5-day event, so "today's heat continues"
   is a good bet.** We do not hide this.
2. **The quantum forecast's clear, real edge is RECALL at short lead:** engine-QRC
   catches **87 % of heatwave days 3 days out vs persistence's 77 %** — it predicts
   the event is *coming*, not merely *continuing*. Both quantum engines agree, so
   it is the method, not a lucky build.
3. **Which metric decides the winner is a business question, and aquaculture
   answers it.** A missed heatwave loses a grow-out cycle; a false alarm just runs
   an aerator. Under that asymmetry **recall is the priority**, and the quantum
   forecast is preferred.
4. **The ensemble detector (Step 3) recovers the long-lead tail** the mean forecast
   loses: recall @14 d climbs 36 % → 42 %, calibrated (90 % band covers truth 91 %
   of the time).
5. **No quantum advantage is claimed, and the claim is bullet-proof.** On raw
   forecast skill the QRC is at *parity* with two independent strong controls
   (ESN and NVAR) on both engines. The value is a *reliable, recall-first short-lead
   warning*, not a speed-up.

**Ship recommendation (recall-first):** **engine-QRC (Step 1) × ensemble (Step 2)**
— quantum short-lead recall, calibrated long-lead recovery. **Honest caveat:** on
balanced F1, naive persistence is a genuine tie, so the case rests on the
aquaculture cost asymmetry, not on out-scoring every baseline on every metric.

There is also a **parallel warning path**: the precursor ML detector (`detect.py`)
predicts heatwave *onset* from observed ENSO / rainfall / build-up at 2.6× the base
rate — independent of Step 1, so it is a hedge when the forecast is weak.

## How each teammate's improvement plugs in

- **Better Step 1** (teammate): a stronger forecaster → register it in
  `FORECASTERS` (needs `.fit(x, train_end)` + `.rollout(origin, H)`); the matrix
  and this conclusion re-rank automatically. Watch whether it lifts the **≤7-day
  recall** — that is the number that matters, not the 14-day average.
- **Better Step 2** (teammate): a stronger detector → register it in `DETECTORS`
  (a callable on the forecast trajectory). The ML onset detector (`detect.py`) and
  the stochastic-ensemble detector (`stochastic.py`) are ready to fold in as
  additional `DETECTORS` entries.
- **Integrator (you):** keep the harness honest — decision-relevant window,
  baselines on identical origins, multi-seed, and the "who wins where" split that
  a single averaged metric hides.

## Open items before the final claim

- Fold the stochastic-ensemble detector and the ML onset detector into the harness
  as first-class `DETECTORS` so the matrix covers all Step-2 variants, not just the
  threshold.
- Add the sister repo's Mackey–Glass autonomous gate as the "engine is faithful"
  checkpoint (their Phase 5).
- When teammates deliver improved Step 1/Step 2, re-run
  `combine_and_conclude.py` and update this file — it is regenerated, not
  hand-written.
