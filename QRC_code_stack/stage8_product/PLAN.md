# Stage 8 plan - product pipeline

Status: BUILT AS A WORKING SKELETON (Phase 6, 2026-07-14; user upgraded
scope from plan-only to code+plan). part11.tex read in full. Every layer
below exists as a runnable, self-tested module; the end-to-end
walk-forward demo (`exp_product_demo.py`) drives all six layers on the
labelled SOLAR SURROGATE through the same code path a live deployment
would use, and its Phase 8-9 acceptance gates (leakage audit, coverage
within 3 pts, battery in every row, valid alerts) all PASS. Module
self-tests are stage-0 anchors (test_product_anchors.py).

What exists per layer:
- L1 data_plane/connectors.py - the dataset registry AS CODE with full
  acquisition detail per source (SURFRAD/NSRDB/ERA5/SCADA recipes,
  licences, sizes); runnable today: real ENSO cache + labelled solar
  surrogate. qc.py - mask-never-interpolate + autocorr gap splitting +
  gap reports.
- L2 data_plane/transforms.py - the leakage firewall as an architectural
  component: fit-once frozen versioned statistics (ClearSkyIndex,
  Anomaly, MinMax), refusal on unfitted apply; calendar lookup features.
- L3 serving/feature_provider.py - the FeatureProvider protocol:
  SimProvider (stage-6 exact, production default at n<=11),
  CachedProvider (memo on quantised windows; 96% hit rate in the demo),
  HardwareProvider stub carrying the stage-7/5 wiring recipe + TVD CI
  gate. serving/heads.py - pinball quantile heads (IRLS), logistic event
  head, per-horizon leak. serving/conformal.py - split + adaptive
  conformal, rolling CoverageMonitor. serving/shadow_battery.py -
  battery columns on every prediction row.
- L4 decision/ - isotonic recalibration + expected-cost thresholds per
  persona cost matrix (cost units NEXT TO F1), regime router
  (clear/convective/overcast/night + CUSUM changepoint), stacker
  fairness table (ridge + GBM on hybrid vs classical-only; zero-delta
  stated), alert engine with the Part XI provenance payload, dedup,
  escalation, fatigue caps; data_kind=surrogate propagates into alerts.
- L5 surfaces/PLAN.md - API/dashboard/agent spec (deliberately not
  built as services; the alert schema and model cards they consume ARE
  built).
- L6 governance/ - KS drift monitor, effective-rank collapse monitor
  (the QRC-specific health metric), promotion gates as data with the
  contractual quantum-demotion path, model cards incl. nulls +
  surrogate banners.

NOT built: real services (API server, dashboard, chat agent), real
Track-A data (fetch recipes in connectors.py; user decision pending),
Phase-12 pilot (needs a design partner). Original Part IX skeleton
below for reference.

## Directory layout (Part IX tree)

```
stage8_product/
  data_plane/   connectors.py qc.py transforms.py feature_store.py replay.py
  serving/      feature_provider.py heads.py conformal.py shadow_battery.py
  decision/     stacker.py regime_router.py thresholds.py alerts.py
  surfaces/     api.py dashboard/ agent/     # tools, guardrails, RAG
  governance/   monitors.py promotion_gates.py model_cards.py
```

## Known constraints (Part IX sec. on Stages 6-8)

- data plane carries a leakage firewall (train-span-only statistics,
  enforced structurally, not by convention);
- serving core sits behind the `FeatureProvider` interface - the only
  surface the decision layer may touch;
- decision layer is calibrated (conformal);
- product surfaces include a provenance-guarded agent;
- governance promotion gates ARE the pre-registered criteria of Parts
  X-XI - no new thresholds invented at this stage.

Everything here imports only frozen stage 6-7 interfaces.
