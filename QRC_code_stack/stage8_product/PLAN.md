# Stage 8 plan - product pipeline

Status: placeholder (Phase 6 of the repo build). Source of truth: handbook
Part XI Phases 8-12 - which will be read in full and converted into a
detailed structural plan when this phase arrives. Recorded now from
Part IX only:

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
