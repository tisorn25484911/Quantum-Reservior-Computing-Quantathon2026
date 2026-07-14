# Surfaces (L5) - plan (Phase 11)

Deliberately NOT built as services in this repository: API servers,
dashboards, and chat agents are deployment infrastructure whose value is
in operation, not in a repo demo. What IS in the repo: the alert schema
(decision/alerts.py, the wire format all surfaces consume), the model
cards (governance/model_cards.py, the dashboard's non-removable
benchmark table source), and this specification.

## API (REST + streaming)

- `GET  /v1/sites/{site}/forecast?horizon=&quantiles=` -> quantile fan
  + battery columns + run_id (battery is NON-REMOVABLE from the payload)
- `GET  /v1/sites/{site}/alerts?since=` and `POST webhook` subscription
- `GET  /v1/benchmark/{site}` -> the honest table (one call, always)
- `GET  /v1/model_cards/{version}` -> card JSON
- Alert payload = decision/alerts.py Alert.to_json() verbatim; the
  degrade-graceful ladder keeps this schema at every rung.
- Auth: per-site API keys; researcher tier read-only + seed-exact replay.

## Dashboard

Panels: live k_t + fan + alerts; regime strip; rolling coverage vs
target; benchmark table (non-removable BY DESIGN - it renders the
battery columns from the same rows the API serves); alert-fatigue
budget burn-down; model card link.

## Agent assistant (adopted patterns 1-5, rejected 6: never actuate)

Tool registry (all read-only; numbers ONLY via tools):
  get_forecast(site, horizon)          -> forecast row incl. battery
  get_alert_rationale(alert_id)        -> regime, attributions, analogues
  run_whatif_backtest(site, window, policy) -> cost delta vs baseline
  get_benchmark_table(site)            -> the honest table
  get_model_card(version)              -> card
  configure_site_draft(nl_description) -> DRAFT config for human approval

Guardrails (non-negotiable): numeric-provenance verifier - every number
in a reply must trace to a tool call, rendered with run_id, zero
tolerance in the Phase-11 audit; uncertainty phrasing bound to monitored
conformal coverage; refusal templates outside validated horizons and
regimes; quantum features described as "quantum-inspired simulated
reservoir features" with the measured ablation delta - no advantage
language. Orchestration: single tool-calling agent first; planner-
verifier pair only if transcript audits show the need.

## Acceptance (Phase 11, unchanged from Part XI)

Grounded-answer evaluation set passes with every numeric claim in
sampled transcripts tracing to a tool call (zero fabricated numbers);
time-to-first-alert for a new site under one day.
