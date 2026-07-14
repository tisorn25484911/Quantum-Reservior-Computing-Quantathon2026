# Stage 7 plan - RF-QRC x qubit-reuse integration + solar application

Status: PARTIALLY BUILT (Phase 5, 2026-07-14). Built and green:
`qreuse_batch.py` (batch/compile/unbatch over the UNTOUCHED stage-5
compiler) and `exp_rfqrc_reuse.py` (all promotion gates pass: per-block
self-calibrated TVD gate; feature interchangeability within shot noise;
compiled width < logical, constant in batch size B, and constant in
memory horizon T while the windowed protocol's width grows with T --
the structural claim, recomputed live). Anchors in
stage0_anchors/test_batch_anchors.py (batching preserves per-block
distributions to 1e-12; hand-built unbatch partition; legality).
Topology note, measured: an all-to-all (Haar) block shows no
WITHIN-block compression but the batch still compresses across block
boundaries -- two separate facts, both reported. Haar blocks must be
transpiled to (u, cx) before the frozen parser sees them.

NOT built, deliberately:
- Solar-ramp study (`solar_ramps_data.py`, `exp_solar_ramps.py`):
  plan-level only. No solar data exists locally; SURFRAD download (NOAA
  ftp, per-station yearly files) or NSRDB (API key) awaits explicit user
  approval, and the vetting checklist (playbook sec. 2) must run against
  the provider pages first. Nothing here is blocked by code.
- `exp_vn_fusion_proto.py`: deferred behind the IR extension for
  measurement-terminated blocks, as Part X and the Phase-4 handoff
  anticipated; batching delivers the width result without it.

The original plan follows for reference.

## Files (Part IX tree)

| File | Role | Notes |
|---|---|---|
| `qreuse_batch.py` | lay B independent RF-QRC step circuits side by side, terminal measurements only, feed through the UNTOUCHED stage-5 compiler | parser-legal today per Part X sec. 2.7; extension of the stage-5 pipeline, compiler modules not modified |
| `exp_rfqrc_reuse.py` | batch-and-compress validation: self-calibrated TVD gate; compression vs entangler topology incl. all-to-all negative control; feature-interchangeability check | headline structural claim: compiled width independent of memory horizon (vs windowed protocol's measured physical = T+1) |
| `exp_vn_fusion_proto.py` | virtual-node fusion prototype: N_v truncated circuits joined by measure-and-reset boundaries on one register | BLOCKED behind an IR extension for measurement-terminated blocks; validated blockwise against unfused truncation; deferred if the IR work balloons - batching delivers without it |
| `solar_ramps_data.py` | SURFRAD 1-min (aggregated to 5-min) primary, NSRDB 30-min secondary; full registry vetting; clear-sky index k_t fit on train span only; mask, never interpolate | ramp label: abs(k_t(t+h) - k_t(t)) >= rho, rho = train-span 95th percentile; horizons 30 min - 4 h |
| `exp_solar_ramps.py` | direct multi-horizon multitask read-outs; multichannel encoding across qubit groups; hybrid posture with quantum-only and classical-only ablation columns | event-level metrics PRIMARY (one-step point forecasting on solar is a known four-way tie with persistence) |

## Pre-registered success criterion (frozen, Part X sec. 6)

Solar: at h = 1 and 2 hours, hybrid F1 on ramp events >= classical-only
ablation + 0.03 absolute; quantum-only column reported; secondary NMSE on
k_t with mean line and persistence rows present. Honesty check: RF-QRC vs
recurrent ring on ENSO, expected recurrence WIN declared in advance.

## Shot study (Phase 6c)

Gated on the analytic-ceiling check; cost axis (shots and wallclock per
prediction) reported next to accuracy.
