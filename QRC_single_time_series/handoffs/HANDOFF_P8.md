# HANDOFF_P8 — Controlled-system campaign + dynamical audits

Written per G10. In-repo (declared deviation, as P0–P7).

## 1. Objective & status vs exit criteria

| Exit criterion | Status | Evidence |
|---|---|---|
| Lorenz β=8/3 vs printed 3/8 audited (G8-1) | ✅ | λ 0.898 vs 0.0007 → β=8/3; `lorenz63.beta_audit` |
| Vallis ENSO ODE audited, λ computed in-repo (G8-2/3) | ✅ (partial) | λ≈0.26–0.28, chaotic; `enso_ode.audit` |
| Benettin ↔ Rosenstein cross-validated on Lorenz | ✅ | both >0 (Benettin 0.90, Rosenstein ~1.24) |
| RK4 integrator + continuous Benettin spectrum | ✅ | `dynamical_systems/integration.py` |
| Multi-origin autonomous campaign, horizons in steps AND LT | ✅ | `run_autonomous_forecasts.py` |
| Dynamical-fidelity suite (invariant measure/spectrum/ACF/recurrence) | ✅ | `evaluation/dynamical_fidelity.py` |
| Horizon-vs-shots slope vs b≈1/(2λΔt), with SNR precondition | ✅ (negative) | slope not measurable; flagged (§4) |
| Suite passes | ✅ | **218 passed, 1 skipped** (~42 s) |

## 2. Artifacts & keys

- `dynamical_systems/integration.py` — `rk4_step`, `integrate`, `benettin_ode`
  (co-integrated tangent frame + QR), `benettin_largest_ode`.
- `dynamical_systems/lorenz63.py` — `field`/`jacobian`/`generate`/`lyapunov_largest`/
  `beta_audit`. `BETA_STANDARD=8/3`, `BETA_PRINTED=3/8`.
- `dynamical_systems/enso_ode.py` — Vallis box model `[u,Te,Tw]`, `PARAMS` (Hamhoum
  Eq. 13), `field`/`jacobian`/`generate`/`lyapunov_largest`/`audit`.
- `evaluation/dynamical_fidelity.py` — `autocorrelation`, `power_spectrum`,
  `invariant_measure_distance`, `recurrence_rate`, `spectral_distance`,
  `acf_distance`, `fidelity_report`.
- Scripts: `estimate_lyapunov_spectra.py` (`make lyapunov-audit`) →
  `results/metrics/lyapunov_spectra.json`; `run_autonomous_forecasts.py`
  (`make grid-autonomous`) → `results/metrics/autonomous_campaign.json`;
  `analyze_dynamical_fidelity.py` (`make dynamical-fidelity`) →
  `results/dynamical_fidelity/summary.json`.
- Tests: `tests/test_dynamical_systems.py` (7).

## 3. Key results

**Lyapunov audits (per unit time).** Lorenz β=8/3 → **0.898** (≈0.906 ✓, chaotic);
β=3/8 → **0.0007** (non-chaotic). Vallis ENSO ODE → **≈0.26–0.28** (LT≈3.6–3.8),
bounded & chaotic, **computed in-repo — the 0.05–0.1 figure is never reused**.
Mackey-Glass perturbation-pair: τ16≈0.0008 (~0), τ17≈0.0091 (>0), signs correct.

**Autonomous campaign (stateful FN, N=5, V=6; median over 8 origins).**

| system | λ/step | LT (steps) | H_error | H_skill(persist) | H_effective | H_eff (LT) | invmeas L1 | std_ratio |
|---|---|---|---|---|---|---|---|---|
| MG τ17 | 0.0079 | 127 | 50 | 120 | 50 | 0.39 | 0.36 | 1.14 |
| Lorenz-x | 0.0186 | 54 | 32 | 73 | 32 | 0.60 | 0.52 | 0.97 |
| Vallis-Te | 0.0034 | 297 | 38 | 40 | 28 | 0.10 | 0.55 | 2.07 |

Horizons are **sub-Lyapunov-time** for this deliberately small, un-swept N=5 QRC —
consistent with the soundness review's warning that a well-tuned reservoir reaches
"a few LT" and that an apparent 20-LT horizon would be a leak. Pointwise valid time
is separated from attractor fidelity (MG τ17 stays "attractor-faithful" by the
invariant-measure/std-ratio test even after pointwise tracking is lost).

## 4. Decisions & deviations

- **Vallis sign audit is numerical, not text-level.** The primary Vallis 1986/1988
  PDFs are scanned images (WebFetch returned only binary/JBIG2), so the box-model
  sign convention is the one reproduced in the secondary literature, **adopted
  because it reproduces the claimed chaotic behaviour** (λ>0, bounded). Recorded in
  the errata register (G8-3) with the caveat; flagged for a text-level re-check.
- **Shot-slope H≈a+b·lnS is NOT measurable for the N=5 config** — emulated finite
  shots collapse the closed loop to H_error=0 at every tested count {128…8192}
  while the exact loop lasts ~46 steps. Reported honestly (`slope_measurable=False`,
  with a note) rather than forcing `b_fit=0` to look like agreement; the real
  shot-slope study belongs in P10 with a stronger-signal config **and** the
  SVD-truncation mitigation arm.
- **Systems reduced to univariate** (MG=series, Lorenz=x, Vallis=Te) for the
  same-variable autonomous interface; multivariate rewind is out of P8 scope.
- Campaign sizes (origins=8, horizon≈140, N=5) are modest so `grid-autonomous`
  finishes in minutes here; they are config knobs — widen for the full budget.

## 5. Open issues / known gaps

- Text-level Vallis sign re-check pending a machine-readable primary source.
- Shot-slope requires a stronger-signal reservoir config (P10).
- Rewind reservoir not yet run in the autonomous campaign (FN only); add the rewind
  arm in P9/P10 where its hardware shape matters.
- Rosenstein overestimates λ on Lorenz (1.24 vs 0.90) — expected estimator bias;
  only the sign/order is used as the precondition for learned-map λ.

## 6. Entry instructions for Phase 9 (observational climate campaign)

1. `make test` green (218 passed, 1 skipped).
2. Real indices only (enso/pdo/soi), NEVER conflated with the Vallis ODE (item 21).
   Full EDA per spec §14 (ACF/PACF, spectra, ADF/KPSS, seasonal decomposition,
   regimes); anomaly/deseasonalise/detrend as ablation arms, climatology from train
   years only (`data/preprocessing.py` already does train-only).
3. Teacher-forced one-step + direct h∈{3,6,12} sharing one feature cache;
   autonomous 20–50 origins (record permitting), full H_* suite, failure labels,
   §18.2 distributional/spectral/ACF/regime fidelity, launch-month skill (spring
   barrier), hybrid-readout ablation on ≥ENSO. Report in normalised/index/steps/
   calendar-months. Evaluate the **preregistered Gate-1** on the untouched test span.
4. `scripts/run_one_step_forecasts.py` + reuse `run_autonomous_forecasts.py`
   machinery; write `HANDOFF_P9.md`.

## 7. Suggested skills

`quantum-reservoir-computing` + `qrc-project-playbook` (min per G10); add
`citation-audit` when Jaeger–Haas 2004 / Pathak 2018 (LT convention) or Gauthier
2021 enter `references.bib`.
