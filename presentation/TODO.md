# TODO — CP Aquaculture Use Case

Next steps, ordered. Read `presentation/README.md` first for the full state.
Checkboxes reflect status as of **2026-07-23** (Gulf-SST result added same day,
after a `new_data/` raw-archive drop landed — see below).

---

## Done ✅

- [x] Choose the product use case (marine anomaly forecasting for Gulf-of-Thailand
      aquaculture — "TideRead") and the reasoning behind it.
- [x] Write `Quantathon_stack/Data/fetch_cp_data.py` (reproducible CP-data recipe).
- [x] Register `got_sst`, `got_ersst`, `oni`, `maeklong_rain` in `dataloader.py`.
- [x] Download ERSSTv5 reconstruction, ONI, and Mae Klong rainfall; verify they
      load through `dataloader.load()`.
- [x] Build `presentation/deck.html` (8-slide pitch, published as an Artifact).
- [x] Write `presentation/speaker_notes.md` (talking points + Q&A + numbers).
- [x] **Finish the Gulf of Thailand daily SST record.** A `new_data/oisst_v2.1/`
      raw archive (46 yearly OISST v2.1 NetCDFs, same source) showed up locally;
      `fetch_cp_data.py` now extracts the Gulf point directly from it instead of
      the slow ERDDAP year-by-year fetch. `got_sst` is complete: 1981-09→2026-07,
      16,394 days, 0% missing. The ERDDAP path is kept as the fallback for a
      clone without `new_data/`.
- [x] **Run the Step-2 skill-decay kill-test on `got_sst`, multi-seed.** Full
      record, H=24, seeds 7/11/23/42/101 (`Anomaly_Forecast/seed_sweep.py`,
      written because the winning kind here — `ising` — is itself seed-dependent,
      unlike TAO's deterministic `xxz_hx`, so a single seed isn't valid).
      **Result: PASS**, and it does beat the baselines, with real caveats:
      - Skill (NMSE<1) at all 24 lead days.
      - Materially beats persistence at day 1, then continuously day 12→24
        (persistence is hard at short lead because SST barely moves day to day,
        then its error compounds linearly while the reservoir's does not).
      - Never materially beaten by the size-matched ESN at any horizon; beats its
        mean curve at 16/24 horizons. But the **per-seed win rate is only ~3/5**
        at a typical horizon (`ising`'s own seed variance is real, unlike the
        deterministic TAO case) — report this, not just the 16/24 headline.
      - `climatology` (period=365, seasonal-mean) is a real, non-degenerate floor
        here (unlike `nino34`) — the reservoir clears it only through ~day 20;
        by day 21-24 all three models (reservoir, ESN, persistence) converge to
        it. **The honest usable lead is ~20 days, not 24.**
      - Full artifacts: `Anomaly_Forecast/results/step2_seedsweep_got_sst.json`,
        `step2_got_sst.json` / `step2_skill_got_sst.png` (single-seed=7 view).
- [x] **Added a real result panel to `deck.html` slide 6** replacing the
      "acquired / next" framing for `got_sst` with the measured result above
      (badge changed `quantum-ready` → `measured`); `speaker_notes.md` slide 6
      section and Q&A updated to match, including the two honest caveats
      (seed-dependent kind, climatology floor).
- [x] **Integrated the rest of the `new_data/` drop where it fit the use case:**
      `got_hadisst` (HadISST1, independent cross-check on the Gulf heat signal)
      and `maeklong_spei` (SPEI-03 drought index, same point as `maeklong_rain`)
      registered in `dataloader.py` and added to the slide-6 driver table.
      `mrc_discharge/` was deliberately **not** wired in — every station in it is
      on the Mun River / Mekong mainstem (drains to the South China Sea, not the
      Gulf of Thailand) — see `new_data/README.md` for the reasoning. Extra
      `enso_indices/` files were left unregistered as duplicative of `nino34`/`oni`.
- [x] Gitignored `new_data/` (was untracked but not ignored — ~21GB of raw
      archives; fixed with an exception so `new_data/README.md` stays tracked).

## Then — Step 3 of the Forecast-then-Detect plan 🔬

Per `Anomaly_Forecast/plan.md` §3 (the next kill-test in the original line):

- [ ] **Stochastic rollout**: sample K≈200 futures (residual bootstrap), verify the
      ensemble is **calibrated** (actual value inside the 90% band ≈90% of the time,
      per horizon). If over-confident, every downstream alarm probability is wrong —
      fix before building on it.
- [ ] Adjust the deck's "calibrated probability" claim to whatever the coverage test
      actually shows.

## Drought target (SPEI) 🌵

- [x] **First SPEI drought forecast run.** Multi-seed kill-test on
      `maeklong_spei` (`seed_sweep.py --dataset maeklong_spei`, H=12, 5 seeds,
      115 yr). **Result: real skill at h=1–2 months** (NMSE 0.43/0.78), beats
      persistence at all 12 horizons, beats the size-matched ESN at 11/12
      (winning kind is the deterministic `xxz_hx` — a memory-dominated task, so
      the integrable chain wins, consistent with the QRC paper). Usable lead ~2
      months. Artifacts: `results/step2_seedsweep_maeklong_spei.json`,
      `results/step2_maeklong_spei.json`, `results/drought_spei.png`.
- [x] **Quantified the ENSO→drought driver** (`drought_spei.py`): El Niño → Mae
      Klong drought, correct sign but **modest** (peak corr −0.21 at 0–3 mo lead).
- [x] **Deck slide added** (slide 6b) + speaker notes + Q&A, scoping drought
      honestly: SPEI as the applied target with a real short-range first forecast,
      SST as the validated driver, short lead + weak teleconnection both stated.
- [ ] **NEXT: SST-conditioned (multivariate) SPEI model.** The current run is
      univariate (SPEI from its own past). Fold the validated ocean signal
      (ONI/`got_sst`/`got_ersst`) in as an exogenous driver to try to extend the
      drought lead. Needs the reservoir to accept an auxiliary input channel
      (currently single-scalar) — a real code change, scoped as the next step.

## Product-facing follow-ups (lower priority)

- [ ] **Rain → pond-salinity model** for `maeklong_rain` (currently ingested as a
      driver only; the salinity link is a Next-phase item — say so in the pitch).
- [ ] **EDA figures for the four CP datasets** via `DataBase_Analysis/analysis.py`
      (spectra, seasonality) — useful backup slides.
- [ ] **Webapp integration**: expose `got_sst` on the `/forecast` and `/anomaly`
      pages so the demo runs on Gulf-of-Thailand data live.

---

## Housekeeping before pushing to GitHub

- [x] Gitignore the intermediate cache, machine-local settings, and the
      manually-placed `new_data/` archive (~21GB; kept `new_data/README.md`
      tracked via a `!` exception). `.claude/settings.local.json` was never
      tracked, so no `git rm --cached` was needed.
- [x] Commit the recipe + completed CSVs (including
      `oisst_gulf_thailand_daily_sst.csv`, `hadisst_gulf_thailand_monthly_sst.csv`,
      `spei03_mae_klong_monthly.csv`) + `presentation/` + `Anomaly_Forecast/seed_sweep.py`
      + the new `results/step2_seedsweep_got_sst.json` and its sibling artifacts.
      Commit `d28407a` on `main`. Not yet pushed to the remote.

---

## Guardrails (do not violate — this project's whole credibility rests on them)

- **No quantum-advantage claim** at 8 qubits / exact simulation. Parity-or-better
  vs a *size-matched* classical baseline is the ceiling of what may be claimed.
- **Every accuracy number ships with persistence + size-matched ESN** on the same
  axis, over **multiple seeds** (the ESN is always a random draw; the quantum
  kind is only deterministic when it's `xxz_hx` — on `got_sst` the winning kind
  is `ising`, which is seed-dependent too, so it gets swept exactly like the ESN).
- **Gulf-of-Thailand forecast results are done**, multi-seed, full record — see
  "Done" above and `deck.html` slide 6. Any future re-run of `got_sst` (new
  history added, different H, different qubit count) must go back through
  `seed_sweep.py`, not a single `run_step2.py` seed, before the deck number
  changes — the whole point of §10's lesson was that one seed reversed a verdict.
