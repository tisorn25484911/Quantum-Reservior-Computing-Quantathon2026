# TODO — CP Aquaculture Use Case

Next steps, ordered. Read `presentation/README.md` first for the full state.
Checkboxes reflect status as of **2026-07-23**.

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

## In progress ⏳

- [ ] **Finish the Gulf of Thailand daily SST download.** Resume with:
      `cd Quantathon_stack/Data && ../../.venv/bin/python fetch_cp_data.py`
      (picks up from cached year-chunks; assembles
      `real/oisst_gulf_thailand_daily_sst.csv` when done). Then confirm:
      `../../.venv/bin/python -c "from DataBase_Analysis.dataloader import load; print(load('got_sst'))"`
      (or run from the `DataBase_Analysis` dir).

## Next — turn generality into a real Gulf result 🎯

- [ ] **Run the Step-2 skill-decay kill-test on `got_sst`** once it's downloaded:
      `cd Quantathon_stack/Anomaly_Forecast && ../../.venv/bin/python run_step2.py --dataset got_sst`
      This produces the QRC-vs-ESN-vs-persistence NMSE curve on CP's own waters.
      `run_step2.py` accepts any `dataloader` key via `--dataset` (no code change
      needed). Two flags to consider: `--max-points` (default 2000; the OISST record
      is ~16000 days, so raise it to use more history) and note that `got_sst` is
      *raw* SST, so the **climatology baseline is intact here** (unlike `nino34`,
      which is already an anomaly series — see `plan.md` §6 open-item 5).
      **Run it multi-seed** (the ESN must be a distribution, not one draw — see
      `Anomaly_Forecast/plan.md` §10).
- [ ] **If Gulf SST shows skill**, add a real result panel to `deck.html` slide 6
      (replace the "acquired / next" framing for `got_sst` with the measured curve).
      Keep the honesty invariant: report persistence + ESN on the same axis.
- [ ] **If it does not beat the baselines**, say so — pick the strongest honest
      framing (e.g. determinism/reliability) and update the deck + notes. Do not
      overclaim.

## Then — Step 3 of the Forecast-then-Detect plan 🔬

Per `Anomaly_Forecast/plan.md` §3 (the next kill-test in the original line):

- [ ] **Stochastic rollout**: sample K≈200 futures (residual bootstrap), verify the
      ensemble is **calibrated** (actual value inside the 90% band ≈90% of the time,
      per horizon). If over-confident, every downstream alarm probability is wrong —
      fix before building on it.
- [ ] Adjust the deck's "calibrated probability" claim to whatever the coverage test
      actually shows.

## Product-facing follow-ups (lower priority)

- [ ] **Rain → pond-salinity model** for `maeklong_rain` (currently ingested as a
      driver only; the salinity link is a Next-phase item — say so in the pitch).
- [ ] **EDA figures for the four CP datasets** via `DataBase_Analysis/analysis.py`
      (spectra, seasonality) — useful backup slides.
- [ ] **Webapp integration**: expose `got_sst` on the `/forecast` and `/anomaly`
      pages so the demo runs on Gulf-of-Thailand data live.

---

## Housekeeping before pushing to GitHub

- [ ] Gitignore the intermediate cache and machine-local settings:
      `printf '_oisst_chunks/\n.claude/settings.local.json\n' >> .gitignore`
      then `git rm --cached .claude/settings.local.json`.
- [ ] Commit the recipe + completed CSVs + `presentation/` now; commit
      `oisst_gulf_thailand_daily_sst.csv` separately once its download finishes.

---

## Guardrails (do not violate — this project's whole credibility rests on them)

- **No quantum-advantage claim** at 8 qubits / exact simulation. Parity-or-better
  vs a *size-matched* classical baseline is the ceiling of what may be claimed.
- **Every accuracy number ships with persistence + size-matched ESN** on the same
  axis, over **multiple seeds** (the ESN is a random draw; `xxz_hx` is deterministic).
- **Gulf-of-Thailand forecast results are NOT done** until `run_step2.py --dataset
  got_sst` has been run and reviewed. Until then the deck says "acquired / next".
