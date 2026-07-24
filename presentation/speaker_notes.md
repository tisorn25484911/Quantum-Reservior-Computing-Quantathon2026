# Speaker notes & Q&A — TideRead / QRC anomaly forecasting

Companion to `deck.html`. Eight slides, ~6–8 min at pitch pace. Each slide below
has: **the one thing to land**, talking points, and the numbers that are safe to
say out loud. A hard rule runs through all of it — **never claim quantum
advantage we can't defend.** The honesty is the pitch.

> **The through-line, in one sentence:** we move anomaly detection from
> *"tell me it's happening"* to *"tell me it's coming," and on daily ocean
> temperature — the cadence a farm acts on — our quantum reservoir holds forecast
> skill longer and more reliably than a matched classical model.*

---

## Slide 1 — Title

**Land:** this is an *early-warning* product, not a science demo.

- Open with the stake in one breath: "An aquaculture pond is a bet on stable
  water. When the ocean gets too warm, or rain crashes the salinity, that bet can
  fail overnight — and today the farm finds out only once it's already happening."
- "We built a forecasting engine, powered by a quantum reservoir, that sees the
  anomaly coming days ahead."
- Name the partner frame: CP-scale shrimp/marine aquaculture, Gulf of Thailand.
  We're not claiming a signed deal — this is the *use case we designed against*.

## Slide 2 — The stake

**Land:** the cost of being late is a whole grow-out cycle; the cost of being
early is running an aerator.

- The asymmetry is the entire business case. Acting early is cheap (aeration,
  a feed adjustment, an earlier partial harvest). Acting late is a write-off.
- The four numbers on screen: **0 hrs** warning from detection today; **1** cycle
  is all it takes to lose; **≈1 week** lead we demonstrate on daily SST; **3**
  drivers we ingest (heat, ENSO, rain).
- Don't over-quantify losses in dollars — we don't have CP's loss book yet, and
  inventing a figure is exactly the kind of overclaim we're avoiding. Say "a bad
  event can end a cycle" and leave the number to the pilot.

## Slide 3 — The shift

**Land:** we didn't invent a new detector; we moved a trusted one earlier in time.

- Detection needs the observation to *exist* — so structurally it can only flag
  what's already occurring. Zero warning is not a tuning problem, it's the
  definition.
- Our move: run the same detection science on a *forecast* trajectory. Warning
  time becomes the forecast horizon.
- This is deliberately un-magical. Reviewers trust it *because* the detector is
  the same one the field already uses. The novelty is the quantum forecast that
  feeds it.

## Slide 4 — How it works

**Land:** one clean pipeline; the quantum core is fixed and reproducible.

- Walk the four nodes left to right: observe → quantum reservoir rollout (feeds
  its own prediction back, K sampled futures) → detectors on each future →
  alarm + ETA + which driver is loading.
- The key technical selling point: **the reservoir is not trained** — it's a fixed
  8-qubit dynamical system with a closed-form linear readout on top. Same input →
  same alarm, every time. For anything operational, that determinism is a
  feature.
- K≈200 sampled futures → a *calibrated probability*, not a single point guess.
  (Why sampling matters is on slide 7 if they push.)

## Slide 5 — Proof (the money chart)

**Land:** on daily ocean temperature, the quantum reservoir keeps skill after the
classical baseline has lost it — and it does so at *every* random seed.

- Read the chart out loud: "lower is better; 1.0 is no-skill. The classical ESN
  (orange) climbs *past* the no-skill line by long lead — worse than guessing the
  average. The quantum reservoir (teal) is still at 0.59."
- The seed point is the one that survives scrutiny: **5 of 5 random seeds**, the
  quantum reservoir wins across the horizon. The classical baseline is also
  *unstable* on this daily series — one seed blows up to NMSE 1.83. Determinism
  removes that failure mode.
- Why TAO and not Gulf of Thailand here: TAO is the closest long, clean, **daily**
  ocean-temperature record — 452 held-out origins. It's the honest proving ground;
  the Gulf data (slide 6) is where we take the *validated* method next.
- **The baseline is not a straw man** — say this explicitly. Same feature count,
  tuned on the same data. Beating a weak baseline proves nothing and we know it.

## Slide 6 — Generality (CP's own waters)

**Land:** the identical pipeline already ingests the real drivers of a Gulf of
Thailand pond — and now has a measured forecast result on the customer's own
heat record, not just acquired data.

- The six records, and *why each one earns its place*:
  - **Gulf of Thailand daily SST (OISST, 0.25°, off the Mae Klong mouth)** — the
    heat signal itself, at the location and cadence that matters. Daily,
    1981-09→2026-07, 16,394 days, the full record.
  - **HadISST1 monthly, 1870→2026** — an independent cross-check on that same
    heat signal (different interpolation/sea-ice treatment than ERSSTv5); a
    warm-event signature showing up in both is not a reconstruction artifact.
  - **ERSSTv5 monthly reconstruction, 1854→2026 (172 years)** — context: is this
    warm event unusual against a century and a half? Warm-pool SST here runs
    26–31 °C.
  - **Oceanic Niño Index, 1950→2026** — the ENSO state that modulates everything;
    ranges −2.0 (La Niña) to +2.8 (strong El Niño).
  - **Mae Klong basin daily rainfall (NASA POWER), 1981→2026** — the freshwater
    shock that drives pond salinity; 0–109 mm/day.
  - **Mae Klong SPEI-03 drought index, 1901→2015** — folds evapotranspiration
    into the rain signal; a sharper drought/salinity-risk number, though this
    vintage stops in 2015 and needs a bias-corrected join to bring current.
- **The Gulf SST result, stated carefully:** on the full 45-year record, 5 seeds,
  24-day horizon — the same kill-test as slide 5 — the reservoir has real skill
  at every lead day, materially beats persistence from day 12 on, and is never
  materially beaten by the size-matched ESN (beats its mean curve at 16 of 24
  horizons). **Two honest differences from the slide-5 TAO result, say both if
  asked:** (1) the reservoir kind that wins here is the *disordered* one, which
  is itself seed-dependent — so we score it as a distribution too, and its
  edge over the ESN holds at only 3 of 5 seeds at a typical horizon, not 5 of 5;
  (2) a real seasonal-mean baseline (climatology) is available on raw SST — the
  reservoir clears it out to about day 20, then dips back under it. So the
  honest usable lead is **~2-3 weeks**, not "day 24."
- One line: "the engine doesn't care whether the buoy sits on the equator or off
  Samut Songkhram — and now we've checked, not just claimed it."

## Slide 6b — Drought (the applied target)

**Land:** drought is a *different* target from ocean heat, and we ran a real
first forecast of it rather than hand-waving — with the scope stated honestly.

- The distinction to make out loud: slides 5–6 prove the engine on **sea-surface
  temperature**. **Drought is not a temperature** — it's a rainfall/soil-moisture
  deficit index (SPEI). Conflating them would be the dishonest move; we don't.
- What we actually did: ran the **same 5-seed kill-test on the Mae Klong SPEI-03
  drought index itself**, 115 years of monthly data. Result: **real skill at 1–2
  months** (NMSE 0.43 at h=1, 0.78 at h=2 — below the no-skill line), it **beats
  persistence at all 12 horizons** (SPEI decorrelates by ~3 months, so repeating
  last month is a bad forecast), and it **beats the size-matched ESN at 11/12
  horizons**. Note the winning kind here is `xxz_hx`, which is *deterministic* —
  so unlike the seed-dependent SST result, that ESN edge is not a lucky draw.
- Why `xxz_hx` wins here and `ising` won on SST: consistent with the QRC paper —
  SPEI forecasting is a **memory-dominated** task, and the integrable chain has
  high memory (but poor nonlinearity). Different task, different best reservoir;
  we report what actually wins.
- The driver story, stated at its true strength: **SST/ENSO leads the monsoon
  that drives this drought** — El Niño loads Mae Klong toward drought, the
  physically correct sign — but the correlation is **modest (~0.2)**, peaking at
  0–3 months lead. So we present ENSO as a *real but modest* driver, and the
  **SST-conditioned (multivariate) SPEI model as the explicit next step**, not as
  a finished result.
- The honesty line to say plainly: "The lead is short — one to two months — and
  we forecast the drought index from its own past here. The ocean driver we've
  already validated is how we extend it. That's the same rain→salinity caveat,
  applied to drought." Backup figure: `Anomaly_Forecast/results/drought_spei.png`.

## Slide 7 — Honesty as a feature

**Land:** every claim ships with the baseline that could beat it — and sometimes
it does, and we say so.

- This is the trust close. "Quantum ML is full of results that vanish under a fair
  classical control. Ours is built so that when the baseline wins, the code says
  so by default."
- Concede the ENSO case *proactively*: on monthly ENSO data the classical model is
  at parity or ahead. We report it. On daily SST — the operational cadence — the
  quantum reservoir is the more reliable model. Knowing exactly where the edge
  lives is what makes it deployable.
- Three guarantees: no leakage (train-span-only fits, structural); calibrated
  probabilities (measured coverage); cost reported next to accuracy.

## Slide 8 — Product & ask

**Land:** an early-warning layer that plugs into decisions that are cheap early
and expensive late; the ask is a pilot with real loss history.

- What the operator sees: per-site, per-week probability + days-to-onset + driver.
- What it feeds: aeration, feed/stocking, harvest timing, insurance triggers.
- Roadmap: Now (validated core + CP data) → Next (calibrated Gulf-SST alarm +
  salinity-from-rain) → Pilot (one farm cluster, back-tested on its own losses).
- **The ask:** a pilot site with pond-level outcome history — the one label no
  public dataset can give us. That's what turns "the alarm fired" into "the alarm
  saved a cycle."

---

## Anticipated Q&A

**Q. Where's the quantum advantage? This is an 8-qubit simulation.**
Straight answer: at 8 qubits on a classical simulator there is *no speed-up to
claim, and we don't claim one*. What we show is that a small quantum reservoir is
a **more reliable forecaster on daily SST than a size-matched classical reservoir**
— parity-or-better accuracy with strictly deterministic behaviour, where the
classical model is seed-unstable. That's a defensible reliability argument today;
the scaling/hardware speed-up is a separate, future question we're not front-
running.

**Q. Why should a quantum reservoir beat a classical one at all?**
We're careful here — on monthly ENSO it doesn't. On daily SST it holds skill
longer and, crucially, is deterministic where the classical ESN swings from good
to worse-than-mean across random seeds. We present it as *demonstrated on this
data*, not as a theorem.

**Q. Why TAO SST for the proof instead of the Gulf of Thailand data?**
TAO is the longest clean *daily* ocean-temperature record with enough held-out
origins (452) to make the skill-decay curve trustworthy. It's the same physical
variable and cadence as the Gulf SST we've acquired. Proving on the clean series,
then transferring the validated method to the customer's waters, is the honest
order — not cherry-picking the series that flatters us.

**Q. Do you have forecast results on the Gulf of Thailand data now?**
Yes — full 45-year daily record, 5 seeds, same kill-test as the TAO proof. Real
skill at every lead out to 24 days, materially beats persistence from day 12 on,
never materially beaten by the size-matched ESN. Two things we say unprompted:
the winning reservoir kind here is seed-dependent (unlike TAO's deterministic
one), and its edge over the ESN holds at 3 of 5 seeds, not 5 of 5 — a real but
more modest edge than the proof slide. And the honest usable lead is nearer
20 days than 24 — a real seasonal-mean baseline exists on raw SST, and every
model, reservoir included, converges to it by the far end of the horizon.

**Q. Why is the Gulf-SST quantum kind different from the TAO one?**
`ising` (disordered couplings) won on Gulf SST; `xxz_hx` (deterministic) won on
TAO. We didn't pick whichever looks best after the fact and call it "the"
reservoir — both are in the evaluation harness, both get run, and this run just
found a different winner on a different series. It's the same discipline as the
ESN seed sweep: score what actually wins, as a distribution, not a single
flattering draw.

**Q. How is the anomaly probability actually calibrated?**
The ridge readout minimises squared error, so its *mean* trajectory is smooth and
would systematically under-alarm. We fix that by sampling K≈200 futures (residual
bootstrap) and reporting the fraction that trip the detector — then we *measure*
whether the 80% band actually contains the truth 80% of the time. If coverage is
off, the probability is wrong and we don't ship it.

**Q. What's the false-alarm rate?**
Set by an extreme-value threshold fit on forecast-of-training scores, with an
empirical-quantile fallback when the tail is thin. We target the realised
false-alarm rate landing within ~2× of the nominal budget on held-out data — and
it's a number we report, not hide.

**Q. You're pitching aquaculture heat but the slide says drought — which is it?**
Both, honestly scoped. The *validated proof* is ocean heat (SST), which drives
both marine-heat pond stress and — via the monsoon — regional drought. Drought is
a distinct **applied target**: we ran a first forecast of the Mae Klong SPEI-03
drought index and got real 1–2 month skill that beats persistence and the ESN
over 5 seeds. We're not claiming a long-range drought forecaster; we're showing
the same pipeline already produces a skillful short-range one, with SST as the
validated driver to extend it.

**Q. How strong is the ENSO → drought link you're leaning on?**
Modest and we say so: peak correlation ~0.2 with the right sign (El Niño → Mae
Klong drought), strongest at 0–3 months lead. That's a real teleconnection but a
weak single predictor — which is exactly why the honest framing is "SST is *a*
driver we fold in," not "ENSO predicts drought." The measured lead-lag is on the
record (`drought_spei.py`).

**Q. Why does a different quantum reservoir win on drought than on SST?**
`xxz_hx` (a clean chain) wins on SPEI; `ising` (disordered) won on daily SST.
Per the founding QRC paper this is expected: the integrable chain has high
*memory* but poor *nonlinearity*, and SPEI forecasting is memory-dominated. We
run both and report whichever actually wins — same discipline as the ESN seed
sweep.

**Q. Isn't rainfall → salinity a big modelling leap?**
Yes, and we scope it honestly: rain is the *driver* we ingest; the rain-to-pond-
salinity model is a Next-phase item, best fit against a pilot farm's own salinity
logs. We're not claiming a finished salinity forecaster today.

**Q. What does this cost to run per prediction?**
Cheap — exact simulation of 8 qubits plus a closed-form linear solve; it runs on a
laptop. We put cost on the same slide as accuracy on purpose: a model that needs a
supercomputer to match a laptop has already answered the business question.

**Q. Why not just use a big deep-learning forecaster?**
Could be a strong baseline and we'd welcome it on the chart. Our pitch isn't
"quantum beats everything" — it's a reproducible, deterministic, low-cost forecast
core with an honest evaluation harness, that already beats the standard reservoir
baseline on the operational cadence. The evaluation discipline transfers to any
model you drop in.

**Q. What do you actually need from us (CP) to go further?**
One pilot site with pond-level outcome history — losses, emergency interventions,
harvest timing. Public data gives us the ocean; only you have the *labels* that
tell us whether an alarm was worth acting on.

---

## Numbers cheat-sheet (all reproducible)

| Claim | Number | Source |
|---|---|---|
| QRC vs classical ESN, daily SST | wins 5/5 seeds across horizon | `run_step2.py --dataset tao`, seeds 7/11/23/42/101 |
| QRC error at 24-day lead | NMSE 0.59 | `results/step2_tao.json` |
| Classical ESN at 24-day lead | NMSE 1.16 (past no-skill) | same |
| Classical ESN instability | one seed → NMSE 1.83 | seed sweep, plan.md §10 |
| Held-out forecast origins (TAO) | 452 | `step2_tao.json` |
| 1-step forecast NMSE (xxz) | 0.0535, matches notebook's 0.0496 ballpark | Step-1 gate |
| ENSO honesty (concede) | classical at parity/ahead on nino34 | `step2_nino34.json`, seed sweep |
| Gulf SST record | daily, 1981-09→2026-07, 16394 d, full record used | `fetch_cp_data.py` OISST (local `new_data/` archive) |
| Gulf SST cross-check | monthly, 1870→2026, 1877 mo | HadISST1 |
| Gulf SST reconstruction | monthly, 1854→2026, 2070 mo, 26–31 °C | ERSSTv5 |
| ONI record | 1950→2026, range −2.0…+2.8 | CPC ONI |
| Mae Klong rainfall | daily, 1981→2026, 0–109 mm/day | NASA POWER |
| Mae Klong SPEI-03 | monthly, 1901→2015 (vintage stops there) | SPEIbase |
| **SPEI drought forecast** | skill h=1–2 mo (NMSE 0.43/0.78), beats persistence all 12, beats ESN 11/12 | `seed_sweep.py --dataset maeklong_spei`, `step2_seedsweep_maeklong_spei.json` |
| SPEI usable lead | ~2 months (short — stated as such) | same |
| ENSO→SPEI teleconnection | peak corr −0.21, El Niño→drought, 0–3 mo lead (modest) | `drought_spei.py` |
| Gulf SST kill-test | PASS: skill all 24 lead-days, beats persistence day 12-24, ESN never materially wins back | `seed_sweep.py --dataset got_sst`, seeds 7/11/23/42/101, `results/step2_seedsweep_got_sst.json` |
| Gulf SST vs classical ESN | QRC (`ising`) beats mean-ESN curve at 16/24 horizons; per-seed win rate ≈3/5 at a typical horizon | same file |
| Gulf SST honest horizon | ~day 20 (climatology floor catches every model from ~day 21) | same file |

**Discipline reminders:** 8 qubits, exact simulation → *no speed-up claimed*. Every
accuracy number carries persistence + size-matched ESN on the same axis, over
multiple seeds. Gulf-of-Thailand forecast results are now measured (above) —
report the seed-dependence and climatology caveats every time, don't just quote
the 16/24 headline alone.
