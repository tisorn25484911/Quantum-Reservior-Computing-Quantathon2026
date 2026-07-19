# Database Analysis

Characterise every dataset in `Quantathon_stack/Data` before any of it is fed
to a reservoir: what it looks like, what periods it contains, how fast it
forgets its initial condition, and how far ahead it can be predicted even in
principle.

```bash
python analysis.py                       # every dataset present on disk
python analysis.py --dataset nino34      # one
python analysis.py --tier chaotic        # one tier
python analysis.py --method rosenstein   # force single-trajectory (see caveats)
python analysis.py --json out.json       # machine-readable numbers too
python compare.py                        # cross-dataset figure + comparison.md
```

One PNG per dataset lands in `figures/`, plus a summary table on stdout.

## Objective

A quantum reservoir is a *dynamical* model, so the case for or against it rests
on properties of the target series that have nothing to do with the model:
how much memory the task needs, where the deterministic structure lives in
frequency, and where the predictability horizon actually is. This directory
measures those three things, so that later claims about forecast skill can be
read against them rather than in a vacuum.

The concrete question the whole thing exists to answer:

> **For a given target and a given forecast horizon H, is the limit we are
> hitting the model or the dynamics?**

That is decidable. Express H in Lyapunov times. Below `t_λ ≈ 1` an initial
error has not yet grown by a factor of e, so any skill shortfall is the
model's fault and is worth engineering against. Far above it, error growth is
the dynamics and no reservoir — quantum or otherwise — recovers it. Getting
that boundary roughly right is worth more than another decimal place on any
single number.

A secondary objective is calibration. The chaotic tier is the only data here
with a *published* λ₁, so it doubles as the control that says how much to
trust the same estimator applied to a real record. Without it, every λ₁ in
this project would be an unfalsifiable fit.

> **What the calibration returned.** It says the single-trajectory estimator
> cannot do the job: measured against known λ₁ it misses by ~93%, and on data
> with realistic observational noise it cannot see the exponential phase at
> all. **No λ₁ is quoted for any observed record in this directory.** The
> exponents that survive are the chaotic tier's, where realizations exist.
> Section [What the calibration proved](#what-the-calibration-proved) is the
> evidence; [Consequence for the QRC work](#consequence-for-the-qrc-work) is
> what to do instead. An earlier revision of this document quoted λ₁ for all
> ten observed records; those numbers were fits to curves with no scaling
> region and have been withdrawn.

## What gets computed

### 1. The series itself

Plotted in physical time, against real dates where the record has them. The
purpose is not decoration — it is the check that the loader interpreted the
file correctly, that gaps and outages were handled, and that the series looks
like the thing its filename claims.

### 2. Amplitude spectrum, linear in both axes

One-sided FFT, mean removed, linearly detrended, Hann-windowed, and normalised
so a sinusoid of amplitude A appears as a line of height A. Frequency in cycles
per time unit; amplitude in the unit of the series.

Deliberately not dB and not log-log. The questions asked of this data are
"which period dominates, and how large is it in degrees C" — both of which a
log axis makes harder to read. Detrending matters more than it looks: a
non-zero mean dumps all its energy into the f=0 bin and a drift smears power
across the lowest few bins, which is precisely where the climate structure
lives.

Peaks are reported as periods, local maxima only, with the lowest two bins
excluded — on a short record the residual near-DC leakage outranks every real
line.

### 3. Largest Lyapunov exponent

Two estimators, and the difference between them is the main methodological
point of this directory.

**Ensemble** (`lyapunov_ensemble`) — needs R realizations of the same system
started from initial conditions separated by a small known perturbation, all
already on the attractor. Divergence is then measured directly between
genuinely independent trajectories:

```
y(k) = ln sqrt( < (x_i(k) - x_j(k))^2 > )   over pairs (i, j)
```

No delay embedding is involved, so none of the embedding's failure modes are
either. Note log-of-RMS rather than mean-of-logs: a *scalar* observable of two
diverging trajectories oscillates through zero even while the underlying state
separation grows monotonically, and taking logs first turns every one of those
crossings into a large negative excursion that fakes a slope.

**Rosenstein** (`lyapunov_rosenstein`) — single trajectory. Delay τ from the
first minimum of the mutual information, dimension m from false nearest
neighbours, then spatial nearest neighbours inside that embedding *substitute*
for independent perturbations. A Theiler window rejects neighbours that are
merely adjacent in time.

`estimate()` prefers the ensemble whenever realizations exist and labels which
estimator produced the number.

Before any of that, a **rise guard**: unless the divergence curve climbs at
least `MIN_RISE_NATS = 3`, λ₁ is reported as *not measurable* rather than
fitted. Fraction-of-rise windowing is only meaningful on a curve that rises;
on a flat or oscillating one it lands on an arbitrary wiggle and returns its
slope. A pure sine wave — not chaotic, λ₁ = 0 by construction — returned
−0.291 before this check, with R² = 0.918. Neither goodness-of-fit nor the
returned value can substitute for the test: a short arc of a sinusoid is very
nearly straight, and the flat-curve cases returned everything from −2.2 to
+0.63. Only the rise distinguishes them.

λ₁ is then the slope of the divergence curve over its scaling region, located
by **fraction of total rise** (10%–60% by default) rather than by fixed index.
The curve climbs from ln(ic_eps) to ln(attractor diameter); the bottom of that
climb is contaminated by re-orientation onto the unstable manifold and the top
by saturation, so the middle is the exponential stretch. A fixed index window
assumes every system leaves its transient on the same schedule, which is false
— it put Hadley +92% off while Lorenz-63 landed within 10%.

### 4. The series against Lyapunov time

The record re-plotted on `t_λ = λ₁ t`, gridline per e-folding. This is the
panel the objective above cashes out in: one unit is one e-folding of initial
error, so "how many units wide is the horizon I want" is readable straight off
the axis.

```
T_λ = 1 / λ₁
```

Infinite for a non-positive exponent — no exponential error growth means no
predictability horizon set by the dynamics.

## Results

### Chaotic tier — the calibration

Ensemble estimator, R = 8–12 realizations, against published λ₁.

| system | published λ₁ | measured | error | T_λ |
|---|---|---|---|---|
| Lorenz-63 | 0.9056 | 0.8231 | −9.1% | 1.21 |
| Vallis ENSO | 0.5478 | 0.5200 | −5.1% | 1.92 |
| Lorenz-84 | 0.4615 | 0.7499 | +62.5% | 1.33 |
| Hadley | 0.2387 | 0.3759 | +57.5% | 2.66 |
| Rikitake | 0.1318 | 0.1049 | −20.4% | 9.54 |

All in natural time units. Mean absolute error ≈ 31%.

**Read this table before trusting any λ₁ produced here.** The ensemble
estimator is the right tool when realizations exist and it beats Rosenstein on
four of five systems, but it is not a precision instrument. A result like
"T_λ = 5.8 yr" should be read as "order 5 years", never as three significant
figures. The residual error is dominated by where the scaling region is placed,
not by the divergence curve itself — that placement is the open problem in
`lyapunov.py`.

### What the calibration proved

The chaotic tier exists to say how far the estimator can be trusted elsewhere.
Run both estimators on it, where λ₁ is known:

| system | truth | ensemble | err | rosenstein | err |
|---|---|---|---|---|---|
| `lorenz63` | 0.9056 | 0.8231 | −9% | 2.3130 | **+155%** |
| `vallieselnino` | 0.5478 | 0.5200 | −5% | 1.3013 | **+138%** |
| `lorenz84` | 0.4615 | 0.7499 | +63% | 0.7003 | +52% |
| `hadley` | 0.2387 | 0.3759 | +58% | 0.1643 | −31% |
| `rikitake` | 0.1318 | 0.1049 | −20% | 0.2494 | +89% |
| | | **mean \|err\| 31%** | | **mean \|err\| 93%** | |

Rosenstein was averaged over every member of each system; per-member scatter is
only a few percent, so those wrong answers are *reproducible* wrong answers.
Consistency is not accuracy. Under the rise guard, **8 of 44 members** produce
a curve that rises 3+ nats — and all eight are Lorenz-84, where neither method
is good.

That is on clean, noise-free, unambiguously chaotic data with the answer known.

### Why observed records cannot work: the noise floor

Take `vallieselnino` — an ENSO oscillator, λ₁ = 0.5478 known — and degrade it
to the Niño 3.4 record's conditions. If the method cannot recover a known
answer under those conditions, it cannot produce one on real data either.

**Length is not the obstacle.** With a clean ensemble:

| N | ensemble λ₁ | err | rise |
|---|---|---|---|
| 2500 | 0.5200 | −5% | 16.5 |
| 918 | 0.4886 | −11% | 14.4 |
| 600 | 0.4774 | −13% | 11.4 |
| 300 | 2.1400 | +291% | 6.8 |

918 monthly samples is plenty.

**Observational noise is the obstacle**, and the wall is sharp (N = 918):

| noise (% of σ) | ensemble λ₁ | err | rise |
|---|---|---|---|
| 0% | 0.4886 | −11% | 14.4 |
| 1% | 0.0301 | **−94%** | 3.4 |
| 2% | -- | -- | 2.4 |
| 5% | -- | -- | 1.8 |
| 10% | -- | -- | 1.6 |

The reason is structural. The exponential phase lives at separations between
`ic_eps` and the attractor diameter. Noise puts a *floor* under separation —
two trajectories can never appear closer than the noise amplitude — so at 1%
of σ the bottom ~9 nats of the curve are buried, and that is exactly where λ₁
was measurable. Adding a seasonal cycle removes what little remains.

Real SST carries considerably more than 2% observational uncertainty. **This is
an information limit, not a code or sampling limit**: no record length and no
number of members recovers a signal that sits below the noise floor.

### Real tier

Ten observed records, no realizations anywhere. Every one is refused:

| dataset | N | rise (nats) | λ₁ |
|---|---|---|---|
| `brest` sea level | 509 | 1.97 | -- |
| `solar` GHI *(surrogate)* | 17,520 | 1.99 | -- |
| `load` system load *(surrogate)* | 26,280 | 1.88 | -- |
| `lax` LA Tmax | 29,935 | 1.80 | -- |
| `opsd` German load | 50,401 | 1.62 | -- |
| `nyc` NYC Tmax | 57,540 | 1.43 | -- |
| `nino12` raw SST | 732 | 1.37 | -- |
| `potomac` discharge | 35,203 | 1.29 | -- |
| `cuxhaven` sea level | 2,058 | 1.19 | -- |
| `nino34` anomaly | 918 | 1.12 | -- |
| **white noise (control)** | 2,000 | **1.01** | -- |
| `tao` 0N140W SST | 1,684 | 0.93 | -- |
| `potomac15` discharge | 175,284 | 0.79 | -- |
| **sine wave (control)** | 4,000 | **0.00** | -- |

Sorted by rise, with the two controls inserted in place. **Every real and
surrogate record sits in the white-noise band**, and two of them rise *less*
than pure noise. The chaotic tier, for contrast, spans 10.4–16.5 with nothing
between 2 and 10.

`nyc` has 57,540 clean daily samples — the best-conditioned record in the
directory — and still rises only 1.43 nats. Record quality is not the issue.

> **Withdrawn.** An earlier revision read λ₁ = 0.0626 /day for `nyc`, i.e.
> T_λ = 16 days, and argued that its agreement with the textbook ~2-week
> atmospheric predictability limit *validated the pipeline*. It does not. That
> number came from a curve rising 1.43 nats, and the estimator producing it
> misses by ~93% where truth is known. The agreement was a coincidence, and
> reading it as validation was the most misleading claim in this document.
> Similar withdrawals: `opsd` vs `load` "real demand is 2.7× less chaotic",
> the `potomac`/`potomac15` factor-of-14 disagreement, `cuxhaven`'s negative
> exponent, and the `T_λ`/period "credibility" column. All were structure read
> into noise-band curves.

What survives from the real tier is everything that does not depend on λ₁:
record length, sampling, gap structure, and the **spectra**, which are
unaffected by any of this. See [comparison.md](comparison.md).

#### ENSO, in detail

The spectral result stands and is the useful one:

| dataset | dominant periods (amplitude) |
|---|---|
| Niño 3.4 anomaly | 3.64 yr (0.483), 4.78 yr (0.420), 2.47 yr (0.396), 12.75 yr (0.322) |
| Niño 1+2 raw SST | **1.00 yr (2.745)**, 3.59 yr (0.737), 5.08 yr (0.535), 2.18 yr (0.358) |

Power spread across 2.5–5 yr with no single dominant line is the canonical
ENSO recurrence, and a broad band rather than a sharp line is the spectral
signature of irregular rather than periodic behaviour. Niño 1+2 is raw SST, so
its seasonal cycle dominates at four times any other line; Niño 3.4 is an
anomaly with that cycle removed. **Use Niño 3.4.** The 12.75 yr peak is about
six cycles into a 76-year record — under-resolved, do not quote it.

The timescale argument now rests on this band and on the Vallis ENSO
oscillator's measured T_λ = 1.92 natural units, *not* on any measurement from
the observed record.

`tao` is the daily counterpart to those monthly indices, and the only real
record here with sibling series (five moorings). It is not an
initial-condition ensemble — the moorings are dynamically coupled along the
equator — so it cannot drive the ensemble estimator either.

### Surrogate tier

Generated series, so any λ₁ would describe the generator, not a climate. Both
are refused on the same grounds as the real tier (rise 1.99 and 1.88 nats).

| dataset | dominant periods |
|---|---|
| Solar GHI | 24, 12, 6 h |
| System load | 24 h, 8760 h, 168 h, 12 h |

The spectra recover the injected structure exactly — diurnal and its harmonics
for solar; diurnal, annual and weekly for load — which is the end-to-end check
that the FFT path is correctly normalised and correctly scaled in frequency.


## Consequence for the QRC work

The objective at the top asked whether a given forecast horizon is limited by
the model or by the dynamics. For observed records **that question cannot be
answered by measuring λ₁** — not with this data, not with a longer record, not
with more members. The noise floor sits above the exponential phase.

What to do instead, in the order I would do it:

**1. Measure skill decay directly.** Run the QRC and its baselines at leads of
1, 3, 6, 12, 24 months and find where skill crosses persistence and
climatology. That *is* the predictability horizon, operationally defined, and
it needs no chaos theory. For a QRC result this is the more defensible number
anyway: it is the thing a reader actually cares about.

**2. Run a surrogate test, for the claim you can defend.** Generate IAAFT
surrogates of Niño 3.4 (same spectrum, same distribution, phases randomised)
and run the identical pipeline on data and surrogates. If the record's
statistics sit inside the surrogate distribution, there is no evidence of
deterministic structure beyond a linear stochastic process — and *that* is
sayable rigorously, which "T_λ = 5.8 yr" never was.

**3. If a λ₁ is genuinely needed, measure a model's.** An initial-condition
large ensemble (CESM2-LENS2's 100 members, listed in `../Data/README.md`) is
structurally the same input `lyapunov_ensemble` already takes — `(R, N)`
instead of `(12, 4000)`. It works because raw model output has no
observational noise and genuinely microscopic IC perturbations. Report it as
*the model's* λ₁, carrying the ±31% from the calibration table. Do not apply
this to observationally-noised fields (ERA5 EDA style); that hits the same
wall documented above.

**4. Forecast error-doubling time**, from a hindcast archive such as NMME.
This measures growth from *realistic* initial error, which lives above the
noise floor and is therefore measurable — and it is the quantity that actually
bounds a forecast, rather than the asymptotic infinitesimal-error rate.

What still stands without qualification: the **spectra**. The ENSO 2.5–5 yr
band, solar's diurnal harmonics, load's diurnal/weekly/annual lines are all
measured, reproducible, and independent of every Lyapunov caveat here. If a
reservoir has to capture the target's frequency content, that is characterised.


## Caveats

1. **No λ₁ is quoted for any observed record**, and this is enforced in code,
   not left to the reader. The rise guard returns *not measurable* whenever
   the divergence curve climbs less than 3 nats. Every real and surrogate
   series here is in that category.
2. **A refusal is not a claim the system is non-chaotic.** It says one finite,
   noisy, scalar record cannot support the measurement. ENSO may well be
   chaotic; this data cannot show it.
3. **Ensemble λ₁ carries ~31% mean absolute error**, so quote one significant
   figure. "T_λ of order 2 natural units", never "1.92".
4. **The error is dominated by window placement, not by the divergence curve.**
   On one fixed Lorenz-63 curve, four hand-picked windows give 0.71, 0.82,
   1.01 and 1.46 — a factor of two with no code change. That spread *is* the
   uncertainty. See check 4 of `manual_check.py`.
5. **Reproducibility is not accuracy.** Rosenstein's per-member scatter is a
   few percent while its error against truth is ~93%; all members share the
   same systematic flaw, so averaging them tightens a wrong answer.
6. **Ground truth in the npz files is asserted, not measured.** `lyap=0.9056`
   is hardcoded in `fetch_data.py` as "the standard published value". It is
   correct — Benettin tangent-space integration at the generator's own step
   size gives 0.9080, +0.3% — but nothing in the pipeline was checking, and
   the four dysts systems have *not* been verified this way. Lorenz-84 and
   Hadley sit at +58–63% ensemble error; whether that is the estimator or
   stale metadata is still open.
7. `dt` for the real and surrogate tiers is **declared, not sniffed**. Monthly
   data on a calendar has unequal spacing in days, and differencing timestamps
   to recover `dt` would inject a spurious 12-month modulation into the
   spectrum.
8. Rosenstein truncates at `max_n=5000` samples — the neighbour search is a
   full O(N²) distance matrix, and the hourly records would otherwise need
   tens of GB. Truncation rather than decimation, because decimating changes
   `dt` and therefore the exponent's units.
9. Interior NaN gaps (the injected sensor outages in the solar series) are
   linearly interpolated. Leaving them would poison both the FFT and the
   divergence curve. Where the outages are too long for that to be honest —
   TAO's multi-year holes, Brest's 120-month gap — the loader instead crops to
   the longest gapless run (`longest_run=True`) and the shortened span is
   reported. Cropping loses record; interpolating invents it.
10. Discharge is analysed as log10(Q): it spans three orders of magnitude, so a
    few flood peaks would otherwise set the scale for both the spectrum and
    the neighbour search.

## Verification

The estimator is checked against answers known independently of it, not
against another implementation of the same method.

```bash
python test_lyapunov.py    # 14 anchors, ~2 s
python manual_check.py     # the same checks, printed step by step
```

| what | result |
|---|---|
| pure exponential, 3 values (isolates the fitter) | exact, 0.0% |
| tent map, λ = ln 2 analytically | exact, 1e-6 |
| logistic map, 40 seeds | −1.2%, sd 6% |
| derivative formula ⟨ln\|f′(x)\|⟩, independent route | 0.693157 vs ln 2 |
| Benettin tangent-space vs published Lorenz | −0.2% |
| **sine wave** | correctly refused |
| **white noise** | correctly refused |

The two null tests are the ones that matter: every positive test above passed
while the estimator was still returning −0.291 for a sine wave. `test_lyapunov.py`
also pins each chaotic system's known error to ±2%, so a refactor that moves
them announces itself.


## Files

| file | role |
|---|---|
| `dataloader.py` | `Series` container; `load(key)` / `load_all(tier)` over all three tiers |
| `fourier.py` | linear-scale amplitude spectrum, Welch PSD, peak picking |
| `lyapunov.py` | embedding, both divergence estimators, scaling-region fit, `T_λ` |
| `visualizer.py` | the four panels and the 2×2 composite figure |
| `analysis.py` | CLI: run the whole pass, print the table, write the PNGs |
| `compare.py` | cross-dataset comparison figure + `comparison.md` |
| `provenance.py` | per-dataset source, observable, known traps |
| `test_lyapunov.py` | 14 anchors: analytic truths, independent routes, null tests |
| `manual_check.py` | the same checks printed step by step, to verify by hand |

Dependencies: numpy, scipy, pandas, matplotlib. See `../Data/README.md` for how
the datasets themselves are built.
