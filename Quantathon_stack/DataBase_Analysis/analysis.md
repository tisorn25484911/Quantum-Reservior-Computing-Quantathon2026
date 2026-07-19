# Database Analysis

Characterise every dataset in `Quantathon_stack/Data` before any of it is fed
to a reservoir: what it looks like, what periods it contains, how fast it
forgets its initial condition, and how far ahead it can be predicted even in
principle.

```bash
python analysis.py                       # every dataset present on disk
python analysis.py --dataset nino34      # one
python analysis.py --tier chaotic        # one tier
python analysis.py --method rosenstein   # force single-trajectory
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

λ₁ is the slope of the divergence curve over its scaling region, located by
**fraction of total rise** (10%–60% by default) rather than by fixed index.
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

### Real tier

Ten observed records, no realizations anywhere, so every row is Rosenstein and
every row is an upper bound. Cross-dataset comparison: `python compare.py`,
written up in [comparison.md](comparison.md).

![cross-dataset comparison](figures/comparison.png)

| dataset | λ₁ | T_λ | T_λ (days) | smp/T_λ | e-folds | dominant | T_λ/period |
|---|---|---|---|---|---|---|---|
| `nyc` NYC Tmax | 0.0626 /day | 16.0 day | 16.0 | 16 | 3603 | 364 day | 0.04 |
| `lax` LA Tmax | 0.0847 /day | 11.8 day | 11.8 | 12 | 2536 | 365 day | 0.03 |
| `tao` 0N140W SST | 0.0983 /day | 10.2 day | 10.2 | 10 | 166 | 421 day | 0.02 |
| `potomac` discharge | 0.1829 /day | 5.47 day | 5.5 | **5** | 6439 | 367 day | 0.01 |
| `potomac15` discharge | 0.1037 /h | 9.65 h | 0.40 | 39 | 4543 | 8760 h | 0.00 |
| `opsd` German load | 0.0177 /h | 56.5 h | 2.35 | 56 | 892 | 24 h | 2.35 |
| `nino34` anomaly | 0.1729 /yr | 5.78 yr | 2111 | 69 | 13 | 3.64 yr | 1.59 |
| `nino12` raw SST | 0.6280 /yr | 1.59 yr | 582 | 19 | 38 | 1.00 yr | 1.59 |
| `brest` sea level | 0.2915 /yr | 3.43 yr | 1252 | 41 | 12 | 1.01 yr | 3.40 |
| `cuxhaven` sea level | **−2.2150 /yr** | ∞ | — | — | 0 | 1.00 yr | — |

Four things in that table are worth more than the exponents themselves.

**The weather stations validate the method.** NYC Tmax gives T_λ = 16 days and
LA 11.8 days, against a textbook atmospheric predictability limit of roughly
two weeks. Neither number was tuned to land there. These are also the
best-conditioned records in the whole directory — 57,540 and 29,935 daily
samples, essentially no gaps — so the agreement is evidence that the pipeline
is sound when the data is good, which is exactly what makes the failures below
interpretable as data problems rather than code problems.

**`cuxhaven` returns a negative exponent, and that is a failure, not a
finding.** A 171-year monthly record with a strong secular trend and a
dominant annual line gives embedded neighbours that converge rather than
diverge; the fit then reports λ₁ < 0 and T_λ = ∞. The honest reading is that
the estimator does not apply to this series as loaded. Detrend and deseasonalise
before believing anything about sea-level dynamics. `brest` avoids the same
fate only because `longest_run` cropped it to 42 years, which drops most of the
trend along with most of the record.

**`potomac` daily is under-resolved.** Five samples per T_λ is below the ~10
needed for a scaling region to exist, so 5.47 days is not supported by its own
data. The 15-minute version of the same gauge has 39 samples per T_λ and gives
T_λ = 9.65 h — a *different* answer for the same river, at 0.4 days versus 5.5.
Sampling rate is setting the exponent here, the same failure documented for the
surrogate load series. Prefer `potomac15`.

**Real load is more predictable than the surrogate.** `opsd` gives T_λ = 56.5 h
against 21.1 h for `load`, i.e. the synthetic series is over-chaotic by 2.7×.
Both have a 24 h dominant line, so the surrogate's AR(1) residual is noisier
than real demand. Anything tuned against the surrogate should be re-checked
against `opsd` before it is believed.

The `T_λ/period` column is the credibility check. Values near 1 mean the
predictability horizon coincides with the dominant cycle, which is what
phase-matching looks like: `nino12` sits at 1.59 with a 1.00 yr dominant line
and is the clearest artefact in the set. The daily records sit at 0.01–0.04,
comfortably decoupled from their annual cycle, which is a further reason to
trust them.

![real-tier spectra](figures/spectra_real.png)

#### ENSO, in detail

| dataset | λ₁ (/yr) | T_λ (yr) | m, τ | record |
|---|---|---|---|---|
| Niño 3.4 anomaly | 0.1729 | 5.78 | 7, 8 | 918 mo, 13.2 e-foldings |
| Niño 1+2 SST | 0.6280 | 1.59 | 4, 4 | 732 mo, 38.3 e-foldings |

**The two must be read together.** Same phenomenon, λ₁ differing by 3.6×, and
the spectra say why: Niño 1+2 is raw SST, so its seasonal cycle dominates at
four times the amplitude of any other line, while Niño 3.4 is an anomaly with
the cycle already removed. Rosenstein inflates exactly when a periodic
component lets embedded neighbours phase-match. The Niño 1+2 number is that
failure mode running live — 0.628 /yr is the annual cycle, not ENSO dynamics.

Use Niño 3.4. And even 0.173 /yr is an upper bound: Rosenstein ran +75% to
+1210% across the ground-truth systems, and no ensemble exists to correct it
here. The defensible statement is **T_λ ≳ 6 yr, order a decade, one
significant figure**.

What *is* solid is the band structure. Power spread across 2.5–5 yr with no
single dominant line is the canonical ENSO recurrence and is the spectral
signature of chaotic rather than periodic behaviour. The 12.75 yr peak is only
about six cycles into a 76-year record — under-resolved, do not quote it.

`tao` is the daily counterpart to those monthly indices, and the only real
record here with sibling series (five moorings). It is not an initial-condition
ensemble — the moorings are dynamically coupled along the equator — so it
cannot drive the ensemble estimator. Its 25–39% missing rate forces
`longest_run`, which leaves 1,684 of 16,919 days; a 46-year record collapses to
a 4.6-year working span.

### Surrogate tier

Generated series, so λ₁ describes the generator, not any climate.

| dataset | λ₁ (/h) | T_λ (h) | dominant periods |
|---|---|---|---|
| Solar GHI | 0.0178 | 56.1 | 24, 12, 6 h |
| System load | 0.0474 | 21.1 | 24 h, 8760 h, 168 h, 12 h |

The spectra recover the injected structure exactly — diurnal and its
harmonics for solar; diurnal, annual, and weekly for load — which is the
end-to-end check that the FFT path is correctly normalised and correctly
scaled in frequency.

## Consequence for the QRC work

For Niño 3.4, a 6–12 month forecast is **0.09–0.17 e-foldings**. That is far
inside the predictability horizon, so skill shortfalls at that range are model
problems and worth engineering against. Beyond roughly three years the dynamics
start to dominate and no reservoir recovers what has been lost.

For the surrogate series, day-ahead forecasting is ~0.4 e-foldings (solar) and
~1.1 (load) — the load task sits right at the boundary, which is where a memory
argument for a reservoir is most likely to be doing real work. But `opsd` puts
real German demand at T_λ = 56.5 h, so day-ahead on *real* load is only ~0.4
e-foldings: comfortably inside the horizon, and an easier task than the
surrogate implied. Re-tune against `opsd` rather than `load`.

The daily weather stations are the most promising untried targets. Day-ahead
Tmax is ~0.06 e-foldings and a two-week forecast is ~0.9 — a task that spans
the interesting range end to end, on 57,540 clean samples with no gap handling
required. Nothing else here offers that combination.

## Caveats

1. **Every λ₁ from a real record here is an upper bound**, roughly factor-of-2.
   Single-trajectory Rosenstein is not a measurement. The output labels these
   rows explicitly.
2. **Ensemble λ₁ carries ~31% mean absolute error**, so quote one significant
   figure.
3. `dt` for the real and surrogate tiers is **declared, not sniffed**. Monthly
   data on a calendar has unequal spacing in days, and differencing timestamps
   to recover `dt` would inject a spurious 12-month modulation into the
   spectrum.
4. Rosenstein truncates at `max_n=5000` samples — the neighbour search is a
   full O(N²) distance matrix, and the hourly records would otherwise need tens
   of GB. Truncation rather than decimation, because decimating changes `dt`
   and therefore the exponent's units.
5. Interior NaN gaps (the injected sensor outages in the solar series) are
   linearly interpolated. Leaving them would poison both the FFT and the
   divergence curve. Where the outages are too long for that to be honest --
   TAO's multi-year holes, Brest's 120-month gap -- the loader instead crops to
   the longest gapless run (`longest_run=True`) and the shortened span is
   reported. Cropping loses record; interpolating invents it.
6. **A negative λ₁ means the estimator failed, not that the system is
   non-chaotic.** `cuxhaven` returns −2.2 /yr because a strong trend plus a
   dominant annual line makes embedded neighbours converge. Detrend and
   deseasonalise first.
7. **Check `samples/T_λ` before quoting any exponent.** `potomac` daily sits at
   5, and its own 15-minute record disagrees with it by a factor of 14. Below
   ~10 samples per e-folding there is no scaling region to fit.
8. Discharge is analysed as log10(Q): it spans three orders of magnitude, so a
   few flood peaks would otherwise set the scale for both the spectrum and the
   neighbour search.

## Files

| file | role |
|---|---|
| `dataloader.py` | `Series` container; `load(key)` / `load_all(tier)` over all three tiers |
| `fourier.py` | linear-scale amplitude spectrum, Welch PSD, peak picking |
| `lyapunov.py` | embedding, both divergence estimators, scaling-region fit, `T_λ` |
| `visualizer.py` | the four panels and the 2×2 composite figure |
| `analysis.py` | CLI: run the whole pass, print the table, write the PNGs |
| `compare.py` | cross-dataset comparison figure + `comparison.md` |

Dependencies: numpy, scipy, pandas, matplotlib. See `../Data/README.md` for how
the datasets themselves are built.
