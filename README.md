# Quantum Reservoir Computing — Quantathon 2026

Quantum reservoir computing applied to time-series forecasting: from an exact
NumPy reservoir, through dataset characterisation, to a runnable demo app.

**Picking this up?** Read [Quick start](#quick-start), then
[Where things stand](#where-things-stand), then [What to do next](#what-to-do-next).

---

## Repo map

| Directory | What it is |
|---|---|
| **`Quantathon_stack/`** | The active working area — data, analysis, evaluation, the anomaly-forecast line. |
| **`webapp/`** | FastAPI + Jinja/HTMX demo over everything below. Fastest way to see what the project does. |
| **`QRC_code_stack/`** | The staged repository (stages 0–8: anchors, NumPy core, exact circuits, noise, Hamiltonian battery, qubit reuse, RF-QRC, integration, product plan). Stages 0–7 green under `run_tests.py`; see its `README.md` and `HANDOFF.md`. |
| **`LaTeX_logbook/`** | The handbook (Parts I–XI) whose Parts IX–X specify the repository design; includes verified companion code and regenerated figures. |
| **`.claude/skills/`** | Project skills used to build and audit this work (QRC playbook, citation audit, arXiv retrieval, etc.). |

### Inside `Quantathon_stack/`

| Directory | Contents |
|---|---|
| `Data/` | 17 datasets in 3 provenance tiers: `real/` (ENSO, SST, river discharge, load, sea level, station temps), `chaotic/` (Lorenz63/84, Hadley, Rikitake — with published λ₁), `surrogate/` (synthetic solar, load). |
| `DataBase_Analysis/` | `dataloader.py` (uniform access by short key), `lyapunov.py` (λ₁, Lyapunov time), `fourier.py` (spectra), `analysis.py` / `compare.py` (per-dataset and cross-dataset reports). |
| `Hamiltonian_QRC/` | `qrc_core.py` — **the reservoir everything uses** (exact density-matrix, pure NumPy, `ising` + `xxz_hx`). Also `Hamiltonians.py` (QuTiP, standalone — see gotchas). |
| `Main_run_Evaluation/` | `memory_capacity.py` (linear MC, held-out scoring + surrogate noise floor), `mc_visualizer.py` (4-panel figure). |
| `Anomaly_Forecast/` | The Forecast-then-Detect line: `plan.md` (the spec — **read first**), `forecast.py` (Steps 1–2), `run_step2.py` (kill-test). |

---

## Quick start

One virtualenv at the repo root serves everything.

```bash
python3 -m venv .venv
.venv/bin/pip install -r webapp/requirements.txt
.venv/bin/pip install scikit-learn stumpy      # anomaly-forecast line
```

**The demo app** — easiest way to see everything:

```bash
cd webapp
../.venv/bin/python -m uvicorn app.main:app --reload --port 8000
# → http://127.0.0.1:8000        API docs at /docs
```

Pages: `/forecast` (QRC forecast + conformal band + baselines), `/datasets`,
`/chaos` (predictability, λ₁), `/anomaly` (recursive-rollout skill decay),
`/diagnostics` (memory capacity), `/about` (what the numbers may claim).

**The scripts**, each self-verifying:

```bash
# reservoir memory capacity — 9 validation anchors, then a demo table
.venv/bin/python Quantathon_stack/Main_run_Evaluation/memory_capacity.py

# 4-panel memory-capacity figure → figures/memory_capacity.png
cd Quantathon_stack/Main_run_Evaluation && ../../.venv/bin/python mc_visualizer.py

# dataset characterisation → figures/<key>_analysis.png
cd Quantathon_stack/DataBase_Analysis && ../../.venv/bin/python analysis.py --dataset nino34

# anomaly-forecast Step 1 gates
cd Quantathon_stack/Anomaly_Forecast && ../../.venv/bin/python forecast.py

# anomaly-forecast Step 2 kill-test → results/
cd Quantathon_stack/Anomaly_Forecast && ../../.venv/bin/python run_step2.py
../../.venv/bin/python run_step2.py --dataset tao
```

---

## Where things stand

### Verified and working

- **`memory_capacity.py`** — linear memory capacity with held-out scoring and a
  surrogate noise floor. 9 anchors pass, including a length-8 delay line scoring
  **MC = 7.000 exactly**, its analytic answer.
- **`webapp/`** — all routes render, error paths correct, charts served. Three
  guards are structural rather than optional: train-span-only scaler (leakage),
  a size-matched ESN scored on identical rows, and *measured* conformal coverage.
- **`Anomaly_Forecast/` Steps 1–2** — gates pass (stepwise driving ≡ batch `run()`
  to 1e-12; `rollout(H=1)` ≡ `forecast_1step`), and 1-step NMSE
  **0.0582 ising / 0.0535 xxz_hx** against the notebook's independently reported
  **0.0577 / 0.0496**.

### The main finding, stated plainly

On **`nino34`** (the current target) the quantum reservoir reaches a usable lead of
**7 months** and materially beats persistence — but it sits **at parity with, or
behind, a size-matched classical ESN**. No quantum advantage.

A 5-seed sweep matters here and was nearly missed:

- **`xxz_hx` is deterministic** — a clean chain with no disorder, so `rng` is
  ignored by design. A single-seed QRC-vs-ESN comparison therefore pits a fixed
  point against one draw of a random variable. That is not a fair test.
- On `nino34` the ESN wins 10–11 of 12 horizons at 3 of 5 seeds. The initial
  "parity" reading came from one weak ESN draw.
- On **`tao`** (daily SST, N≈7000) QRC wins 10–11 of 12 horizons at **every** seed,
  and the ESN is unstable there (σ = 0.50 at h=6; one seed reaches NMSE 1.83).

**So: always compare against several ESN seeds, quoting mean and spread.**

Full detail and resolved open items:
[`Quantathon_stack/Anomaly_Forecast/plan.md`](Quantathon_stack/Anomaly_Forecast/plan.md) §10.

---

## What to do next

**1. Switch the anomaly-forecast target from `nino34` to `tao`.** Highest value,
lowest effort — already supported (`run_step2.py --dataset tao`). `nino34` was
chosen before any of this was measurable; the evidence now says it is the weaker
target on every axis:

| | `nino34` | `tao` |
|---|---|---|
| QRC vs ESN | ESN wins at 3/5 seeds | QRC wins at 5/5 seeds |
| N | 918 | ~7000 |
| EVT tail fitting (plan §2 risk) | ~18 exceedances — noisy | >100 — fine |
| λ₁ empirical / physical | 0.10 — discrepant | 0.79 — consistent |
| climatology baseline | **degenerate** (already an anomaly series) | intact |

Keep `nino34` as the hard-case companion.

**2. Then plan.md Step 3 — the stochastic rollout.** The next kill-test: ridge
minimises MSE, so the mean trajectory is smooth and will systematically
*under-alarm*. Sample K trajectories and verify the ensemble is calibrated before
building anything on top of it. Adjust one expectation: at h=6 on `nino34`,
NMSE ≈ 0.87 means the mean trajectory is nearly flat, so realistic
anomaly-prediction lead is likely **h ≈ 1–3**, not the full 7.

**3. Add validation anchors to `webapp/app/domain/`.** This is the open gap against
the project's own rule R1 ("no stage accepts code before its validation anchors
exist"). Stage 0 has 77 anchors, `memory_capacity.py` has 9, the webapp has
**zero**. Highest-risk item is persistence target alignment (`y[k] = x[k+h]` scored
against `pred = x[k]`) — an off-by-one there shifts every skill number the app
displays and would still look entirely plausible.

**4. A multi-seed harness**, so the ESN baseline is a distribution rather than a
draw. Worth doing for its own sake: during this work a weak abstraction produced a
confidently wrong verdict twice, and both times only an explicit check caught it.

---

## Gotchas

- **Use `.venv` at the repo root.** The system `python3` here has NumPy but *not*
  matplotlib; the Homebrew `python3` has matplotlib but is a different interpreter.
  Mixing them produces confusing import errors.
- **`Hamiltonians.py` needs QuTiP, which is not installed — and nothing else
  imports it.** The reservoir actually in use is `qrc_core.py` (pure NumPy), which
  defines its own `ising` and `xxz_hx` builders mirroring that file. They are
  duplicate definitions of the same physics; worth consolidating, but do not assume
  `Hamiltonians.py` is on a live code path.
- **`predict_timeseries.ipynb` is stale** — it imports modules that do not exist
  (`experiments`, `data_loader`). `Anomaly_Forecast/forecast.py` replaces it by
  decision; the notebook is left untouched.
- **`.claude/settings.local.json`** is machine-local config that keeps appearing as
  modified and will block `git pull --rebase`. Consider
  `echo ".claude/settings.local.json" >> .gitignore && git rm --cached` on it.
- **Memory capacity needs an i.i.d. drive.** It is measured on synthetic noise, not
  on the datasets, deliberately: the real series have lag-1 autocorrelation
  0.95–0.99, where a memoryless read-out can infer past inputs from the present one
  and inflate MC into meaninglessness.
- **Rosenstein λ₁ is unreliable on seasonal records.** `analysis.py` says so itself
  (~93% mean error where truth is known). Treat `H_max` on `nino34` as a scale, not
  a measurement, and set horizons from measured skill decay instead.

---

## The claim discipline

At these sizes (n ≤ 8 qubits, exact simulation) **there is no quantum speed-up to
demonstrate, and none is claimed.** Every result is classical simulation of a small
quantum system. The defensible statement is *parity with a size-matched classical
baseline*, and every reported number ships with persistence and ESN controls on
identical data. When a baseline wins, the code and the app are both written to make
saying so the default outcome rather than an exception.
