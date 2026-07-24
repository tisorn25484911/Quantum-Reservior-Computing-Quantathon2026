# Makefile -- one target per experiment group. The LARGEST grids are listed but
# deliberately NOT wired into `make all` (spec s31). PY points at the local venv
# `.venv` (created at repo relocation; qiskit/statsmodels installed on demand for
# P3/P8). Override with `PY=... make <target>` to use a different interpreter.
PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  %-22s %s\n", $$1, $$2}'

# ---- Phase 0 (audit + smoke) ----------------------------------------------
.PHONY: audit test all
audit: ## P0: inspect data + repo, (re)write manifests
	$(PY) scripts/inspect_data.py
	$(PY) scripts/inspect_repository.py

test: ## run the (fast) test suite
	$(PY) -m pytest -q

all: audit test ## SAFE default: audit + fast tests only (never the big grids)

# ---- Per-phase entry points (wired as they are implemented) ---------------
.PHONY: validate-exact validate-ladder mackey-glass baselines
validate-exact: ## P2: exact-QRC anchors + FN Fig.5/Table-I reproduction
	$(PY) scripts/validate_exact_qrc.py
validate-ladder: ## P3: exact<->Trotter<->Qiskit<->noisy validation ladder
	$(PY) scripts/validate_qiskit_qrc.py
mackey-glass: ## P5 GATE: teacher-forced -> autonomous Mackey-Glass
	$(PY) scripts/reproduce_mackey_glass.py
baselines: ## P7: full baseline battery (teacher-forced + autonomous)
	$(PY) scripts/run_baselines.py

# ---- EXPENSIVE grids: run explicitly, never via `make all` -----------------
.PHONY: grid-autonomous grid-shots grid-noise report
grid-autonomous: ## P8/P9: multi-origin autonomous campaign (hours)
	$(PY) scripts/run_autonomous_forecasts.py
grid-shots: ## P10: shot-budget x seed x origin sweep (hours-day)
	$(PY) scripts/compare_shot_budgets.py
grid-noise: ## P10: noise-model sweep + feature-space diagnostics (hours-day)
	$(PY) scripts/compare_noise_models.py
report: ## P12: assemble the final report from stored results only
	$(PY) scripts/generate_report.py
