# HANDOFF_P0 — Repository/data audit, environment, scaffold, PLAN.md

Written per G10. Location is in-repo (declared deviation from the `handoff`
skill's temp-dir default) so the next session finds it.

## 1. Objective & status vs exit criteria

Phase-0 goal: everything in the spec §6 first-deliverable list minus physics —
inspect repo + the three CSVs, pin the environment, create the scaffold, write
`PLAN.md`, freeze the preregistration.

| Exit criterion | Status | Evidence |
|---|---|---|
| Manifests exist and are checksummed | ✅ | `data/manifests/{enso,pdo,soi}.json`; `test_manifest_checksum_matches_raw` green |
| `PLAN.md` written (spec §6 contents) | ✅ | `PLAN.md` (20 sections) |
| Preregistration frozen (G6) | ✅ | `configs/preregistration.yaml` (version 1, locked) |
| `pytest` runs (suite passes) | ✅ | **20 passed, 16 skipped** in 0.54s |
| Nothing outside the folder touched | ✅ | all 169 files under `QRC_single_time_series/`; only external action = `pip install pyyaml` into the shared venv (documented) |

## 2. Artifacts & keys

- Raw data (unmodified, checksummed): `data/raw/{enso,pdo,soi}.csv`.
  - enso sha256 `5ad05f6f50b6…`, 732 rows, 1950-01…2010-12.
  - pdo  sha256 `82e94fd452b6…`, 932 rows, 1948-01…2025-08.
  - soi  sha256 `f0e5a98e93d4…`, 906 rows, 1951-01…2026-06.
- Manifests: `data/manifests/*.json` (identity, units, coverage, policies, stats).
- Repo inventory + environment: `results/logs/repo_inventory.json`.
- Frozen config: `configs/preregistration.yaml`; dataset registry `configs/data.yaml`.
- Scaffold: full spec-§6 tree; 62 package modules (stubs + real P0 modules),
  17 script stubs, 16 skipped test anchors, 7 placeholder notebooks.
- Pinned env: `requirements-lock.txt` (qiskit 2.5.0, aer 0.17.2, numpy 2.4.6,
  scipy 1.17.1, pandas 3.0.3, statsmodels 0.14.6, pennylane 0.45.1, pyyaml 6.0.3).

## 3. Decisions & deviations (with reasons)

- **Project root** = `Quantathon2026/final_pitch/QRC_single_time_series/` (moved
  into `final_pitch/` at the user's request, 2026-07-24; self-contained folder).
  The pinned venv is now reached at `../../QRC_main_stack/QRC_code_stack/.venv`
  (Makefile `PY` and the README quick-start updated accordingly). All in-package
  paths use file-relative roots, so the move required no code change.
- **Interpreter** = the shared pinned venv `QRC_main_stack/QRC_code_stack/.venv`
  (already has qiskit/aer/statsmodels/pennylane). Added only `pyyaml` (missing
  light dep). `torch` left as an optional Phase-7 extra. Rationale: avoids a
  redundant multi-GB qiskit reinstall; versions are pinned in the lock file.
- **Chronology frozen in `data/loaders.py`**: final 20% untouched test; rolling
  origin inside the first 80%. Encoded so no later code can re-split (soundness
  review). `TEST_FRACTION=0.20`.
- **Anomaly convention**: model all three as anomalies vs a train-years-only
  monthly climatology (removes ENSO's annual cycle; mild for PDO/SOI). Transforms
  are ablation arms (`configs/data.yaml`).
- Handoff lives in-repo (G10 declared deviation).

## 4. Resolved index identities (the spec's key trap) & anomaly convention

- **enso.csv** = statsmodels `elnino` **exactly** (first-12 == row0; equal means):
  NOAA **Niño 1+2** region (0–10°S, 90–80°W), **raw** monthly SST [°C] — **NOT**
  an anomaly, **NOT** ONI/Niño3.4/MEI.
- **soi.csv** = NOAA CPC **ANOMALY** block (Tahiti−Darwin SLP anomaly) — **not**
  Troup-standardised. Already an anomaly.
- **pdo.csv** = NOAA PSL **Mantua-style** PDO index. Already an anomaly
  (distinguish from ERSSTv5 variants post-2002).
- All monthly, gap-free, no duplicate dates, no NaN, no −99.9/−999 sentinels.

## 5. Frozen preregistration values (G6 highlights)

ε ∈ {0.25, 0.5, 1.0} (σ_train-normalised), primary ε=0.5, sustained K=3;
reliability p ∈ {0.8, 0.9}; clip budget 0.20, boundary-dwell 0.30; shot grid
{128…8192}×8 seeds; Jτ grid over ≥2 decades; input-gain ≥3 settings; horizons
{1,3,6,12} months; 20–50 origins; block bootstrap (stationary, block =
decorrelation time). **Gate-1 criterion locked**: on the untouched test span,
exact QRC beats persistence (skill>0, lower 90%-CI>0) at h∈{1,3} on ≥1 index;
Gate-3 additionally needs H_effective(ε=0.5) ≥ 3 months with survival ≥ 0.8.

## 6. Open issues / known gaps

- All physics modules are docstring stubs (by design); the 16 skipped tests are
  their placeholders.
- `torch` not installed (Phase-7 neural baselines will need `pip install torch`).
- Vallis-1986 full text not yet fetched (Phase-8 equation audit, errata #3).

## 7. Entry instructions for Phase 1 (exact operator core + anchors)

1. `cd final_pitch/QRC_single_time_series && make test` must be green first (20 passed).
2. Implement `quantum/{pauli,tensor,partial_trace,hamiltonians,input_channels,
   observables}.py` per PLAN §4 and G2.
3. Turn the skipped anchors green: `tests/{test_pauli,test_partial_trace,
   test_hamiltonian,test_encoding}.py`. Required anchors (from the phase file):
   re-injection identity (⟨Z_inj⟩=1−2s, ⟨X_inj⟩=2√(s(1−s)), other sites
   untouched); trace/Hermiticity/positivity under injection∘evolution; U
   unitary; `expm` == eigendecomposition to 1e−12; optional Atas ⟨r⟩ diagnostic.
4. Exit at N ∈ {2,3,5} all green; write `handoffs/HANDOFF_P1.md`.

## 8. Suggested skills for the next session

`quantum-reservoir-computing` + `qrc-project-playbook` (minimum, per G10);
add `citation-audit` only when a document ships.
