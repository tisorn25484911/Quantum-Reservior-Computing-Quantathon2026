# HANDOFF_P1 — Exact operator core and validation anchors

Written per G10. In-repo (declared deviation from the `handoff` skill's temp-dir
default), same as HANDOFF_P0.

## 0. Environment fix landed first (blocker from the repo move)

The pinned venv `../../QRC_main_stack/QRC_code_stack/.venv` referenced by the P0
Makefile **no longer exists** — the repo was relocated to
`Projects/Quantathon2026-QRC_detection/QRC_single_time_series/` with no sibling
`QRC_main_stack`. Fix:

- Created a local **`.venv`** (`python -m venv`, homebrew py3.13) and installed
  the *light* pinned deps only: `numpy==2.4.6 scipy==1.17.1 pandas==3.0.3
  pyyaml==6.0.3 pytest==9.1.1`. P1 needs only numpy/scipy; P0 tests need
  pandas/pyyaml. **qiskit / qiskit-aer / statsmodels / pennylane / matplotlib
  are NOT installed** — defer to when P3 (qiskit ladder) / P8 (statsmodels) need
  them: `uv pip install --python .venv/bin/python qiskit==2.5.0 ...`.
- Makefile `PY` default → `.venv/bin/python` (still overridable: `PY=... make test`).
- Added `.gitignore` (`.venv/`, `__pycache__/`, `.pytest_cache/`,
  `results/features/*.npy`) — none existed before, repo would have tracked the venv.

## 1. Objective & status vs exit criteria

Phase-1 goal: `quantum/{pauli,tensor,partial_trace,hamiltonians,input_channels}.py`
implemented and validated against known exact answers before any reservoir exists.

| Exit criterion | Status | Evidence |
|---|---|---|
| Operator core implemented (G2 order) | ✅ | 5 modules below |
| G2 re-injection anchor green | ✅ | `test_partial_trace.py::test_g2_reinjection_anchor` — <Z>=(+1,0,0) |
| ⟨Z_inj⟩=1−2s, ⟨X_inj⟩=2√(s(1−s)) | ✅ | `test_encoding.py::test_injection_sets_inj_qubit_leaves_others` |
| ⟨Z_other⟩ untouched by injection | ✅ | same test, sites 1..N−1 within 1e−12 |
| trace/Hermiticity/positivity under injection∘evolution | ✅ | `test_partial_trace.py::test_injection_then_evolution_is_cptp` |
| U unitary | ✅ | `test_hamiltonian.py::test_evolution_is_unitary` |
| expm == eigendecomposition to 1e−12 | ✅ | `test_hamiltonian.py::test_expm_matches_eigendecomposition` (actual ~1e−15) |
| purity falls 1→plateau under repeated injection | ✅ | `test_encoding.py::test_purity_falls_from_one_to_plateau` |
| Atas ⟨r⟩ diagnostic | ✅ | `test_hamiltonian.py::test_r_value_signals_chaos_at_n5` |
| All anchors green at N ∈ {2,3,5} | ✅ | tests parametrized over NS=[2,3,5] |
| Suite passes | ✅ | **95 passed, 12 skipped** in ~1s (was 20 passed/16 skipped) |

## 2. Artifacts & keys

Modules (all `src/qrc_single_time_series/quantum/`):
- `tensor.py` — `kron_all`, `embed(op,site,N)`, `embed_pair`. **G2 contract**: site 0
  is the leftmost `np.kron` factor (MSB of basis index); embed = I(2**site) ⊗ op ⊗
  I(2**(N−1−site)).
- `pauli.py` — `I,X,Y,Z`; basis DMs `KET0,KET1,PLUS,MINUS`; `site_operator`,
  `pauli_string("ZIX")`, `expect(rho,op)` (asserts real), `site_expect`.
- `partial_trace.py` — `partial_trace(rho,keep,N)` (einsum over shared traced
  labels), `trace_out(rho,sites,N)`, `reinsert(reduced,rho1,site,N)` (structural
  inverse; einsum outer product with labeled axes). N≤10 (needs 2N≤52 letters, fine).
- `input_channels.py` — `rho_s(s)=½(I+2√(s(1−s))X+(1−2s)Z)` (pure), `ry_density(θ)`
  (= rho_s(sin²(θ/2))), `inject(rho,s,N,site=0)` = trace_out∘reinsert (CPTP by
  construction, spec §8.3 satisfied structurally — never a statevector unitary),
  `purity`.
- `hamiltonians.py` — `Reservoir` dataclass (H, J matrix, h, seed, topology, cached
  eigh); `fc_tfi(N,J,h,seed)` FN fully-connected, `nn_tfi(...,periodic=)` Hamhoum
  chain. J_ij~U[−J/2,J/2] frozen per seed. `.evolution(τ)` (eig path),
  `.evolution_expm(τ)` (scipy ref), `.eigenvalues`, `.r_value()`.

Tests (real, replacing the P1 skip placeholders): `tests/test_{pauli,partial_trace,
hamiltonian,encoding}.py`.

## 3. Endianness / G2 contract test names (recorded per G10)

- `test_partial_trace.py::test_g2_reinjection_anchor` — the gotcha anchor
  (|0⟩⟨0|⊗|1⟩⟨1|⊗|+⟩⟨+|, trace out site 1, re-inject |+⟩ → single-site ⟨Z⟩=(+1,0,0)).
- `test_partial_trace.py::test_reinsert_is_trace_out_inverse_on_product` — reinsert
  ∘ trace_out identity on product-cut states, every site, N∈{2,3,5}.
- `test_pauli.py::test_embed_is_leftmost_site0` — asserts site 0 = leftmost factor.
- **Note**: the *Qiskit-wire* half of G2 (`quantum/endianness.py`: logical↔wire map +
  counts-key reversal) is still a stub — it lands in **P3** with the qiskit path,
  per the phase file. No qiskit imported yet.

## 4. Recorded ⟨r⟩ for the default reservoir seed (reservoir_seed=7, fc, J=h=1)

| N | ⟨r⟩ | note |
|---|---|---|
| 2 | nan | central-50% spectrum too short (1 gap) — expected; ⟨r⟩ needs N≳5 |
| 3 | 0.4300 | small-N, not diagnostic |
| 5 | 0.5957 | near/above GOE 0.5307 → chaotic |
| 6 | 0.4917 | between Poisson 0.3863 and GOE |
| 7 | 0.4963 | " |

Atas anchors: Poisson 0.3863, GOE 0.5307. The N=5 default config is comfortably
chaotic; the ⟨r⟩ test gate is `0.40 < r < 0.65` at N=5. Use ⟨r⟩ only at N≥5.

## 5. Decisions & deviations

- **Injection is CPTP, not unitary** — implemented as partial-trace-then-tensor
  (spec §8.3), matching the soundness-review requirement structurally.
- **⟨r⟩ central 50%** of the sorted spectrum with gaps <1e−12 dropped (spec).
- `partial_trace`/`reinsert` use einsum with per-axis labels (clear over manual
  reshape/transpose); capped at N≤10 by the 52-letter alphabet — well above the
  dense N≤10 budget wall.
- No commit made (user hasn't asked). Working tree has the 5 modules, 4 tests,
  Makefile edit, new `.gitignore`, and `.venv/` (gitignored).

## 6. Open issues / known gaps

- `quantum/endianness.py`, `observables.py` still stubs (P3 and P2 respectively).
- qiskit/aer/statsmodels/pennylane/matplotlib/torch not installed (install on demand).
- 12 skipped tests remain = the P2+ placeholders.

## 7. Entry instructions for Phase 2 (exact FN reservoir + feature→readout→metric)

1. `make test` green first (95 passed, 12 skipped).
2. Implement `quantum/exact_qrc.py` (`run(inputs,x0)→Iterator[(k,v,ρ)]`, sample at
   t=kτ+(v+1)τ/V per errata G8-4; debug v=−1 post-injection sample for the G3 test),
   `quantum/observables.py` (Pauli-string set, fast Z-family path), `models/readout.py`
   (centered ridge via one economy SVD, λ=0 pinv mode, GCV), `evaluation/metrics.py`
   (both NMSE conventions labelled G8-5, capacity floor as mean over τ_B∈[450,500]).
3. Wire trivial baselines (mean/persistence/seasonal-persistence, train-only).
4. Encode the assert: feature count MV+1 ≤ L/5 (FN overdetermination).
5. Turn green: `tests/test_exact_qrc.py` + readout suite + G3 STM-τ_B=0 exactness.
   Reproduce FN Fig.5 (C_STM rises→saturates ~V≈10) + NARMA Table-I ordering
   qualitatively on one N=5 seed=7 reservoir. Order-of-magnitude only (couplings
   random, pinv cutoff unstated — G8).
6. `evaluation/metrics.py` + effective-rank on every feature matrix from P2 on (G7).
7. Write `handoffs/HANDOFF_P2.md`; record Fig-5 file paths, effective ranks,
   feature-count/L budget table.

## 8. Suggested skills for the next session

`quantum-reservoir-computing` + `qrc-project-playbook` (min per G10); add
`citation-audit` only when a document ships.
