# IMPLEMENTATION_NOTES.md

Cross-cutting engineering contracts and the **Errata register**. Updated every
phase. Contracts are single-point-of-truth rules with one enforcing test each.

## Contracts

- **G2 — qubit order (one place, one test).** NumPy/SciPy: injection qubit is
  the **leftmost** tensor factor (index 0 in `np.kron`), so re-injection needs no
  permutation. Qiskit is little-endian; exactly ONE translator
  (`quantum/endianness.py`) maps logical↔wire and reverses counts keys. Anchors:
  re-injection identity (P1); `X` on Qiskit qubit 0 → counts bit position (P3).
- **G3 — alignment (one function).** `data/windows.py::align(features_by_k, y,
  horizon)` is the only place features pair with targets. Next-step convention:
  a row built from information available at step `k` pairs with `y_{k+1}`.
  Exactness test: STM at τ_B=0 with the debug post-injection sample gives
  capacity `C(0)=1.0` to machine precision (P2).
- **G4 — config identity.** Every experiment is a frozen dataclass/YAML; SHA-256
  over canonical JSON is the artifact key (`utils/configuration.py`). Features
  cached float32 on disk, float64 in solves (≤1% NMSE, asserted).
- **G5 — no future access.** `evaluation/autonomous.py::rollout` receives only
  the history slice. Tests: no-future mutation (bit-identical under garbage
  future); feedback identity (`input_{t+1}==prediction_t`); rewind provenance.

## Errata register (source texts; each resolved by in-repo audit, never silent)

1. **Hamhoum Lorenz (a,b,c)=(10,28,3/8).** Standard chaotic Lorenz uses β=8/3;
   their quoted LLE≈0.9 is the 8/3 figure. → Integrate BOTH at (σ,ρ)=(10,28);
   document which reproduces the behaviour; select and record (Phase 8).
   **RESOLVED (P8):** Benettin QR on the exact Jacobian gives λ_max≈**0.898** for
   β=8/3 (reproduces the ~0.906 claim) and ≈**0.0007** for β=3/8 (non-chaotic).
   The project uses **β=8/3**. `dynamical_systems/lorenz63.py::beta_audit`;
   `scripts/estimate_lyapunov_spectra.py`.
2. **Hamhoum ENSO LLE 0.05–0.1** is quoted from an observational
   coastal-temperature study, not their ODE. → Never transfer; compute λ for the
   actual integrated system (Phase 8). **RESOLVED (P8):** computed in-repo,
   λ_max≈**0.26–0.28 per unit time** (LT≈3.6–3.8); the 0.05–0.1 figure is NOT used
   anywhere. `dynamical_systems/enso_ode.py::audit`.
3. **Hamhoum "ENSO" ODE = Vallis (1986, Science 232, 243) box model**
   (dimensional form). → Audit signs/parameters against Vallis 1986/1988 before
   trusting Eq. 13 (Phase 8). **RESOLVED (P8, partial):** the dimensional box model
   `[u,Te,Tw]` with Hamhoum's Eq. 13 parameters (B=940, Δx=7.5, C=3, u*=−14.2,
   T̄=16, A=1, T*=28) is implemented and **verified chaotic numerically** (bounded
   trajectory, λ_max>0). Caveat: the primary Vallis 1986/1988 PDFs are **scanned
   images** that could not be machine-read in-session, so the sign convention is
   the form reproduced in the secondary Vallis-chaos literature, adopted because it
   reproduces the claimed chaotic behaviour; flagged for a later text-level
   re-check. `dynamical_systems/enso_ode.py`.
4. **FN Eq. 15 (trained observable) is written for V=1.** The V>1 operator is the
   Heisenberg-spread sum `Σ_{n,v} w_{nv} e^{iHvδt} Z_n e^{−iHvδt}`. → Read
   observables at `t+(v+1)τ/V`, v=0..V−1 (NV signals); resolves the index
   off-by-one (Phase 2).
5. **FN NMSE (Eq. A1) normalises by `Σȳ²`,** not target variance — for NARMA10
   the two differ by ~×400. → Compute and label BOTH conventions everywhere
   (`evaluation/metrics.py`).
6. **Spec §13 heading "MACKAY–GLASS"** → Mackey–Glass.
7. **FN ensemble (NMR) readout has no back-action;** any per-shot protocol on the
   FN stateful trajectory is simulation-only emulation, never a hardware claim.

## Phase-0 audit findings (this project's data)

- `enso.csv` is the statsmodels `elnino` dataset **exactly** (first 12 values ==
  row 0; means identical): NOAA Niño **1+2** region (0–10°S, 90–80°W), **raw**
  monthly SST in °C — **not an anomaly** and **not** ONI/Niño3.4/MEI. Modelled as
  an anomaly vs a train-years-only monthly climatology (removes the annual cycle).
- `soi.csv` is the NOAA CPC **ANOMALY** block (Tahiti−Darwin SLP anomaly), **not**
  the Troup-standardised SOI (which differs by a scale factor). Already an anomaly.
- `pdo.csv` is the NOAA PSL Mantua-style PDO index. Already an anomaly. Distinguish
  from ERSSTv5-based PDO variants (differ after ~2002).
- All three: monthly, gap-free, no duplicate dates, no NaNs, no −99.9/−999
  sentinels. Coverage/checksums in `data/manifests/`.

## Environment note

Interpreter is the shared pinned venv `QRC_main_stack/QRC_code_stack/.venv`
(qiskit 2.5.0, aer 0.17.2, numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, statsmodels
0.14.6, pennylane 0.45.1, pyyaml 6.0.3). `pyyaml` was added in Phase 0 (only
missing light dep); `torch` remains an optional Phase-7 extra. Versions pinned in
`requirements-lock.txt`.
