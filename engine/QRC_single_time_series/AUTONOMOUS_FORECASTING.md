# AUTONOMOUS_FORECASTING.md

The autonomous (closed-loop) protocol and its three horizon concepts
(populated in Phase 4). Reserved sections:
1. Open-loop vs autonomous (spec §4.1); the structural closure condition (§4.2).
2. Stateful-FN rollout (persistent ρ) vs rewind-buffer rollout (provenance mask).
3. Domain policies (clip / smooth / terminate / wide-encoding) + clip telemetry.
4. Valid trajectory time `H_error(ε)`; skill horizon `H_skill`; reliability
   `H_reliable(p)`; `H_effective = min(...)` with all components reported.
5. Dynamical / attractor fidelity vs pointwise accuracy — never conflated.
6. Failure taxonomy and first-failure time.
7. Shot-noise compounding in the feedback loop; `T_valid ≈ a + (1/2λ)·ln S`.

Frozen thresholds live in `configs/preregistration.yaml`.
