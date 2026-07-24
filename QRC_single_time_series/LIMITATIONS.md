# LIMITATIONS.md

Standing limitations (updated every phase; finalised in Phase 12).

- **Short observational records** (enso 732, soi 906, pdo 932 monthly points).
  Origin overlap is unavoidable → block bootstrap + effective-sample-size beside
  every CI; per-origin results shown, not only aggregates.
- **Dense-ρ memory wall**: the FN stateful track cannot be purified (reset each
  step) → capped at N≤10 (paper uses ≤7). Rewind track escapes this via
  Stinespring dilation (exact pure statevector).
- **ENSO predictability barrier**: operational skill degrades beyond ~6–12 months
  and is worst for boreal-spring starts; a univariate index-only autonomous model
  sits at the weak end. >12-month autonomous skill triggers a leakage hunt.
- **Shot-based rewind map is stochastic** → a Lyapunov *spectrum* is undefined;
  those runs report divergence rates only.
- **Identity caveats**: enso.csv is raw Niño 1+2 SST (not ONI/Niño3.4/MEI); soi is
  the CPC anomaly (not Troup); results are specific to these variants.
- **No real-QPU results** unless a separate L6 run is added; simulator-only.
- **Product track is a design**, not an executed validation; no commercial claim.
