# Model card: hybrid-demo-0.1
Generated: 2026-07-14T05:21:43Z
Claims scope: **simulation**  |  DATA INCLUDES SURROGATE - so do all numbers below

## Configuration
```json
{
 "n_qubits": 4,
 "window_m": 1,
 "n_uploads": 2,
 "entangler": "brickwork_zz",
 "entangler_layers": 2,
 "gamma": 0.7853981633974483,
 "tau": 1.0,
 "disorder_W": 0.0,
 "n_virtual_nodes": 2,
 "readout": "local_zz",
 "leak_eps": 0.3,
 "shots": null,
 "denoise": "none",
 "seed": 7
}
```

## Pre-registered gates and outcomes (nulls included)
```json
{
 "passed": false,
 "quantum_demotion": false,
 "gates": [
  {
   "gate": "ramp_f1_vs_incumbent",
   "value": -0.19619450317124745,
   "target": 0.05,
   "comparator": ">=",
   "status": "FAIL"
  },
  {
   "gate": "latency_sla",
   "value": 1.0,
   "target": 60.0,
   "comparator": "<=",
   "status": "PASS"
  },
  {
   "gate": "provenance_violations",
   "value": 0.0,
   "target": 0.0,
   "comparator": "<=",
   "status": "PASS"
  },
  {
   "gate": "coverage",
   "value": 0.025000000000000022,
   "target": 0.03,
   "comparator": "<=",
   "status": "PASS"
  },
  {
   "gate": "quantum_delta",
   "value": 1.0894952526054067,
   "status": "PASS"
  }
 ]
}
```

## Baseline battery (non-removable)
```json
{
 "stacker": {
  "ridge": {
   "hybrid_nmse": 0.5779576713926591,
   "classical_only_nmse": 1.6674529239980658,
   "quantum_delta": 1.0894952526054067,
   "delta_is_zero": false
  },
  "gbm": {
   "hybrid_nmse": 0.6820743003049614,
   "classical_only_nmse": 0.8846445191628226,
   "quantum_delta": 0.2025702188578612,
   "delta_is_zero": false
  }
 },
 "event_f1": {
  "model": 0.16744186046511625,
  "persistence": 0.3636363636363637
 }
}
```

## Data sources
- solar_surrogate [surrogate] licence: see registry

## Notes
- run_id 2026-07-14T05:21:42Z#demo
- end-to-end demo

Banned claims regardless of outcome: quantum advantage, speed-up, hardware implications from simulation.