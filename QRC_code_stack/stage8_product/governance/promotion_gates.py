"""promotion_gates.py -- promotion between model versions ONLY through
the pre-registered gates (Phase 12; R6 transposed to production).

Gates are DATA, declared before evaluation, evaluated once per review;
the evaluator is a pure function producing a signed verdict record.
The contractual quantum-optional demotion lives here: if the quantum
delta gate fails, the verdict instructs shipping classical-only and the
quantum track continues as research (Part XI de-risking posture).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


@dataclass
class Gate:
    name: str
    metric: str
    comparator: str          # '>=' | '<=' | '>'
    target: float
    description: str = ""

    def passes(self, value: float) -> bool:
        return {">=": value >= self.target,
                "<=": value <= self.target,
                ">": value > self.target}[self.comparator]


# The Part XI pilot-exit defaults (rewrite per contract BEFORE the pilot)
PILOT_GATES = [
    Gate("ramp_f1_vs_incumbent", "ramp_f1_delta", ">=", 0.05,
         "ramp F1 >= incumbent + 0.05 at the operator's false-alarm budget"),
    Gate("latency_sla", "alert_latency_s", "<=", 60.0,
         "alerts within 60 s of data arrival"),
    Gate("provenance_violations", "n_fabricated_numbers", "<=", 0.0,
         "zero tolerance"),
    Gate("coverage", "coverage_gap_abs", "<=", 0.03,
         "rolling conformal coverage within 3 points of target"),
]

QUANTUM_DELTA_GATE = Gate(
    "quantum_delta", "hybrid_minus_classical_f1", ">", 0.0,
    "quantum feature block must beat its own classical ablation; on "
    "failure ship classical-only (contractual demotion)")


@dataclass
class Verdict:
    passed: bool
    gates: list = field(default_factory=list)
    quantum_demotion: bool = False
    run_id: str = ""
    evaluated_at: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, default=str)


def evaluate(measurements: dict, gates: list[Gate] | None = None,
             run_id: str = "") -> Verdict:
    gates = gates if gates is not None else PILOT_GATES
    rows, ok = [], True
    for g in gates:
        if g.metric not in measurements:
            rows.append({"gate": g.name, "status": "MISSING METRIC"})
            ok = False
            continue
        v = float(measurements[g.metric])
        p = g.passes(v)
        ok &= p
        rows.append({"gate": g.name, "value": v, "target": g.target,
                     "comparator": g.comparator,
                     "status": "PASS" if p else "FAIL"})
    demote = False
    if QUANTUM_DELTA_GATE.metric in measurements:
        qv = float(measurements[QUANTUM_DELTA_GATE.metric])
        demote = not QUANTUM_DELTA_GATE.passes(qv)
        rows.append({"gate": QUANTUM_DELTA_GATE.name, "value": qv,
                     "status": "PASS" if not demote
                     else "FAIL -> ship classical-only"})
    return Verdict(passed=ok, gates=rows, quantum_demotion=demote,
                   run_id=run_id,
                   evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()))


if __name__ == "__main__":
    v = evaluate({"ramp_f1_delta": 0.07, "alert_latency_s": 12.0,
                  "n_fabricated_numbers": 0, "coverage_gap_abs": 0.02,
                  "hybrid_minus_classical_f1": -0.01},
                 run_id="demo#1")
    print(v.to_json())
    assert v.passed and v.quantum_demotion   # gates pass, quantum demoted
    v2 = evaluate({"ramp_f1_delta": 0.01, "alert_latency_s": 12.0,
                   "n_fabricated_numbers": 0, "coverage_gap_abs": 0.02})
    assert not v2.passed
    print("promotion gates self-test PASS (incl. quantum demotion path)")
