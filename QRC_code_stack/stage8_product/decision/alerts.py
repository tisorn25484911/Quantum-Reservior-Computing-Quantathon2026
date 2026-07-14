"""alerts.py -- the alert engine: honesty as payload (Phase 10).

Alert objects carry their own provenance and their delta over
persistence, exactly the Part XI schema. Dedup (one live alert per
site/event/horizon bucket), escalation on magnitude jump, and fatigue
caps (weekly false-alarm budget per persona) are enforced here.
The alert schema is IDENTICAL at every rung of the degrade-graceful
ladder (quantum -> classical-only -> smart-persistence), so downgrades
are invisible to integrations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Alert:
    site: str
    event: str                      # "ramp_down" | "ramp_up" | ...
    horizon_min: int
    p_event: float
    magnitude_q10_q50_q90: list
    lead_time_min: int
    regime: str
    conformal_coverage_target: float
    run_id: str
    model_version: str
    baseline_delta: dict            # e.g. {"vs_persistence_F1": +0.11}
    data_kind: str = "real"         # "surrogate" propagates from L1 (!)
    notes: list = field(default_factory=list)

    def validate(self) -> list[str]:
        issues = []
        if not 0.0 <= self.p_event <= 1.0:
            issues.append("p_event outside [0,1]")
        q = self.magnitude_q10_q50_q90
        if len(q) != 3 or not (q[0] <= q[1] <= q[2]):
            issues.append("quantile fan not monotone")
        if not self.run_id:
            issues.append("missing run_id (provenance)")
        if "vs_persistence_F1" not in self.baseline_delta:
            issues.append("missing persistence delta (honesty payload)")
        return issues

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class AlertEngine:
    def __init__(self, weekly_false_alarm_budget: int = 20,
                 escalation_jump: float = 0.15):
        self.budget = weekly_false_alarm_budget
        self.escalation_jump = escalation_jump
        self.live: dict[tuple, Alert] = {}
        self.emitted: list[Alert] = []
        self.suppressed_dupes = 0
        self.suppressed_fatigue = 0
        self.week_fires = 0

    def new_week(self) -> None:
        self.week_fires = 0

    def submit(self, a: Alert) -> str:
        """Returns 'emitted' | 'escalated' | 'dedup' | 'fatigue' |
        'invalid:<why>'."""
        issues = a.validate()
        if issues:
            return "invalid:" + ";".join(issues)
        key = (a.site, a.event, a.horizon_min)
        if key in self.live:
            prev = self.live[key]
            if abs(a.magnitude_q10_q50_q90[1]
                   - prev.magnitude_q10_q50_q90[1]) >= self.escalation_jump:
                a.notes.append("escalation: magnitude jump vs live alert")
                self.live[key] = a
                self.emitted.append(a)
                return "escalated"
            self.suppressed_dupes += 1
            return "dedup"
        if self.week_fires >= self.budget:
            self.suppressed_fatigue += 1
            return "fatigue"
        self.live[key] = a
        self.emitted.append(a)
        self.week_fires += 1
        return "emitted"

    def resolve(self, site: str, event: str, horizon_min: int) -> None:
        self.live.pop((site, event, horizon_min), None)


if __name__ == "__main__":
    eng = AlertEngine(weekly_false_alarm_budget=2)
    base = dict(site="sg-01", event="ramp_down", horizon_min=60,
                p_event=0.83, magnitude_q10_q50_q90=[-0.31, -0.22, -0.12],
                lead_time_min=47, regime="convective_afternoon",
                conformal_coverage_target=0.9,
                run_id="2026-07-14T12:00Z#demo", model_version="hybrid-0.1",
                baseline_delta={"vs_persistence_F1": 0.11},
                data_kind="surrogate")
    print(eng.submit(Alert(**base)))                       # emitted
    print(eng.submit(Alert(**base)))                       # dedup
    esc = dict(base, magnitude_q10_q50_q90=[-0.6, -0.5, -0.4])
    print(eng.submit(Alert(**esc)))                        # escalated
    other = dict(base, event="ramp_up",
                 magnitude_q10_q50_q90=[0.1, 0.2, 0.3])
    print(eng.submit(Alert(**other)))                      # emitted (2nd)
    third = dict(base, horizon_min=120)
    print(eng.submit(Alert(**third)))                      # fatigue
    bad = dict(base, run_id="")
    print(eng.submit(Alert(**bad)))                        # invalid
    assert eng.week_fires == 2 and eng.suppressed_fatigue == 1
    print("alerts self-test PASS")
