"""thresholds.py -- cost-sensitive alert thresholds (Phase 10).

Expected-cost minimisation, NOT F1 maximisation: each persona supplies a
cost matrix (miss cost, false-alarm cost per event) and the optimiser
picks the probability threshold minimising expected cost on the
validation span. Decision-layer value is REPORTED IN COST UNITS next to
F1 (Phase-10 acceptance).

Persona defaults (Part XI personas table; cost units arbitrary but
consistent within a persona -- calibrate per contract):
    microgrid_operator   miss=10, false_alarm=1   (diesel spin-up vs
                                                   battery cycling)
    utility_vpp          miss=6,  false_alarm=2   (reserve procurement)
    trader               miss=4,  false_alarm=3   (imbalance vs position)
"""

from __future__ import annotations

import numpy as np

PERSONA_COSTS = {
    "microgrid_operator": {"miss": 10.0, "false_alarm": 1.0},
    "utility_vpp": {"miss": 6.0, "false_alarm": 2.0},
    "trader": {"miss": 4.0, "false_alarm": 3.0},
}


def expected_cost(p_event: np.ndarray, event: np.ndarray, thr: float,
                  miss: float, false_alarm: float) -> float:
    fire = p_event >= thr
    n_miss = int(np.sum(event & ~fire))
    n_fa = int(np.sum(~event & fire))
    return (miss * n_miss + false_alarm * n_fa) / max(len(event), 1)


def f1(p_event: np.ndarray, event: np.ndarray, thr: float) -> float:
    fire = p_event >= thr
    tp = int(np.sum(event & fire))
    fp = int(np.sum(~event & fire))
    fn = int(np.sum(event & ~fire))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def optimise_threshold(p_val: np.ndarray, event_val: np.ndarray,
                       persona: str) -> dict:
    """Grid over thresholds on the VALIDATION span; returns the chosen
    threshold with its cost AND F1 (both reported, cost decides)."""
    c = PERSONA_COSTS[persona]
    grid = np.linspace(0.02, 0.98, 49)
    costs = [expected_cost(p_val, event_val, t, **c) for t in grid]
    i = int(np.argmin(costs))
    return {"persona": persona, "threshold": float(grid[i]),
            "expected_cost": float(costs[i]),
            "f1_at_threshold": f1(p_val, event_val, grid[i]),
            "cost_matrix": c,
            "cost_all_silent": expected_cost(p_val, event_val, 1.1, **c),
            "cost_all_fire": expected_cost(p_val, event_val, -0.1, **c)}


def isotonic_recalibrate(p_val: np.ndarray, event_val: np.ndarray):
    """Pool-adjacent-violators isotonic fit of P(event | p). Returns a
    callable recalibrator (piecewise-constant, monotone)."""
    order = np.argsort(p_val)
    x, y = np.asarray(p_val)[order], np.asarray(event_val,
                                                dtype=float)[order]
    w = np.ones_like(y)
    vals, wts, idx = list(y), list(w), [[i] for i in range(len(y))]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1]:
            tot = wts[i] + wts[i + 1]
            merged = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tot
            vals[i:i + 2] = [merged]
            wts[i:i + 2] = [tot]
            idx[i:i + 2] = [idx[i] + idx[i + 1]]
            i = max(i - 1, 0)
        else:
            i += 1
    knots_x = np.array([x[block[-1]] for block in idx])
    knots_y = np.array(vals)

    def recal(p):
        j = np.searchsorted(knots_x, np.asarray(p), side="left")
        return knots_y[np.clip(j, 0, len(knots_y) - 1)]
    return recal


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    event = rng.uniform(size=2000) < 0.15
    p = np.clip(0.15 + 0.5 * (event.astype(float) - 0.15)
                + 0.2 * rng.normal(size=2000), 0, 1)
    out = optimise_threshold(p, event, "microgrid_operator")
    print({k: (round(v, 3) if isinstance(v, float) else v)
           for k, v in out.items()})
    assert out["expected_cost"] <= out["cost_all_silent"]
    assert out["expected_cost"] <= out["cost_all_fire"]
    # hand case: miss-heavy persona fires at a LOWER threshold than a
    # false-alarm-averse one
    thr_mg = out["threshold"]
    thr_tr = optimise_threshold(p, event, "trader")["threshold"]
    print(f"microgrid thr {thr_mg:.2f} <= trader thr {thr_tr:.2f}: "
          f"{thr_mg <= thr_tr}")
    assert thr_mg <= thr_tr
    recal = isotonic_recalibrate(p, event)
    pr = recal(np.array([0.1, 0.5, 0.9]))
    assert np.all(np.diff(pr) >= -1e-12)
    print("thresholds self-test PASS")
