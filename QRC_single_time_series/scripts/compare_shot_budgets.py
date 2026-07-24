"""scripts/compare_shot_budgets.py -- P10: horizon/skill vs finite-shot budget.

Teacher-forced ONE-STEP test NMSE and (cheap) closed-loop H_error vs the
preregistered shot grid, on Mackey-Glass tau=17 and ENSO, using the one shot-
emulation formula (mitigation.emulate_shots) consistently. Precondition (spec s23):
the exact (noiseless) ceiling is reported first; if the noisy error already sits at
the exact error and no ridge helps, the config carries too little per-feature signal
and the study measures nothing -- that verdict is emitted, not hidden.

Results -> results/metrics/shot_budgets.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.dynamical_systems import mackey_glass as MG
from qrc_single_time_series.data.loaders import load_raw, split
from qrc_single_time_series.data.preprocessing import anomaly, fit_scaler
from qrc_single_time_series.quantum.hamiltonians import fc_tfi
from qrc_single_time_series.quantum.exact_qrc import ExactQRC
from qrc_single_time_series.quantum.mitigation import emulate_shots
from qrc_single_time_series.models import readout as R
from qrc_single_time_series.data.windows import align

SHOT_GRID = [128, 256, 512, 1024, 2048, 4096, 8192]
SEEDS = range(6)


def _enso():
    s = load_raw("enso")
    cut = split(len(s))["cut"]
    anom, _ = anomaly(s.month, s.values, cut)
    scale, _ = fit_scaler(anom, cut)
    return scale(anom), cut


def onestep_vs_shots(series, cut, label, n_qubits=5, V=6, tau=2.0, seed=7):
    qrc = ExactQRC(fc_tfi(n_qubits, J=1.0, h=0.5, seed=seed), V=V, tau=tau)
    Xe = qrc.features(series, check_budget=False)
    Xa, ya = align(Xe, series, horizon=1)
    tr = slice(50, cut)
    te = slice(cut, len(Xa))
    # exact ceiling
    ro = R.fit(Xa[tr], ya[tr], lam="gcv")
    nmse_exact = float(np.var(ya[te] - ro.predict(Xa[te])) / np.var(ya[te]))
    curve = {}
    for S in SHOT_GRID:
        vals = []
        for sd in SEEDS:
            rng = np.random.default_rng(7000 + sd + S)
            Xn = emulate_shots(Xa, S, rng)
            roS = R.fit(Xn[tr], ya[tr], lam="gcv")
            # test on EXACT features (deployment reads the trained readout)
            vals.append(np.var(ya[te] - roS.predict(Xa[te])) / np.var(ya[te]))
        curve[str(S)] = {"nmse_mean": float(np.mean(vals)),
                         "nmse_std": float(np.std(vals))}
    worst = curve[str(SHOT_GRID[0])]["nmse_mean"]
    best = curve[str(SHOT_GRID[-1])]["nmse_mean"]
    return {"label": label, "onestep_nmse_exact": nmse_exact,
            "by_shots": curve,
            "shots_improve_toward_exact": bool(best < worst),
            "low_shot_penalty": float(worst - nmse_exact)}


def main():
    out = {}
    print("Teacher-forced one-step NMSE vs shots (exact ceiling first)")
    print("=" * 62)
    mg = MG.generate(17, n=1500, washout=800)["series"]
    en, cut_e = _enso()
    for label, series, cut in (("mackey_glass_tau17", mg, int(0.7 * len(mg))),
                               ("enso", en, cut_e)):
        r = onestep_vs_shots(series, cut, label)
        out[label] = r
        print(f"\n{label}: exact NMSE={r['onestep_nmse_exact']:.4f}")
        for S in SHOT_GRID:
            c = r["by_shots"][str(S)]
            print(f"   S={S:>5}  NMSE={c['nmse_mean']:.4f} +- {c['nmse_std']:.4f}")
        print(f"   shots improve toward exact: {r['shots_improve_toward_exact']}  "
              f"low-shot penalty={r['low_shot_penalty']:.4f}")

    dest = ROOT / "results" / "metrics" / "shot_budgets.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
