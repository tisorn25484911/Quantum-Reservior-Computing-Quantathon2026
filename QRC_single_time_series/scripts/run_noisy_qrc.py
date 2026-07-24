"""scripts/run_noisy_qrc.py -- P10: feature-space diagnostics, exact vs finite-shot.

Quantifies how finite sampling deforms the reservoir feature matrix relative to the
exact (noiseless) ceiling, across the preregistered shot grid, on both a controlled
system (Mackey-Glass tau=17) and an observational index (ENSO). Emits the spec-s27
feature-space dashboard (bias/variance/RMSE vs exact, cosine, covariance distance,
effective rank 1/HHI, condition number) per shot count.

The SNR ceiling caveat (spec): open-loop feature RMSE understates closed-loop damage;
these dashboards are diagnostic, the horizon-vs-shots curves (compare_shot_budgets)
are the decision output. Results -> results/metrics/noisy_feature_space.json.
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
from qrc_single_time_series.evaluation import diagnostics as D

SHOT_GRID = [128, 256, 512, 1024, 2048, 4096, 8192]


def _enso():
    s = load_raw("enso")
    cut = split(len(s))["cut"]
    anom, _ = anomaly(s.month, s.values, cut)
    scale, _ = fit_scaler(anom, cut)
    return scale(anom)


def dashboard(series, label, n_qubits=5, V=6, tau=2.0, seed=7):
    qrc = ExactQRC(fc_tfi(n_qubits, J=1.0, h=0.5, seed=seed), V=V, tau=tau)
    Xe = qrc.features(series, check_budget=False)
    rows = {}
    for S in SHOT_GRID:
        rng = np.random.default_rng(100 + S)
        Xn = emulate_shots(Xe, S, rng)
        rows[str(S)] = D.feature_space_report(Xe, Xn)
    return {"label": label, "n": len(series),
            "eff_rank_exact": D.effective_rank_hhi(Xe),
            "explained_variance_top5": D.explained_variance(Xe, 5),
            "by_shots": rows}


def main():
    out = {}
    print("Feature-space dashboard: exact vs finite-shot (spec s27)")
    print("=" * 62)
    for label, series in (("mackey_glass_tau17",
                           MG.generate(17, n=1500, washout=800)["series"]),
                          ("enso", _enso())):
        d = dashboard(series, label)
        out[label] = d
        print(f"\n{label}: eff_rank(exact 1/HHI)={d['eff_rank_exact']:.2f}")
        print(f"   {'shots':>6} {'rmse':>7} {'cos':>7} {'effrank':>8} {'cond':>9}")
        for S in SHOT_GRID:
            r = d["by_shots"][str(S)]
            print(f"   {S:>6} {r['rmse']:>7.4f} {r['cosine_rows']:>7.4f} "
                  f"{r['eff_rank_noisy']:>8.2f} {r['condition_noisy']:>9.1f}")

    dest = ROOT / "results" / "metrics" / "noisy_feature_space.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
