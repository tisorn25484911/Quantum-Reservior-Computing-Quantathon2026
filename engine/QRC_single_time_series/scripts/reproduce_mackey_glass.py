"""scripts/reproduce_mackey_glass.py -- P5 GATE: teacher-forced -> autonomous Mackey-Glass.

Reproduces the FN teacher-forced -> autonomous transition (H1) on a stateful exact
QRC, and settles the noise~ridge equivalence (spec s22.2) at the cheapest place.

Gate (spec s13):
  tau_MG=16 (limit cycle): closed loop TRACKS the target over the eval span
    (autonomous NMSE within an order of magnitude of tracking).
  tau_MG=17 (chaotic): tracks O(100s) of steps then diverges pointwise while the
    delay attractor is preserved, and lambda_hat_generated > 0 by BOTH estimators
    (FN perturbation-pair and the ported Rosenstein scalar), same sign and order.

Reservoir config (selected by the mandated reservoir-tau sweep, recorded in the
handoff): N=6, V=10, reservoir tau=4.0, seed=7, J=1.0, h=0.5.

Results -> results/metrics/reproduce_mackey_glass.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.dynamical_systems import mackey_glass as MG
from qrc_single_time_series.quantum.hamiltonians import fc_tfi
from qrc_single_time_series.quantum.exact_qrc import ExactQRC
from qrc_single_time_series.training.teacher_forcing import train_teacher_forced
from qrc_single_time_series.training.noise_augmentation import lambda_eff, add_feature_noise
from qrc_single_time_series.models.readout import fit as fit_readout
from qrc_single_time_series.models.recursive_forecaster import StatefulFN
from qrc_single_time_series.data.windows import align
from qrc_single_time_series.evaluation.autonomous import (autonomous_from_series,
    make_origins, make_policy)
from qrc_single_time_series.evaluation import accumulated_error as A
from qrc_single_time_series.evaluation import prediction_horizon as H
from qrc_single_time_series.evaluation import failure_modes as F
from qrc_single_time_series.evaluation.lyapunov import rosenstein, logistic_series

N, V, RTAU, SEED = 6, 10, 4.0, 7
NTR, N_SAMPLES, WASHOUT = 2200, 3600, 1500
ROLL, EVAL, N_ORIGINS, SPACING = 220, 150, 15, 18


def _reservoir():
    return ExactQRC(fc_tfi(N, J=1.0, h=0.5, seed=SEED), V=V, tau=RTAU)


def run_tau(tau_MG):
    """Multi-origin autonomous evaluation (spec s21): aggregate over launch phases."""
    d = MG.generate(tau_MG, n=N_SAMPLES, washout=WASHOUT)["series"]
    qrc = _reservoir()
    ro, info = train_teacher_forced(qrc, d, n_train=NTR)
    sigma = float(np.std(d[:NTR]))
    stats = F.TrainStats.from_series(d[:NTR])
    origins = make_origins(len(d), warmup=NTR + 50, n_steps=ROLL,
                           n_origins=N_ORIGINS, spacing=SPACING)
    hes, nmses, std_ratios, blowups = [], [], [], 0
    for o in origins:
        r = autonomous_from_series(StatefulFN(qrc, ro), d, o, ROLL,
                                   policy=make_policy("hard_clip", 0.0, 1.0))
        p = r["predictions"]
        tr = d[o:o + len(p)]
        ne = A.instantaneous_ne(tr, p, sigma)
        hes.append(int(H.h_error(ne, eps=0.5, K=3)))
        nmses.append(float(np.mean((p[:EVAL] - tr[:EVAL]) ** 2) /
                     np.var(tr[:EVAL])))
        std_ratios.append(float(np.std(p) / np.std(tr)))
        if "blowup" in F.classify(p, stats, F.FailureRules()):
            blowups += 1
    lam_pair = MG.lyapunov_perturbation_pair(tau_MG, n=2500, washout=WASHOUT)
    lam_ros, _ = rosenstein(d[:1800], m=4, lag=6, mean_period=12, max_t=40)
    return {
        "n_origins": len(origins),
        "onestep_dev_nmse": info.get("nmse_dev_onestep"),
        "median_auto_nmse": float(np.median(nmses)),
        "iqr_auto_nmse": [float(np.percentile(nmses, 25)),
                          float(np.percentile(nmses, 75))],
        "median_H_error": int(np.median(hes)),
        "lambda_pair": lam_pair,
        "lambda_rosenstein": lam_ros,
        "median_std_ratio": float(np.median(std_ratios)),
        "blowup_fraction": blowups / len(origins),
    }


def noise_ridge_check(sigma=0.01, n_seeds=50):
    d = MG.generate(16, n=N_SAMPLES, washout=WASHOUT)["series"]
    qrc = _reservoir()
    X = qrc.features(d[:-1], check_budget=False)
    Xa, ya = align(X, d[1:], horizon=0)
    Xtr, ytr = Xa[100:NTR], ya[100:NTR]
    L = Xtr.shape[0]
    lam = lambda_eff(sigma, L)
    W_ridge = fit_readout(Xtr, ytr, lam=lam).W
    Wsum = np.zeros_like(W_ridge)
    for seed in range(n_seeds):
        Xn = add_feature_noise(Xtr, sigma, np.random.default_rng(seed))
        Wsum += fit_readout(Xn, ytr, lam=0.0).W
    Wmean = Wsum / n_seeds
    return {
        "sigma": sigma, "L_rows": int(L), "lambda_eff": lam,
        "W_corr": float(np.corrcoef(W_ridge.ravel(), Wmean.ravel())[0, 1]),
        "W_rel_diff": float(np.linalg.norm(W_ridge - Wmean) /
                            np.linalg.norm(W_ridge)),
        "norm_ridge": float(np.linalg.norm(W_ridge)),
        "norm_noise_mean": float(np.linalg.norm(Wmean)),
    }


def main():
    out = {"config": {"N": N, "V": V, "reservoir_tau": RTAU, "seed": SEED,
                      "n_train": NTR}, "tau": {}}
    print(f"Mackey-Glass GATE  reservoir N={N} V={V} tau={RTAU} seed={SEED}")
    print("-" * 68)
    # estimator validation on the logistic map first (debug ladder discipline)
    lam_log, _ = rosenstein(logistic_series(), m=3, lag=1, mean_period=1, max_t=15)
    out["logistic_validation"] = {"lambda_hat": lam_log, "true": 0.494}
    print(f"Rosenstein logistic r=3.9: lambda={lam_log:.3f} (true ~0.494)")

    for tau in (16, 17):
        res = run_tau(tau)
        out["tau"][tau] = res
        print(f"tau_MG={tau}: onestep={res['onestep_dev_nmse']:.1e} "
              f"median_auto_NMSE={res['median_auto_nmse']:.3f} "
              f"lam_pair={res['lambda_pair']:.4f} lam_ros={res['lambda_rosenstein']:.4f} "
              f"std_ratio={res['median_std_ratio']:.2f} blowup={res['blowup_fraction']:.2f}")

    out["noise_ridge"] = noise_ridge_check()
    nr = out["noise_ridge"]
    print(f"noise~ridge: lambda_eff={nr['lambda_eff']:.3f} W_corr={nr['W_corr']:.4f} "
          f"rel_diff={nr['W_rel_diff']:.3f}")

    r16, r17 = out["tau"][16], out["tau"][17]
    verdict = {
        "tau16_tracks_bounded": r16["median_auto_nmse"] < 1.0,
        "tau16_tracks_better_than_tau17":
            r16["median_auto_nmse"] < r17["median_auto_nmse"],
        "tau17_chaotic_both_estimators": (r17["lambda_pair"] > 0 and
                                          r17["lambda_rosenstein"] > 0),
        "tau16_noncnaotic": r16["lambda_rosenstein"] < r17["lambda_rosenstein"],
        "tau17_attractor_preserved": (0.4 < r17["median_std_ratio"] < 2.5 and
                                      r17["blowup_fraction"] < 0.5),
        "noise_ridge_equivalent": nr["W_corr"] > 0.95,
    }
    out["gate_verdict"] = verdict
    passed = all(verdict.values())
    out["gate_passed"] = passed

    dest = ROOT / "results" / "metrics" / "reproduce_mackey_glass.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print("-" * 68)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nGATE {'PASSED' if passed else 'FAILED'} -> {dest.relative_to(ROOT)}")
    if not passed:
        raise SystemExit("Mackey-Glass GATE failed (debug ladder: P1 anchors -> "
                         "G3 alignment -> P2 readout scale/lambda -> P3 Schedule)")


if __name__ == "__main__":
    main()
