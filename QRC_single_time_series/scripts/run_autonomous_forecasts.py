"""scripts/run_autonomous_forecasts.py -- P8: controlled-system autonomous campaign.

Multi-origin closed-loop forecasting on the three controlled systems (Mackey-Glass
tau=17, Lorenz-63 x, Vallis 'ENSO' Te), with the stateful FN reservoir driven
through the ONE rollout engine (G5). Reports, per system:

  * H_error / H_skill / H_effective distributions (steps AND Lyapunov times -- LT
    from the P8 audit, so numbers are mutually interpretable across systems,
    spec s20.1);
  * dynamical-fidelity distances (invariant measure / spectrum / ACF / recurrence)
    of the autonomous trajectory vs the truth (spec s18.1); and
  * a horizon-vs-shots slope on Mackey-Glass: H_effective = a + b ln S, with b
    compared to the chaotic prediction b ~ 1/(2 lambda dt) (spec s23), after a
    signal-to-noise ceiling check.

This is an EXPENSIVE grid (spec s31): `make grid-autonomous`, never `make all`.
Results -> results/metrics/autonomous_campaign.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.dynamical_systems import mackey_glass as MG
from qrc_single_time_series.dynamical_systems import lorenz63 as LZ
from qrc_single_time_series.dynamical_systems import enso_ode as EN
from qrc_single_time_series.quantum.hamiltonians import fc_tfi
from qrc_single_time_series.quantum.exact_qrc import ExactQRC
from qrc_single_time_series.training.teacher_forcing import train_teacher_forced
from qrc_single_time_series.models.recursive_forecaster import StatefulFN
from qrc_single_time_series.models import autoregression as AR
from qrc_single_time_series.evaluation.autonomous import (autonomous_from_series,
    make_origins, make_policy)
from qrc_single_time_series.evaluation import accumulated_error as A
from qrc_single_time_series.evaluation import prediction_horizon as H
from qrc_single_time_series.evaluation import dynamical_fidelity as DF


def _systems():
    """Scaled univariate series + per-sample lambda_max for each controlled system."""
    mg = MG.generate(17, n=3000, washout=1500)["series"]
    lam_mg = MG.lyapunov_perturbation_pair(17, n=2500, washout=1500)     # per sample
    lz = LZ.generate(n=4000, dt=0.02, washout=2000)
    lam_lz = LZ.lyapunov_largest(dt=0.02, n_steps=15000, washout=2000) * lz["sample_dt"]
    en = EN.generate(n=4000, dt=0.01, washout=4000, component=1)
    lam_en = EN.lyapunov_largest(dt=0.01, n_steps=25000, washout=4000) * en["sample_dt"]
    return {
        "mackey_glass_tau17": (mg, lam_mg),
        "lorenz_x": (lz["series"], lam_lz),
        "vallis_enso_Te": (en["series"], lam_en),
    }


def campaign(series, lam_per_step, n_qubits=5, V=6, tau=2.0, seed=7,
             n_origins=8, horizon=140, eps=0.5, K=3):
    series = np.asarray(series, dtype=float)
    n = len(series)
    n_train = int(0.65 * n)
    qrc = ExactQRC(fc_tfi(n_qubits, J=1.0, h=0.5, seed=seed), V=V, tau=tau)
    ro, info = train_teacher_forced(qrc, series, n_train=n_train)
    sigma = float(np.std(series[:n_train]))
    origins = make_origins(n, warmup=n_train + 20, n_steps=horizon,
                           n_origins=n_origins, spacing=15)

    hes, herr, hskl, fid = [], [], [], []
    for o in origins:
        r = autonomous_from_series(StatefulFN(qrc, ro), series, o, horizon,
                                   policy=make_policy("hard_clip", 0.0, 1.0))
        p = r["predictions"]
        truth = series[o:o + len(p)]
        ne = A.instantaneous_ne(truth, p, sigma)
        he = H.h_error(ne, eps, K)
        # skill vs persistence through the SAME engine
        b = autonomous_from_series(AR.RidgeAR(1, np.array([1.0]), 0.0, 0.0),
                                   series, o, horizon)
        m = min(len(p), len(b["predictions"]))
        hs = H.h_skill(A.cnrmse(truth[:m], p[:m], sigma),
                       A.cnrmse(series[o:o + m], b["predictions"][:m], sigma), K)
        hes.append(min(he, hs))
        herr.append(he)
        hskl.append(hs)
        if len(p) > 40:
            fid.append(DF.fidelity_report(truth, p))

    lt = (1.0 / lam_per_step) if lam_per_step > 1e-6 else None
    med_fid = {k: float(np.median([f[k] for f in fid])) for k in fid[0]} if fid else {}
    return {
        "n_origins": len(origins),
        "n_train": n_train,
        "onestep_dev_nmse": info.get("nmse_dev_onestep"),
        "lambda_per_step": lam_per_step,
        "lyapunov_time_steps": lt,
        "H_error": {"median": float(np.median(herr)),
                    "iqr": [float(np.percentile(herr, 25)),
                            float(np.percentile(herr, 75))]},
        "H_skill_vs_persistence": {"median": float(np.median(hskl))},
        "H_effective": {"median": float(np.median(hes)),
                        "iqr": [float(np.percentile(hes, 25)),
                                float(np.percentile(hes, 75))],
                        "in_lyapunov_times": (float(np.median(hes)) / lt) if lt else None},
        "dynamical_fidelity_median": med_fid,
    }


def _shot_features_adapter(qrc, ro, shots, seed):
    """StatefulFN whose per-step features carry emulated finite-shot noise.

    Uses the gotchas shot formula on z=2x'-1: sigma(z)=2 sqrt(p(1-p)/S), p=(1+z)/2,
    emulate clip(z + xi sigma, -1, 1); the readout then sees the noisy x'.
    """
    rng = np.random.default_rng(seed)

    class _Shot(StatefulFN):
        def step(self, rho, u):
            row, rho2 = self.qrc.step(rho, float(u))
            z = 2.0 * row - 1.0
            p = 0.5 * (1.0 + z)
            sig = 2.0 * np.sqrt(np.clip(p * (1 - p), 0, None) / shots)
            zc = np.clip(z + rng.standard_normal(z.shape) * sig, -1, 1)
            feat = np.concatenate([0.5 * (1 + zc), [1.0]])
            return rho2, float(self.readout.predict(feat[None, :])[0])

    return _Shot(qrc, ro)


def shot_slope(series, lam_per_step, shots_grid=(128, 512, 2048, 8192),
               n_qubits=5, V=6, tau=2.0, seed=7, n_origins=6, horizon=120,
               eps=0.5, K=3):
    """H_effective vs ln S; fit slope b and compare to b~1/(2 lambda) (per step)."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    n_train = int(0.65 * n)
    qrc = ExactQRC(fc_tfi(n_qubits, J=1.0, h=0.5, seed=seed), V=V, tau=tau)
    ro, _ = train_teacher_forced(qrc, series, n_train=n_train)
    sigma = float(np.std(series[:n_train]))
    origins = make_origins(n, warmup=n_train + 20, n_steps=horizon,
                           n_origins=n_origins, spacing=15)

    # signal-to-noise ceiling check: noiseless (exact) horizon first
    def median_he(model_factory):
        hs = []
        for o in origins:
            r = autonomous_from_series(model_factory(), series, o, horizon,
                                       policy=make_policy("hard_clip", 0.0, 1.0))
            p = r["predictions"]
            hs.append(H.h_error(A.instantaneous_ne(series[o:o + len(p)], p, sigma),
                                eps, K))
        return float(np.median(hs))

    exact_he = median_he(lambda: StatefulFN(qrc, ro))
    curve = {}
    for S in shots_grid:
        curve[int(S)] = median_he(lambda: _shot_features_adapter(qrc, ro, S, seed + S))
    lnS = np.log(list(shots_grid))
    hev = np.array([curve[int(S)] for S in shots_grid])
    b_fit = float(np.polyfit(lnS, hev, 1)[0]) if len(shots_grid) > 1 else None
    b_theory = 1.0 / (2.0 * lam_per_step) if lam_per_step > 1e-6 else None
    # The slope is only interpretable if the closed loop survives >0 steps at every
    # shot count AND the curve actually varies; otherwise the study "measures
    # nothing" (spec s23 precondition) -- reported honestly, never forced.
    measurable = bool(hev.min() > 0 and hev.max() > hev.min())
    return {
        "exact_median_H_error": exact_he,
        "shots_curve": {str(k): v for k, v in curve.items()},
        "slope_b_fit": b_fit,
        "slope_b_theory_half_over_lambda": b_theory,
        "slope_measurable": measurable,
        "note": ("shots collapse the closed loop at every tested count for this "
                 "under-tuned config; the H~a+b lnS law is not measurable here -- "
                 "needs a stronger-signal config + SVD-truncation mitigation (P10)")
        if not measurable else "slope fit is interpretable",
    }


def main():
    out = {}
    systems = _systems()
    print("Controlled-system autonomous campaign")
    print("=" * 62)
    for name, (series, lam) in systems.items():
        res = campaign(series, lam)
        out[name] = res
        he = res["H_effective"]
        print(f"\n{name}: lambda/step={lam:.4f}  LT={res['lyapunov_time_steps']}")
        print(f"   H_error median={res['H_error']['median']:.0f}  "
              f"H_skill(persist) median={res['H_skill_vs_persistence']['median']:.0f}  "
              f"H_effective median={he['median']:.0f} steps "
              f"({he['in_lyapunov_times']} LT)" if he['in_lyapunov_times']
              else f"   H_effective median={he['median']:.0f} steps")
        if res["dynamical_fidelity_median"]:
            f = res["dynamical_fidelity_median"]
            print(f"   fidelity: invmeas_L1={f['invariant_measure_L1']:.3f} "
                  f"acf_rms={f['acf_rms']:.3f} std_ratio={f['std_ratio']:.2f}")

    print("\nShot-slope study (Mackey-Glass tau=17)")
    mg, lam_mg = systems["mackey_glass_tau17"]
    ss = shot_slope(mg, lam_mg)
    out["shot_slope_mackey_glass"] = ss
    print(f"   exact H_error={ss['exact_median_H_error']:.0f}  "
          f"b_fit={ss['slope_b_fit']}  b_theory(1/2lambda)={ss['slope_b_theory_half_over_lambda']}")
    print(f"   shots curve: {ss['shots_curve']}  slope_measurable={ss['slope_measurable']}")

    dest = ROOT / "results" / "metrics" / "autonomous_campaign.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
