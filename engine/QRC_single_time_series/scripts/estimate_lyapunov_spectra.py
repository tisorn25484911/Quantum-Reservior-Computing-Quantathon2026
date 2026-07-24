"""scripts/estimate_lyapunov_spectra.py -- P8 dynamical audits (G8-1/2/3).

Computes, IN-REPO, the largest Lyapunov exponent for every controlled system and
cross-validates the two estimators (Benettin QR on the exact Jacobian vs the scalar
Rosenstein method) on Lorenz, so that any later learned-map lambda is only reported
where the machinery is trusted (spec s20.2). Settles:

  * G8-1  Lorenz beta = 8/3 vs printed 3/8 -- which reproduces lambda ~ 0.906.
  * G8-2/3 Vallis 'ENSO' ODE lambda computed here (NOT the 0.05-0.1 observational
    figure), with the time-unit note.
  * Mackey-Glass tau=16 vs 17 largest exponent (sign) as a consistency check.

Results -> results/metrics/lyapunov_spectra.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.dynamical_systems import lorenz63 as LZ
from qrc_single_time_series.dynamical_systems import enso_ode as EN
from qrc_single_time_series.dynamical_systems import mackey_glass as MG
from qrc_single_time_series.evaluation.lyapunov import rosenstein


def main():
    out = {}

    # --- Lorenz beta audit (G8-1) + estimator cross-validation ---------------
    print("Lorenz-63 beta audit (Benettin QR)")
    ba = LZ.beta_audit(dt=0.01, n_steps=20000, washout=2000)
    out["lorenz_beta_audit"] = ba
    print(f"  beta=8/3 lambda_max={ba['beta_8_3']['lambda_max']:.4f}  "
          f"beta=3/8 lambda_max={ba['beta_3_8']['lambda_max']:.4f}  "
          f"-> {ba['reproduces_0.906']} reproduces ~0.906")

    g = LZ.generate(n=8000, dt=0.02, washout=2000, beta=LZ.BETA_STANDARD)
    lam_ros, _ = rosenstein(g["trajectory"][:, 0], m=5, lag=8, mean_period=15,
                            max_t=40)
    lam_ros_per_time = lam_ros / g["sample_dt"]                # per unit time
    out["lorenz_estimator_crossval"] = {
        "benettin_per_time": ba["beta_8_3"]["lambda_max"],
        "rosenstein_per_time": lam_ros_per_time,
        "agree_sign": bool(lam_ros_per_time > 0),
    }
    print(f"  estimator cross-val: Benettin={ba['beta_8_3']['lambda_max']:.3f}  "
          f"Rosenstein={lam_ros_per_time:.3f} (per time)")

    # --- Vallis ENSO ODE audit (G8-2/3) --------------------------------------
    print("Vallis 'ENSO' ODE audit (Benettin QR)")
    va = EN.audit(dt=0.005, n_steps=40000, washout=4000)
    out["vallis_enso_audit"] = va
    print(f"  lambda_max={va['lambda_max']:.4f}  chaotic={va['chaotic']}  "
          f"LT={va['lyapunov_time']:.2f}  (0.05-0.1 NOT reused)")

    # --- Mackey-Glass sign check ---------------------------------------------
    out["mackey_glass"] = {}
    for tau in (16, 17):
        lam = MG.lyapunov_perturbation_pair(tau, n=2500, washout=2000)
        out["mackey_glass"][f"tau_{tau}"] = {"lambda_pair_per_sample": lam,
                                             "chaotic": bool(lam > 1e-3)}
    print(f"  Mackey-Glass tau16="
          f"{out['mackey_glass']['tau_16']['lambda_pair_per_sample']:.4f}"
          f"  tau17={out['mackey_glass']['tau_17']['lambda_pair_per_sample']:.4f}")

    dest = ROOT / "results" / "metrics" / "lyapunov_spectra.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
