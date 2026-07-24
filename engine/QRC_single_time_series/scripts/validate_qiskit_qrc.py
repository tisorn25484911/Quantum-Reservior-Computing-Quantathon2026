"""scripts/validate_qiskit_qrc.py -- P3 validation ladder (spec s12), one command.

Runs the exact<->Trotter<->Qiskit<->noisy ladder on FN reservoirs at N in {3,5,8}
and prints/stores the parity numbers:

  L1 exact dense e^{-iHt}   vs L2 exact matrix Trotter -> Trotter error vs kappa
  L2 matrix Trotter         vs L3 Aer density_matrix   -> structural parity (~1e-10)
  L3 ideal                  vs L4 finite-shot emulation -> binomial spread
  L4 ideal                  vs L5 synthetic noise        -> attributable deviation

Results -> results/metrics/validate_qiskit_qrc.json. Pinned: qiskit 2.5.0,
qiskit-aer 0.17.2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.quantum.hamiltonians import fc_tfi
from qrc_single_time_series.quantum.exact_qrc import ExactQRC
from qrc_single_time_series.quantum.exact_trotter_qrc import ExactTrotterQRC
from qrc_single_time_series.quantum.qiskit_qrc import QiskitReservoir, shot_emulate
from qrc_single_time_series.quantum.noise_models import build_synthetic
from qrc_single_time_series.quantum.noisy_qiskit_qrc import NoisyQiskitReservoir

NS = [3, 5, 8]
V, TAU, KAPPA = 3, 2.0, 2


def main():
    out = {"pins": {"qiskit": "2.5.0", "qiskit_aer": "0.17.2"},
           "config": {"V": V, "tau": TAU, "kappa": KAPPA, "seed": 7}, "levels": {}}
    print("QRC validation ladder (FN track)  V=%d tau=%.1f kappa=%d" % (V, TAU, KAPPA))
    print("-" * 66)
    for N in NS:
        res = fc_tfi(N, J=1.0, h=0.5, seed=7)
        s = np.random.default_rng(0).uniform(0, 1, 14)
        l1 = ExactQRC(res, V=V, tau=TAU).features(s, check_budget=False)
        # L1 vs L2 Trotter error vs kappa
        trot = {k: float(np.max(np.abs(l1 - ExactTrotterQRC(res, V=V, tau=TAU, kappa=k)
                .features(s, check_budget=False)))) for k in (1, 4, 16)}
        l2 = ExactTrotterQRC(res, V=V, tau=TAU, kappa=KAPPA).features(s, check_budget=False)
        l3 = QiskitReservoir(res, V=V, tau=TAU, kappa=KAPPA).features(s, check_budget=False)
        l2_l3 = float(np.max(np.abs(l2 - l3)))
        # L3 vs L4 finite shots
        rng = np.random.default_rng(101)
        z = 2 * l3[:, :-1] - 1
        l4 = 0.5 * (1 + shot_emulate(z, 1024, rng))
        l3_l4 = float(np.std((l4 - l3[:, :-1])))
        # L4 vs L5 synthetic noise
        nm = build_synthetic({"depol_1q": 1e-3, "depol_2q": 7e-3, "readout_p": 0.015})
        l5 = NoisyQiskitReservoir(res, nm, V=V, tau=TAU, kappa=KAPPA).features(
            s, check_budget=False)
        l4_l5 = float(np.max(np.abs(l5 - l3)))
        out["levels"][N] = {"trotter_error": trot, "L2_vs_L3": l2_l3,
                            "L3_vs_L4_shotstd": l3_l4, "L4_vs_L5_noise": l4_l5}
        print(f"N={N}: Trotter(k=1,4,16)={trot[1]:.2e},{trot[4]:.2e},{trot[16]:.2e}  "
              f"L2vsL3={l2_l3:.1e}  L3vsL4std={l3_l4:.3f}  L4vsL5={l4_l5:.3f}")
        assert l2_l3 < 1e-9, f"N={N} L2 vs L3 parity failed"
        assert trot[1] > trot[4] > trot[16], f"N={N} Trotter error not decreasing"

    dest = ROOT / "results" / "metrics" / "validate_qiskit_qrc.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print("-" * 66 + f"\nwrote {dest.relative_to(ROOT)}\nladder parity PASSED (N in {NS})")


if __name__ == "__main__":
    main()
