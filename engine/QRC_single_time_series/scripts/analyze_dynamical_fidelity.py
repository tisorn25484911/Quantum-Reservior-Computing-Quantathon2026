"""scripts/analyze_dynamical_fidelity.py -- P8: consolidate the dynamical-fidelity table.

Reads the autonomous campaign output (``results/metrics/autonomous_campaign.json``,
produced by ``make grid-autonomous``) and emits a single dynamical-fidelity summary
that separates POINTWISE horizon (H_effective) from DISTRIBUTIONAL/SPECTRAL fidelity
(spec s18.1, acceptance item 16): a model can lose pointwise tracking after a few
Lyapunov times yet still reproduce the attractor's invariant measure, spectrum, ACF
and recurrence structure. Writes results/dynamical_fidelity/summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "metrics" / "autonomous_campaign.json"


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC.relative_to(ROOT)} -- run `make grid-autonomous` first")
    camp = json.loads(SRC.read_text())

    summary = {}
    print("Dynamical fidelity vs pointwise horizon (controlled systems)")
    print("=" * 66)
    print(f"{'system':22s} {'H_eff(LT)':>10s} {'invmeas':>8s} {'spec':>7s} "
          f"{'acf':>7s} {'std_r':>7s}")
    for name, res in camp.items():
        if not isinstance(res, dict) or "H_effective" not in res:
            continue
        he = res["H_effective"]
        f = res.get("dynamical_fidelity_median", {})
        lt = he.get("in_lyapunov_times")
        row = {
            "H_effective_steps": he["median"],
            "H_effective_LT": lt,
            "invariant_measure_L1": f.get("invariant_measure_L1"),
            "spectral_L1": f.get("spectral_L1"),
            "acf_rms": f.get("acf_rms"),
            "std_ratio": f.get("std_ratio"),
            "verdict": ("attractor-faithful" if f.get("invariant_measure_L1", 1) < 0.4
                        and 0.5 < f.get("std_ratio", 0) < 2.0
                        else "pointwise-only / partial"),
        }
        summary[name] = row
        print(f"{name:22s} {(lt or 0):10.2f} {row['invariant_measure_L1'] or 0:8.3f} "
              f"{row['spectral_L1'] or 0:7.3f} {row['acf_rms'] or 0:7.3f} "
              f"{row['std_ratio'] or 0:7.2f}   {row['verdict']}")

    dest = ROOT / "results" / "dynamical_fidelity" / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
