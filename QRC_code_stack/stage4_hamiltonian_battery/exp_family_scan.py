"""exp_family_scan.py -- <r> level-spacing diagnostics across the battery.

Two scans, every number printed (R4):
  1. <r> for each of the seven benchmark families (dense form built from
     the SAME PauliTerm lists that make the Trotter circuits - one
     Hamiltonian definition, two consumers);
  2. <r> versus the disorder dial W on the mixed-field Ising chain (the
     ergodic/GOE -> MBL/Poisson crossover).

Usage:
    python exp_family_scan.py            full scan + figure
    python exp_family_scan.py --check    reduced grid, no figure; exit 0
                                         iff the RMT anchors pass (stage-4
                                         promotion gate)

RMT anchors (exact analytic limits, no simulation involved):
    GOE random matrix        <r> -> 0.5307
    Poisson (iid spectrum)   <r> -> 0.3863
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from hamiltonians import (GOE_R, POISSON_R, chaos_dial_scan,
                          dense_from_terms, level_spacing_ratio,
                          mixed_field_ising)
from benchmark_hamiltonians import (spec_mixed_ising_chain, spec_sk_all_to_all,
                                    spec_syk4, spec_tfim_2d, spec_tfim_chain,
                                    spec_xxz_chain, spec_xy_chain)

SEED = 7


def goe_r_sample(dim: int, seed: int) -> float:
    """<r> of one GOE sample (exact-limit anchor as dim -> inf)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim))
    return level_spacing_ratio((A + A.T) / 2)


def poisson_r_sample(dim: int, seed: int) -> float:
    """<r> of an iid (Poissonian) spectrum."""
    rng = np.random.default_rng(seed)
    return level_spacing_ratio(np.diag(np.sort(rng.uniform(size=dim))))


def rmt_anchors(dim: int = 2000, n_avg: int = 4, tol: float = 0.012) -> bool:
    """The stage-4 gate: sampled GOE / Poisson <r> hit the analytic values.

    dim=2000, n_avg=4 gives ~4000 r-samples per ensemble; the sampling
    std of the mean is ~0.004, so tol=0.012 is a 3-sigma gate. Not
    reduced in --check mode: a noisy anchor is not an anchor.
    """
    goe = np.mean([goe_r_sample(dim, SEED + i) for i in range(n_avg)])
    poi = np.mean([poisson_r_sample(dim, SEED + i) for i in range(n_avg)])
    ok_goe = abs(goe - GOE_R) < tol
    ok_poi = abs(poi - POISSON_R) < tol
    print(f"anchor GOE     <r> = {goe:.4f}  (target {GOE_R})  "
          f"[{'PASS' if ok_goe else 'FAIL'}]")
    print(f"anchor Poisson <r> = {poi:.4f}  (target {POISSON_R})  "
          f"[{'PASS' if ok_poi else 'FAIL'}]")
    return ok_goe and ok_poi


def family_table(n_small: int) -> None:
    """<r> for each benchmark family at exact-diagonalisable size."""
    # sizes reduced from the circuit benchmark's defaults so ED stays cheap;
    # SYK4 is fixed at 4 qubits by its 8-Majorana construction
    specs = [
        spec_tfim_chain(n=n_small),
        spec_mixed_ising_chain(n=n_small),
        spec_xxz_chain(n=n_small),
        spec_xy_chain(n=n_small),
        spec_tfim_2d(rows=3, cols=3),
        spec_sk_all_to_all(n=min(n_small, 8), seed=SEED),
        spec_syk4(),
    ]
    print("\nfamily <r> table (dense ED from the benchmark PauliTerm lists):")
    print("  CAVEAT: clean uniform chains keep parity/reflection (and XXZ a")
    print("  U(1)) symmetry; mixed sectors bias <r> toward Poisson. Values")
    print("  here are descriptive, NOT gated - the gated crossover is the")
    print("  disorder scan below, where W breaks the symmetries.")
    print(f"  {'family':12s} {'n':>3s} {'<r>':>7s}   nearest reference")
    for spec in specs:
        H = dense_from_terms(spec["terms"], spec["n"])
        r = level_spacing_ratio(H)
        near = ("GOE" if abs(r - GOE_R) < abs(r - POISSON_R) else "Poisson")
        print(f"  {spec['name']:12s} {spec['n']:3d} {r:7.4f}   ~{near}")


def disorder_scan(n: int, Ws, n_real: int, make_figure: bool) -> np.ndarray:
    """<r> vs W. Grid starts at W > 0: the clean (W=0) chain keeps its
    reflection symmetry and mixes sectors, suppressing <r> spuriously."""
    print(f"\ndisorder-dial scan: mixed-field Ising chain n={n}, "
          f"{n_real} realisations per W")
    rs = chaos_dial_scan(n, Ws, n_realizations=n_real, base_seed=SEED)
    print(f"  {'W':>6s} {'<r>':>7s}")
    for W, r in zip(Ws, rs):
        print(f"  {W:6.2f} {r:7.4f}")
    print(f"  references: GOE {GOE_R}, Poisson {POISSON_R}")
    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(Ws, rs, "o-", label=f"n={n} chain")
        ax.axhline(GOE_R, ls="--", c="tab:red", label=f"GOE {GOE_R}")
        ax.axhline(POISSON_R, ls="--", c="tab:blue",
                   label=f"Poisson {POISSON_R}")
        ax.set_xlabel("disorder W")
        ax.set_ylabel(r"$\langle r \rangle$")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("family_scan.png", dpi=200)
        print("wrote family_scan.png")
    return rs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="reduced grid, no figure; anchors + crossover "
                         "decide the exit code")
    args = ap.parse_args()

    print(f"exp_family_scan config: seed={SEED} check={args.check}")

    # gate 1: RMT reference anchors (full statistics in BOTH modes)
    anchors_ok = rmt_anchors()

    # gate 2: the physical crossover - weak disorder lands on the GOE
    # side, strong disorder on the Poisson side
    if args.check:
        family_table(n_small=8)
        rs = disorder_scan(8, np.array([1.0, 12.0]), n_real=6,
                           make_figure=False)
    else:
        family_table(n_small=10)
        rs = disorder_scan(10, np.array([0.5, 1.0, 2.0, 4.0, 6.0, 9.0,
                                         12.0, 14.0]), n_real=8,
                           make_figure=True)
    r_weak, r_strong = rs[np.argmin(np.abs(rs - GOE_R))], rs[-1]
    weak_ok = abs(rs[0] - GOE_R) < abs(rs[0] - POISSON_R)
    strong_ok = abs(rs[-1] - POISSON_R) < abs(rs[-1] - GOE_R)
    print(f"\ncrossover gate: W={1.0 if args.check else 0.5} on GOE side "
          f"[{'PASS' if weak_ok else 'FAIL'}], strong-W on Poisson side "
          f"[{'PASS' if strong_ok else 'FAIL'}]")

    ok = anchors_ok and weak_ok and strong_ok
    print(f"\nexit: {'PASS' if ok else 'FAIL'} "
          "(RMT anchors + disorder crossover gate)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
