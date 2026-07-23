"""seed_sweep.py -- multi-seed Step-2 kill-test (plan.md Sec.10 seed sweep,
generalised to any dataset).

`run_step2.py` fixes `SEED = 7`. That is fine for `xxz_hx` (deterministic,
no disorder -- see `Hamiltonian_QRC/qrc_core.py::xxz_hx_hamiltonian`), but
`ising`'s couplings are drawn from `rng` and the ESN's weights always are.
plan.md Sec.10 found that reading one ESN draw as "the" classical baseline on
`nino34` reversed the single-seed verdict once 5 seeds were checked. `got_sst`
turns out to be won by `ising`, not `xxz_hx` (see TODO.md) -- so this dataset
needs both random kinds swept, not just the ESN.

Reuses `forecast.py` directly (not a subprocess loop over `run_step2.py`) so
the drive/rollout cost is paid once per seed per kind, same as the original
script.

    python seed_sweep.py --dataset got_sst --horizon 24 --max-points 16394
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "DataBase_Analysis"))

from dataloader import load                          # noqa: E402
from forecast import make_reservoir, fit_readout, skill_vs_horizon  # noqa: E402

SEEDS = (7, 11, 23, 42, 101)
KINDS = ("ising", "xxz_hx", "esn")
MATERIAL = 0.05

# Annual cycle in samples for daily-cadence raw (non-anomaly) SST records --
# see plan.md Sec.6 open-item 5: nino34/nino12 are already anomalies, so their
# climatology baseline degenerates to the mean predictor and is kept only to
# show that degeneracy. got_sst is raw SST, so 365 is a real seasonal-mean
# baseline. It is an approximation (no leap-day correction), which very
# slowly drifts the phase over a multi-decade record; reported, not hidden.
PERIODS = {"nino34": 12, "nino12": 12, "got_sst": 365}


def wins(curve, reference, margin=0.0):
    return [h for h, (a, b) in enumerate(zip(curve, reference), start=1)
            if a < b * (1.0 - margin)]


def as_range(hs, H):
    if not hs:
        return "never"
    if hs == list(range(hs[0], hs[-1] + 1)):
        span = f"h={hs[0]}" if len(hs) == 1 else f"h={hs[0]}-{hs[-1]}"
        return f"all {H}" if len(hs) == H else span
    return f"{len(hs)}/{H} horizons"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="got_sst")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--qubits", type=int, default=5)
    ap.add_argument("--max-points", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    s = load(args.dataset)
    x = np.asarray(s.x, float)
    if len(x) > args.max_points:
        x = x[-args.max_points:]
    H = args.horizon
    period = PERIODS.get(s.key)

    print(f"=== seed sweep: {s.name} ({s.key}) ===")
    print(f"N={len(x)}  H={H}  qubits={args.qubits}  "
          f"seeds={args.seeds}  period={period or '-'}")

    curves = {k: [] for k in KINDS}
    baselines, n_origins, clip_rates = None, None, {k: [] for k in KINDS}
    for seed in args.seeds:
        for kind in KINDS:
            res = make_reservoir(kind, n_qubits=args.qubits, seed=seed)
            ro = fit_readout(x, res, washout=100, train_frac=0.7)
            out = skill_vs_horizon(x, res, ro, H, period=period)
            curves[kind].append(out["nmse"]["reservoir"])
            clip_rates[kind].append(out["clip_rate"])
            if baselines is None:
                baselines = {k: v for k, v in out["nmse"].items()
                             if k != "reservoir"}
                n_origins = out["n_origins"]
        print(f"  seed {seed:4d} done")

    arrs = {k: np.array(v) for k, v in curves.items()}   # (n_seeds, H)
    mean = {k: v.mean(axis=0) for k, v in arrs.items()}
    std = {k: v.std(axis=0) for k, v in arrs.items()}

    print(f"\norigins={n_origins}  "
          + "  ".join(f"clip[{k}]={np.mean(v):.1%}" for k, v in clip_rates.items()))

    hdr = "  h  " + "".join(f"{k:>18s}" for k in KINDS) + f"{'persistence':>14s}"
    if period:
        hdr += f"{'climatology':>14s}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, h in enumerate(range(1, H + 1)):
        if h <= 6 or h % 3 == 0:
            row = f"{h:3d}  "
            for k in KINDS:
                row += f"{mean[k][i]:8.4f}+-{std[k][i]:.3f}"
            row += f"{baselines['persistence'][i]:14.4f}"
            if period:
                row += f"{baselines['climatology'][i]:14.4f}"
            print(row)

    # ---- verdict, mean-curve basis, all material margins over the sweep ---
    print("\n-- kill-test on the seed-mean curve (plan.md Sec.7.1 + Sec.10) --")
    verdicts = {}
    for kind in ("ising", "xxz_hx"):
        has_skill = wins(mean[kind].tolist(), [1.0] * H)
        vs_p = wins(mean[kind].tolist(), baselines["persistence"], MATERIAL)
        vs_e = wins(mean[kind].tolist(), mean["esn"].tolist(), MATERIAL)
        esn_beats = wins(mean["esn"].tolist(), mean[kind].tolist(), MATERIAL)
        useful = sorted(set(has_skill) & set(vs_p))
        verdicts[kind] = dict(has_skill=has_skill, vs_p=vs_p, vs_e=vs_e,
                              esn_beats=esn_beats, useful=useful)
        print(f"  {kind:7s} skill {as_range(has_skill, H):>10s} | "
              f"beats persistence {as_range(vs_p, H):>10s} | "
              f"useful {as_range(useful, H):>10s}")
        print(f"          vs mean-ESN (>{MATERIAL:.0%}): "
              f"QRC better {as_range(vs_e, H)}, ESN better {as_range(esn_beats, H)}")
        if kind == "ising":
            win_rate = (arrs["ising"] < arrs["esn"] * (1 - MATERIAL)).mean(axis=0)
            print(f"          per-seed win rate vs ESN (fraction of the "
                  f"{len(args.seeds)} seeds where ising materially beats "
                  f"that seed's ESN), by horizon 1..{H}:")
            print("          " + " ".join(f"{w:.1f}" for w in win_rate))

    best = min(("ising", "xxz_hx"), key=lambda k: float(mean[k].mean()))
    useful = verdicts[best]["useful"]
    passed = bool(useful)
    print(f"\n  best kind by mean NMSE (seed-averaged): {best}")
    if passed:
        print(f"  VERDICT: PASS -- {best} useful at {as_range(useful, H)} "
              f"(mean over {len(args.seeds)} seeds)")
    else:
        print("  VERDICT: FAIL -- no horizon with real skill that materially "
              "beats persistence, on the seed-mean curve")
    if verdicts[best]["vs_e"]:
        print(f"  vs CLASSICAL: {best} materially beats the mean-ESN curve "
              f"at {as_range(verdicts[best]['vs_e'], H)} only.")
    else:
        print(f"  vs CLASSICAL: {best} never materially beats the mean-ESN "
              "curve at any lead.")

    out_path = _HERE / "results" / f"step2_seedsweep_{s.key}.json"
    payload = dict(
        dataset=s.key, name=s.name, N=len(x), horizon=H, qubits=args.qubits,
        seeds=args.seeds, period=period, n_origins=n_origins,
        mean={k: mean[k].tolist() for k in KINDS},
        std={k: std[k].tolist() for k in KINDS},
        per_seed={k: arrs[k].tolist() for k in KINDS},
        baselines=baselines, verdicts=verdicts, best=best, passed=passed,
    )
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path.relative_to(_HERE)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
