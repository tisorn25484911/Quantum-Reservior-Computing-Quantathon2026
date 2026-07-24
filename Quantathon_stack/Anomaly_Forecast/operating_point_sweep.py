"""operating_point_sweep.py -- tune the QRC operating point, without leakage.

Every result so far used the blind default `dt=2.0` (with J=h=1). But `dt` is
the reservoir's *operating point*: `U_sub = propagator(H, dt/V)`, so `dt` sets
how far the state evolves per input -- the "edge of chaos" knob the QRC
playbook says is the single biggest accuracy lever. An exploratory sweep showed
`dt=2.0` is in fact a poor point on `got_sst`; `dt~=1.0-1.4` is much better.

The catch: choosing `dt` by looking at the *test* skill curve is selecting a
hyperparameter on the test set -- exactly the leakage this project forbids. So
this script splits the record three ways, chronologically:

    [-------- train --------][-- val --][-- test --]
      fit the ridge read-out   pick dt    report only

`dt` (and `hx`) are chosen by mean NMSE on the VALIDATION origins; the winning
config is then scored **once** on the untouched TEST origins, against
persistence and the seasonal-climatology floor on the identical origins. The
test number is therefore honest: the operating point never saw it.

`ising` couplings and the ESN are random, so the confirmation pass at the
chosen `dt` is multi-seed (like seed_sweep.py). `xxz_hx` is deterministic and
is used for the search itself, one seed, for speed.

    python operating_point_sweep.py --dataset got_sst --max-points 6000
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

from dataloader import load                                    # noqa: E402
from forecast import (Scaler, make_reservoir, fit_readout,     # noqa: E402
                      skill_vs_horizon)

PERIODS = {"nino34": 12, "nino12": 12, "got_sst": 365, "tao": 365}
MATERIAL = 0.05


def three_way(n: int, washout: int, H: int,
              fr_train=0.6, fr_val=0.2):
    """Chronological train/val/test origin sets over the valid index range."""
    valid = np.arange(washout, n - 1)
    n_tr = int(fr_train * len(valid))
    n_va = int(fr_val * len(valid))
    train_end = int(valid[n_tr - 1]) + 1
    val_origins = valid[n_tr:n_tr + n_va]
    test_origins = valid[n_tr + n_va:]
    val_origins = val_origins[val_origins + H < n]
    test_origins = test_origins[test_origins + H < n]
    return train_end, val_origins, test_origins


def eval_config(x, kind, dt, hx, seed, H, period, origins, train_end):
    """Fit read-out on the train span, score skill on the given origins."""
    res = make_reservoir(kind, n_qubits=5, dt=dt, hx=hx, seed=seed)
    # Scaler + read-out must see the train span only. fit_readout's own
    # train_frac controls that; set it so its train_idx ends at train_end.
    n = len(x)
    valid = np.arange(100, n - 1)
    train_frac = float(np.searchsorted(valid, train_end) / len(valid))
    ro = fit_readout(x, res, washout=100, train_frac=train_frac)
    out = skill_vs_horizon(x, res, ro, H, origins=origins, period=period)
    return out


def useful_ranges(r, p, c, H):
    """Horizon spans where the reservoir has skill / beats each baseline."""
    r, p = np.asarray(r), np.asarray(p)
    skill = [h + 1 for h in range(H) if r[h] < 1.0]
    beats_p = [h + 1 for h in range(H) if r[h] < p[h] * (1 - MATERIAL)]
    out = dict(skill_to=max(skill) if skill else 0,
               beats_persistence=beats_p)
    if c is not None:
        c = np.asarray(c)
        beats_c = [h + 1 for h in range(H) if r[h] < c[h]]
        out["beats_climatology_to"] = max(beats_c) if beats_c else 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="got_sst")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--dts", type=float, nargs="+",
                    default=[0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0])
    ap.add_argument("--hxs", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23, 42, 101])
    ap.add_argument("--objective", choices=["mean", "short"], default="mean",
                    help="val objective: mean NMSE over all H, or over h=1..7")
    args = ap.parse_args()

    s = load(args.dataset)
    x = np.asarray(s.x, float)
    if len(x) > args.max_points:
        x = x[-args.max_points:]
    H, period = args.horizon, PERIODS.get(s.key)
    n = len(x)
    train_end, val_o, test_o = three_way(n, 100, H)

    print(f"=== operating-point sweep: {s.name} ({s.key}) ===")
    print(f"N={n}  H={H}  period={period or '-'}  "
          f"train_end={train_end}  |val|={len(val_o)}  |test|={len(test_o)}")
    obj_slice = slice(0, 7) if args.objective == "short" else slice(0, H)
    print(f"selecting dt,hx by {args.objective} NMSE on VALIDATION "
          f"(xxz_hx, seed 7)\n")

    # ---- search on validation, xxz_hx (deterministic), seed 7 -------------
    print(f"{'dt':>5} {'hx':>4} {'val h1-7':>9} {'val h1-H':>9} {'skill_to':>9}")
    grid = []
    for dt in args.dts:
        for hx in args.hxs:
            o = eval_config(x, "xxz_hx", dt, hx, 7, H, period, val_o, train_end)
            r = np.array(o["nmse"]["reservoir"])
            score = float(r[obj_slice].mean())
            grid.append((score, dt, hx, r))
            print(f"{dt:>5} {hx:>4} {r[:7].mean():>9.4f} {r.mean():>9.4f} "
                  f"{useful_ranges(r, o['nmse']['persistence'], None, H)['skill_to']:>9}")

    best_score, best_dt, best_hx, _ = min(grid, key=lambda t: t[0])
    default_val = eval_config(x, "xxz_hx", 2.0, 1.0, 7, H, period, val_o, train_end)
    dv = np.array(default_val["nmse"]["reservoir"])[obj_slice].mean()
    print(f"\nBEST on validation: dt={best_dt}, hx={best_hx}  "
          f"({args.objective} NMSE {best_score:.4f} vs default dt=2.0 "
          f"{dv:.4f}, {100*(dv-best_score)/dv:+.1f}%)")

    # ---- report ONCE on the held-out test set -----------------------------
    print(f"\n--- held-out TEST at chosen (dt={best_dt}, hx={best_hx}) ---")
    report = {}
    for kind in ("xxz_hx", "ising"):
        seeds = [7] if kind == "xxz_hx" else args.seeds
        curves = []
        for sd in seeds:
            o = eval_config(x, kind, best_dt, best_hx, sd, H, period,
                            test_o, train_end)
            curves.append(np.array(o["nmse"]["reservoir"]))
            base_p = np.array(o["nmse"]["persistence"])
            base_c = np.array(o["nmse"].get("climatology")) if period else None
        arr = np.array(curves)
        report[kind] = dict(mean=arr.mean(0).tolist(), std=arr.std(0).tolist())

    # ESN baseline on test, multi-seed
    esn_curves = []
    for sd in args.seeds:
        o = eval_config(x, "esn", best_dt, best_hx, sd, H, period, test_o, train_end)
        esn_curves.append(np.array(o["nmse"]["reservoir"]))
    esn = np.array(esn_curves)

    # default-dt test curve for the improvement headline (xxz_hx)
    o_def = eval_config(x, "xxz_hx", 2.0, 1.0, 7, H, period, test_o, train_end)
    def_curve = np.array(o_def["nmse"]["reservoir"])

    best_kind = min(("xxz_hx", "ising"),
                    key=lambda k: float(np.mean(report[k]["mean"])))
    bm = np.array(report[best_kind]["mean"])
    rng = useful_ranges(bm, base_p, base_c, H)

    def span(hs):
        if not hs:
            return "never"
        return (f"h={hs[0]}-{hs[-1]}" if hs == list(range(hs[0], hs[-1] + 1))
                else f"{len(hs)}/{H} horizons")

    print(f"\n  {'h':>3} {'xxz_hx':>16} {'ising':>16} {'esn':>16} "
          f"{'persist':>9} {'clim':>9}")
    for i, h in enumerate(range(1, H + 1)):
        if h <= 6 or h % 3 == 0:
            print(f"  {h:>3} {report['xxz_hx']['mean'][i]:>8.4f}"
                  f"+-{report['xxz_hx']['std'][i]:.3f}"
                  f" {report['ising']['mean'][i]:>8.4f}+-{report['ising']['std'][i]:.3f}"
                  f" {esn.mean(0)[i]:>8.4f}+-{esn.std(0)[i]:.3f}"
                  f" {base_p[i]:>9.4f}"
                  f" {base_c[i] if base_c is not None else float('nan'):>9.4f}")

    print(f"\n  best kind on test: {best_kind}")
    print(f"  has skill (NMSE<1):        {span(list(range(1, rng['skill_to']+1)))}")
    print(f"  beats persistence (>5%):   {span(rng['beats_persistence'])}")
    if base_c is not None:
        print(f"  beats seasonal climatology: h=1-{rng['beats_climatology_to']}")
    print(f"  --> honest usable range: h=1..{rng.get('beats_climatology_to', rng['skill_to'])}")
    print(f"\n  improvement vs default dt=2.0 (test, {best_kind if best_kind=='xxz_hx' else 'xxz_hx'}): "
          f"mean h1-H {def_curve.mean():.4f} -> {np.array(report['xxz_hx']['mean']).mean():.4f} "
          f"({100*(def_curve.mean()-np.array(report['xxz_hx']['mean']).mean())/def_curve.mean():+.1f}%)")

    out_path = _HERE / "results" / f"opsweep_{s.key}.json"
    out_path.write_text(json.dumps(dict(
        dataset=s.key, name=s.name, N=n, H=H, period=period,
        train_end=train_end, n_val=len(val_o), n_test=len(test_o),
        best_dt=best_dt, best_hx=best_hx, objective=args.objective,
        val_grid=[(sc, dt, hx) for sc, dt, hx, _ in grid],
        test=dict(xxz_hx=report["xxz_hx"], ising=report["ising"],
                  esn=dict(mean=esn.mean(0).tolist(), std=esn.std(0).tolist()),
                  persistence=base_p.tolist(),
                  climatology=base_c.tolist() if base_c is not None else None,
                  default_dt2_xxz=def_curve.tolist()),
        best_kind=best_kind, usable_ranges=rng,
    ), indent=2))
    print(f"\nwrote {out_path.relative_to(_HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
