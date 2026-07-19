"""analysis.py -- run the whole inspection over one dataset or all of them.

For each series: plot the raw record, its linear-scale amplitude spectrum, the
divergence curve behind lambda_1, and the record re-plotted against Lyapunov
time; then report lambda_1, T_lambda, and the dominant periods.

    python analysis.py                       # every dataset present on disk
    python analysis.py --dataset nino34      # one
    python analysis.py --tier chaotic        # one tier
    python analysis.py --method rosenstein   # force single-trajectory
    python analysis.py --json out.json       # machine-readable numbers too

Figures land in `figures/` beside this file, one PNG per dataset, plus a
summary table on stdout.

The chaotic tier doubles as the validation: it is the only tier where
lambda_1 is published, so its error column is the calibration for every other
number the script prints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dataloader as dl
import fourier
import lyapunov as lyap
import visualizer as viz

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"

# The fast records have a Nyquist far above anything meaningful: half-hourly
# solar reaches 1 cycle/h while the diurnal line sits at 1/24. Cropping the
# spectrum axis per dataset is the difference between a readable plot and a
# spike at the origin.
FMAX = {
    "solar": 0.25, "load": 0.25, "opsd": 0.25, "potomac15": 0.25,   # per hour
    "nino34": 3.0, "nino12": 3.0, "brest": 3.0, "cuxhaven": 3.0,    # per year
    # per day: the annual line sits at 1/365 = 0.0027, the weekly at 0.14, so
    # 0.2 keeps both in frame while cropping the empty decade up to Nyquist
    "tao": 0.05, "nyc": 0.05, "lax": 0.05, "potomac": 0.2,
}

# Likewise for panel (d): 900 e-foldings of Lorenz-63 is a solid block of ink.
MAX_UNITS = {"lorenz63": 20.0, "vallieselnino": 20.0, "lorenz84": 20.0,
             "hadley": 20.0, "rikitake": 20.0}


def analyse_one(series, method="auto", peaks=5, save=True, show=False):
    """Full pass over one Series. Returns a dict of the reported numbers."""
    spec = fourier.spectrum(series)
    # drop the lowest few bins before peak-picking: on a short record the
    # residual near-DC leakage outranks every real line.
    fmin = 2.0 / (len(series.x) * series.dt)
    top = fourier.dominant_periods(spec.f, spec.amp, k=peaks, fmin=fmin)
    est = lyap.estimate(series, method=method)

    path = viz.figure(series, spec, est, peaks=top,
                      fmax=FMAX.get(series.key),
                      max_units=MAX_UNITS.get(series.key),
                      outdir=FIGDIR if save else None, show=show)

    span = len(series.x) * series.dt
    return {
        "key": series.key, "name": series.name, "tier": series.tier,
        "n": int(len(series.x)), "dt": float(series.dt),
        "time_unit": series.time_unit, "span": float(span),
        "n_realizations": series.n_realizations,
        "lyap": float(est.lyap), "lyap_time": float(est.lyap_time),
        "lyap_method": est.method, "lyap_truth": est.truth,
        "lyap_error": est.error, "embedding": est.embedding,
        "e_foldings_in_record": float(span * est.lyap) if est.lyap > 0 else 0.0,
        "dominant_periods": top,
        "figure": str(path) if path else None,
        "_est": est,
    }


def _report(r):
    u = r["time_unit"] or "sample"
    print(f"\n=== {r['name']}  [{r['tier']}]")
    print(f"  N = {r['n']}   dt = {r['dt']:.4g} {u}   span = {r['span']:.4g} {u}"
          + (f"   R = {r['n_realizations']}" if r["n_realizations"] else ""))
    print("  " + r["_est"].summary().replace("\n", "\n  "))
    print(f"  record length = {r['e_foldings_in_record']:.1f} e-foldings")
    if r["dominant_periods"]:
        print("  dominant periods:")
        for p in r["dominant_periods"]:
            print(f"    {p['period']:10.4g} {u}   "
                  f"(f = {p['f']:.5g} /{u},  amp = {p['amp']:.4g})")
    if r["figure"]:
        print(f"  figure -> {r['figure']}")


def _table(rows):
    u = lambda r: r["time_unit"] or "smp"                       # noqa: E731
    print("\n" + "=" * 78)
    print(f"{'dataset':16s} {'tier':10s} {'lambda_1':>10s} {'T_lambda':>12s} "
          f"{'method':>11s} {'err':>8s}")
    print("-" * 78)
    for r in rows:
        err = "" if r["lyap_error"] is None else f"{r['lyap_error']:+.1%}"
        print(f"{r['key']:16s} {r['tier']:10s} {r['lyap']:10.4f} "
              f"{r['lyap_time']:9.3g} {u(r):<2s} {r['lyap_method']:>11s} {err:>8s}")
    print("=" * 78)
    if any(r["lyap_method"] == "rosenstein" for r in rows):
        print("rosenstein rows are single-trajectory: upper bounds, not "
              "measurements.\nsee Data/README.md for the ground-truth "
              "comparison that calibrates them.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", "-d", help=f"one of: {', '.join(dl.ALL_KEYS)}")
    ap.add_argument("--tier", choices=("chaotic", "real", "surrogate"))
    ap.add_argument("--method", default="auto",
                    choices=("auto", "ensemble", "rosenstein"))
    ap.add_argument("--member", type=int, default=0,
                    help="which realization becomes the plotted trajectory")
    ap.add_argument("--peaks", type=int, default=5)
    ap.add_argument("--no-save", action="store_true", help="skip writing PNGs")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--json", help="also write the numbers to this path")
    a = ap.parse_args(argv)

    if a.dataset:
        series = {a.dataset: dl.load(a.dataset, member=a.member)}
    else:
        series = dl.load_all(a.tier, member=a.member)
    if not series:
        raise SystemExit("no datasets found -- run `python fetch_data.py` "
                         f"in {dl.DATA}")

    rows = []
    for key, s in series.items():
        try:
            r = analyse_one(s, method=a.method, peaks=a.peaks,
                            save=not a.no_save, show=a.show)
        except Exception as exc:                       # noqa: BLE001
            print(f"\n=== {key}: FAILED -- {exc}")
            continue
        _report(r)
        rows.append(r)

    if rows:
        _table(rows)
    if a.json and rows:
        out = [{k: v for k, v in r.items() if k != "_est"} for r in rows]
        Path(a.json).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nnumbers -> {a.json}")
    return rows


if __name__ == "__main__":
    main()
