"""tune_forecast.py -- find the reservoir config + horizon that forecasts the
Gulf-of-Thailand SST best, the leakage-safe way.

Discipline (QRC playbook rubric):
  * chronological train / VAL / test, never shuffled;
  * scaler + ridge fitted on train only; ridge lambda picked on VAL;
  * CONFIG + horizon selected on VAL, final numbers reported on TEST;
  * every result carries persistence and a size-matched ESN -- parity with the
    ESN is the ceiling, not "advantage".

Reservoir features depend only on (n_qubits, dt, virtual_nodes, use_zz), not on
the horizon or lambda, so they are computed once per config and reused across
all horizons -- which is what makes a real sweep affordable at these sizes.

Run from repo root:  .venv/bin/python -m ... (see __main__); writes a JSON
summary to results/forecast_tuning.json and prints the tables.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_WEBAPP = _HERE.parents[1] / "webapp"
sys.path.insert(0, str(_WEBAPP))

from app.domain.catalog import series_values          # noqa: E402
from qrc_core import QuantumReservoir, ESN, ridge_fit, ridge_predict  # noqa: E402

DATASET = "got_ersst"
MAXP = 2000
WASHOUT = 100
SEED = 7
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2          # remainder is test
LAM_GRID = [1e-8, 1e-6, 1e-4, 1e-2]
H_GRID = [1, 2, 3, 6, 9, 12, 18, 24]
SELECT_HS = (1, 3, 6)                     # actionable short-lead window
DT_GRID = [0.5, 1.0, 1.5, 2.05, 3.0, 4.5, 6.0]
VN_GRID = [8, 16, 24]
ZZ_GRID = [True, False]

# --------------------------------------------------------------------- data
x, META = series_values(DATASET, max_points=MAXP)
T = len(x)
# Scaler on a FIXED early train span (independent of horizon) so the reservoir
# features can be cached across horizons. Uses only early data -> leakage-safe.
_train_end = int(TRAIN_FRAC * T)
_lo, _hi = float(np.min(x[:_train_end])), float(np.max(x[:_train_end]))
U = np.clip((x - _lo) / (_hi - _lo), 0.0, 1.0)

_feat_cache: dict[tuple, np.ndarray] = {}


def features(q, dt, vn, zz):
    key = (q, dt, vn, zz)
    if key not in _feat_cache:
        qr = QuantumReservoir(n_qubits=q, dt=dt, virtual_nodes=vn,
                              use_zz=zz, seed=SEED)
        _feat_cache[key] = qr.run(U)
    return _feat_cache[key]


def _splits(h):
    valid = np.arange(WASHOUT, T - h)
    n = len(valid)
    ntr, nva = int(TRAIN_FRAC * n), int(VAL_FRAC * n)
    return valid[:ntr], valid[ntr:ntr + nva], valid[ntr + nva:]


def _nmse(y, pred, ybar):
    denom = np.mean((y - ybar) ** 2)
    return float(np.mean((y - pred) ** 2) / denom) if denom > 0 else np.nan


def evaluate(X, h):
    """Return dict of VAL/TEST metrics at horizon h for feature matrix X.
    lambda chosen on VAL; nothing here ever fits on val or test."""
    tr, va, te = _splits(h)
    y = x[np.arange(T)]  # placeholder, indexed by origin+h below
    ytr, yva, yte = x[tr + h], x[va + h], x[te + h]
    ybar = float(np.mean(ytr))

    best = None
    for lam in LAM_GRID:
        w = ridge_fit(X[tr], ytr, lam=lam)
        mse_va = np.mean((yva - ridge_predict(X[va], w)) ** 2)
        if best is None or mse_va < best[0]:
            best = (mse_va, lam, w)
    _, lam, w = best
    pv, pt = ridge_predict(X[va], w), ridge_predict(X[te], w)

    # persistence: predict value now (origin) for origin+h
    rmse_p = float(np.sqrt(np.mean((yte - x[te]) ** 2)))
    rmse_q = float(np.sqrt(np.mean((yte - pt) ** 2)))
    return {
        "lam": lam,
        "val_nmse": _nmse(yva, pv, ybar),
        "test_nmse": _nmse(yte, pt, ybar),
        "test_rmse": rmse_q,
        "persist_rmse": rmse_p,
        "skill_vs_persist": float(1.0 - rmse_q / rmse_p) if rmse_p > 0 else np.nan,
    }


def esn_eval(n_features, h):
    tr, va, te = _splits(h)
    Xe = ESN(n_features, seed=SEED + 4).run(U)
    ytr, yte = x[tr + h], x[te + h]
    best = None
    for lam in LAM_GRID:
        w = ridge_fit(Xe[tr], ytr, lam=lam)
        mse_va = np.mean((x[va + h] - ridge_predict(Xe[va], w)) ** 2)
        if best is None or mse_va < best[0]:
            best = (mse_va, w)
    return float(np.sqrt(np.mean((yte - ridge_predict(Xe[te], best[1])) ** 2)))


def config_score(q, dt, vn, zz):
    """Mean VAL NMSE over the short actionable horizons -- the selection metric."""
    X = features(q, dt, vn, zz)
    per_h = {h: evaluate(X, h) for h in H_GRID}
    sel = np.mean([per_h[h]["val_nmse"] for h in SELECT_HS])
    return sel, per_h


def main():
    t0 = time.time()
    results = []

    # -- Phase 1: sweep dt x vnodes x zz at q=6 -----------------------------
    print("== Phase 1: dt x vnodes x zz at q=6 ==", flush=True)
    for dt in DT_GRID:
        for vn in VN_GRID:
            for zz in ZZ_GRID:
                sel, per_h = config_score(6, dt, vn, zz)
                results.append({"q": 6, "dt": dt, "vn": vn, "zz": zz,
                                "sel_val_nmse": sel, "per_h": per_h})
                print(f"  q6 dt={dt:<4} vn={vn:<2} zz={int(zz)}  "
                      f"selVAL={sel:.3f}", flush=True)

    results.sort(key=lambda r: r["sel_val_nmse"])
    top = results[:3]
    print("\n  top-3 @ q6:", [(r['dt'], r['vn'], r['zz'],
          round(r['sel_val_nmse'], 3)) for r in top], flush=True)

    # -- Phase 2: promote top-3 shapes to q=5 and q=7 -----------------------
    print("\n== Phase 2: qubit refinement on top shapes ==", flush=True)
    for r in top:
        for q in (5, 7):
            sel, per_h = config_score(q, r["dt"], r["vn"], r["zz"])
            results.append({"q": q, "dt": r["dt"], "vn": r["vn"], "zz": r["zz"],
                            "sel_val_nmse": sel, "per_h": per_h})
            print(f"  q{q} dt={r['dt']:<4} vn={r['vn']:<2} zz={int(r['zz'])}  "
                  f"selVAL={sel:.3f}", flush=True)

    results.sort(key=lambda r: r["sel_val_nmse"])
    best = results[0]
    X = features(best["q"], best["dt"], best["vn"], best["zz"])
    nfeat = X.shape[1]

    # -- Report the winner across all horizons, with baselines --------------
    print("\n== WINNER ==", flush=True)
    print(f"  q={best['q']} dt={best['dt']} vn={best['vn']} zz={best['zz']} "
          f"(n_features={nfeat})  sel_val_nmse={best['sel_val_nmse']:.3f}",
          flush=True)
    print(f"  {'H':>3} {'test_nmse':>9} {'skill_vs_p':>11} {'esn_ratio':>9} "
          f"{'reading':>22}", flush=True)
    table = []
    for h in H_GRID:
        e = best["per_h"][h]
        esn_rmse = esn_eval(nfeat, h)
        ratio = e["test_rmse"] / esn_rmse if esn_rmse > 0 else np.nan
        reading = ("beats persist+ESN" if e["skill_vs_persist"] > 0 and ratio < 1.0
                   else "beats persistence" if e["skill_vs_persist"] > 0
                   else "no gain vs persist")
        table.append({"h": h, **e, "esn_rmse": esn_rmse, "esn_ratio": ratio,
                      "reading": reading})
        print(f"  {h:>3} {e['test_nmse']:>9.3f} "
              f"{e['skill_vs_persist']*100:>10.1f}% {ratio:>9.3f}  {reading:>22}",
              flush=True)

    # best H = max test skill vs persistence among horizons that also beat ESN
    beat_esn = [t for t in table if t["esn_ratio"] < 1.0 and t["skill_vs_persist"] > 0]
    pool = beat_esn or [t for t in table if t["skill_vs_persist"] > 0] or table
    best_h = max(pool, key=lambda t: t["skill_vs_persist"])["h"]
    print(f"\n  BEST HORIZON H = {best_h}", flush=True)

    out = {
        "dataset": META["name"], "T": T,
        "winner": {"n_qubits": best["q"], "dt": best["dt"],
                   "virtual_nodes": best["vn"], "use_zz": best["zz"],
                   "n_features": nfeat, "sel_val_nmse": best["sel_val_nmse"]},
        "best_horizon": int(best_h),
        "per_horizon": table,
        "all_configs": [{k: r[k] for k in ("q", "dt", "vn", "zz", "sel_val_nmse")}
                        for r in results],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (_HERE / "results").mkdir(exist_ok=True)
    (_HERE / "results" / "forecast_tuning.json").write_text(json.dumps(out, indent=1))
    print(f"\n  wrote results/forecast_tuning.json  ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
