"""stochastic.py -- plan.md Step 3: the K-sample rollout, and the fix it buys.

`compose.py` showed the honest failure: the ridge readout minimises squared
error, so its single ("mean") forecast trajectory is smooth and reverts to
climatology, and by long lead it stops crossing the heatwave threshold at all --
recall collapses. That is a property of the loss, not a bug.

The fix (plan.md Sec.3): don't threshold the mean. At each rollout step add a
residual resampled from the training-residual pool (the `bootstrap=True` path
already in `forecast.rollout`), giving K plausible trajectories. Two payoffs,
both tested here:

  1. CALIBRATION -- the ensemble should contain the truth about as often as its
     nominal band says (actual in the 90% band ~90% of the time, per lead).
     If it is over-confident, every downstream probability is wrong. Gate first.

  2. RECOVERED RECALL -- alarm on the *fraction* of the K trajectories that
     breach the seasonal threshold, not on whether the mean does. Excursions the
     mean smooths away survive in individual members, so long-lead warnings come
     back. Scored against the mean-trajectory composition and persistence.

    python stochastic.py --kind xxz_hx --horizon 14 --K 120 --stride 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "DataBase_Analysis"))

from forecast import (make_reservoir, fit_readout, drive, rollout,   # noqa: E402
                      persistence_forecast)

LABELS = _HERE / "labels"
RESULTS = _HERE / "results"


def _f1(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="xxz_hx")
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--K", type=int, default=120)
    ap.add_argument("--stride", type=int, default=3,
                    help="subsample origins for the K-sample cost")
    ap.add_argument("--max-points", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    frame = pd.read_csv(LABELS / "got_sst_mhw_labeled.csv", parse_dates=["date"])
    if len(frame) > a.max_points:
        frame = frame.iloc[-a.max_points:].reset_index(drop=True)
    x = frame["intensity"].to_numpy(float)
    theta = (frame["thresh"] - frame["clim"]).to_numpy(float)
    actual = frame["mhw"].to_numpy(bool)
    H, n, K = a.horizon, len(frame), a.K

    res = make_reservoir(a.kind, n_qubits=5, seed=a.seed)
    ro = fit_readout(x, res, washout=100, train_frac=0.7)
    u_all = np.clip(ro.scaler.to_unit(x), 0.0, 1.0)
    feats, states = drive(res, u_all, checkpoints=True)
    rng = np.random.default_rng(a.seed)

    tr = ro.train_idx[ro.train_idx + H < n][::a.stride]
    te = ro.test_idx[ro.test_idx + H < n][::a.stride]
    print(f"=== Step 3 stochastic rollout ({a.kind}) ===")
    print(f"N={n} H={H} K={K}  train_origins={len(tr)} test_origins={len(te)} "
          f"(stride {a.stride})")

    def ensemble(origins):
        """Return mean traj, ensemble (len,K,H), theta@lead, actual@lead, truth."""
        mean = np.empty((len(origins), H))
        ens = np.empty((len(origins), K, H))
        th = np.empty((len(origins), H)); ac = np.empty((len(origins), H), bool)
        tru = np.empty((len(origins), H))
        for i, o in enumerate(origins):
            mean[i] = rollout(x, res, ro, int(o), H, state=states[o],
                              feat=feats[o]).yhat
            for k in range(K):
                ens[i, k] = rollout(x, res, ro, int(o), H, state=states[o],
                                    feat=feats[o], rng=rng, bootstrap=True).yhat
            th[i] = theta[o + 1:o + H + 1]; ac[i] = actual[o + 1:o + H + 1]
            tru[i] = x[o + 1:o + H + 1]
        return mean, ens, th, ac, tru

    print("  rolling ensembles (test)...", flush=True)
    mean_te, ens_te, th_te, ac_te, tru_te = ensemble(te)

    # ---- GATE 1: calibration / coverage vs lead ----
    lo = np.percentile(ens_te, 5, axis=1)     # (len,H)
    hi = np.percentile(ens_te, 95, axis=1)
    cover = ((tru_te >= lo) & (tru_te <= hi)).mean(axis=0)   # per lead, nominal .90
    print("\n  GATE 1 -- 90% band coverage vs lead (nominal 0.90):")
    print("   h   :", " ".join(f"{h:4d}" for h in range(1, H + 1)))
    print("   cov :", " ".join(f"{c:4.2f}" for c in cover))
    print(f"   mean coverage {cover.mean():.2f}  "
          f"({'OK ~calibrated' if cover.mean() > 0.8 else 'OVER-CONFIDENT'})")

    # ---- calibrate the ensemble alarm on forecast-of-training ----
    print("  rolling ensembles (train, for calibration)...", flush=True)
    _, ens_tr, th_tr, ac_tr, _ = ensemble(tr)
    # alarm score = fraction of members breaching c*theta; tune (c, p) by F1
    best = (-1, 1.0, 0.5)
    for c in np.linspace(0.4, 1.2, 9):
        frac_tr = (ens_tr > c * th_tr[:, None, :]).mean(axis=1)   # (len,H)
        for p in np.linspace(0.1, 0.6, 11):
            _, _, f1 = _f1(frac_tr > p, ac_tr)
            if f1 > best[0]:
                best = (f1, c, p)
    _, c_star, p_star = best
    print(f"\n  ensemble alarm calibrated on train: c={c_star:.2f} "
          f"p={p_star:.2f} (F1 {best[0]:.3f})")

    # ---- compare three composed warnings on test ----
    frac_te = (ens_te > c_star * th_te[:, None, :]).mean(axis=1)
    # mean-trajectory composition, re-using compose.py's calibration idea here
    cm = 0.85   # from compose.py's forecast-of-training fit (xxz_hx)
    print(f"\n  {'h':>3} {'ENSEMBLE':>18} {'MEAN-traj':>18} {'persistence':>18} {'MHWrate':>8}")
    print(f"  {'':>3} {'prec  rec   F1':>18} {'prec  rec   F1':>18} {'prec  rec   F1':>18}")
    rows = {"ensemble": [], "mean": [], "persistence": []}
    for j in range(H):
        pe = _f1(frac_te[:, j] > p_star, ac_te[:, j])
        pm = _f1(mean_te[:, j] > cm * th_te[:, j], ac_te[:, j])
        pers = np.array([persistence_forecast(x, int(o), H)[j] for o in te])
        pp = _f1(pers > 1.05 * th_te[:, j], ac_te[:, j])
        rows["ensemble"].append(pe); rows["mean"].append(pm); rows["persistence"].append(pp)
        if j < 4 or (j + 1) % 2 == 0:
            print(f"  {j+1:>3} {pe[0]:>5.2f}{pe[1]:>6.2f}{pe[2]:>6.2f}   "
                  f"{pm[0]:>5.2f}{pm[1]:>6.2f}{pm[2]:>6.2f}   "
                  f"{pp[0]:>5.2f}{pp[1]:>6.2f}{pp[2]:>6.2f}   {ac_te[:,j].mean():>8.2f}")

    E = np.array(rows["ensemble"]); M = np.array(rows["mean"]); P = np.array(rows["persistence"])
    print(f"\n  mean F1 over 1..{H}:  ensemble {E[:,2].mean():.3f}  "
          f"mean-traj {M[:,2].mean():.3f}  persistence {P[:,2].mean():.3f}")
    for h in (7, 10, 14):
        if h <= H:
            print(f"  lead {h:>2}d recall:  ensemble {E[h-1,1]:.0%}  "
                  f"mean-traj {M[h-1,1]:.0%}  persistence {P[h-1,1]:.0%}"
                  + ("   <- recovered" if E[h-1,1] > M[h-1,1] + 0.05 else ""))

    _plot(cover, E, M, P, H, a.kind)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"step3_stochastic_{a.kind}.json").write_text(json.dumps(dict(
        kind=a.kind, H=H, K=K, n_test=int(len(te)),
        coverage_per_lead=cover.tolist(), mean_coverage=float(cover.mean()),
        calib=dict(c=float(c_star), p=float(p_star), train_f1=float(best[0])),
        ensemble=dict(precision=E[:, 0].tolist(), recall=E[:, 1].tolist(), f1=E[:, 2].tolist()),
        mean_traj=dict(precision=M[:, 0].tolist(), recall=M[:, 1].tolist(), f1=M[:, 2].tolist()),
        persistence=dict(precision=P[:, 0].tolist(), recall=P[:, 1].tolist(), f1=P[:, 2].tolist()),
    ), indent=2))
    print(f"\n  wrote results/step3_stochastic_{a.kind}.json + figure")


def _plot(cover, E, M, P, H, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.arange(1, H + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].axhline(0.9, color="#6f8a8c", ls=":", lw=1, label="nominal 0.90")
    ax[0].plot(h, cover, color="#1596a4", lw=2, marker="o", ms=3)
    ax[0].set_title("Gate 1: ensemble 90% coverage vs lead")
    ax[0].set_xlabel("lead (days ahead)"); ax[0].set_ylabel("coverage")
    ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
    ax[1].plot(h, E[:, 1], color="#e0483a", lw=2, label="ensemble recall")
    ax[1].plot(h, M[:, 1], color="#e0483a", lw=1.4, ls="--", label="mean-traj recall")
    ax[1].plot(h, P[:, 1], color="#6f8a8c", lw=1.4, ls=":", label="persistence recall")
    ax[1].set_title("Recovered long-lead recall"); ax[1].set_xlabel("lead (days ahead)")
    ax[1].set_ylabel("recall of real MHW days"); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8)
    fig.suptitle(f"Step 3 stochastic rollout ({kind}) -- Gulf of Thailand MHW")
    fig.tight_layout(); fig.savefig(RESULTS / f"step3_stochastic_{kind}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
