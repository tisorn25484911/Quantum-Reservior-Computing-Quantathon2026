"""compose.py -- the end-to-end claim: predict the heatwave BEFORE it happens.

Step 1 (QRC) forecasts the MHW-intensity trajectory; Step 2 detects heatwaves.
Composition = run the detector on the *forecast* so a heatwave is flagged ahead
of time. This module wires them and measures how the warning degrades with lead.

Detector used here is the **Hobday threshold itself** (the rule that defines an
MHW): a day is a predicted heatwave when forecast intensity exceeds the seasonal
90th-percentile anomaly `theta = threshold - climatology`. That keeps the
composition honest -- no second model to overfit -- and it is exactly the
"classical detector on the QRC trajectory" the plan describes.

The calibration step the plan (Sec.5) and both repos insist on: a threshold fit
on *observations* applied to *forecasts* essentially never fires, because a ridge
forecast is smoother and lower-variance than reality. So the applied threshold is
re-fitted on **forecast-of-training** -- back-cast the QRC over the training span,
and pick the scale `c` on `c*theta` that maximises F1 there. Only then is it
applied, once, to the held-out test forecasts.

Reported against a **persistence-then-detect** baseline (freeze today's intensity,
threshold it) so the QRC forecast has to earn the warning.

    python compose.py --kind xxz_hx --horizon 14 --max-points 6000
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
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="xxz_hx")
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    # aligned intensity, seasonal threshold, and the actual MHW label
    frame = pd.read_csv(LABELS / "got_sst_mhw_labeled.csv", parse_dates=["date"])
    if len(frame) > a.max_points:
        frame = frame.iloc[-a.max_points:].reset_index(drop=True)
    x = frame["intensity"].to_numpy(float)
    theta = (frame["thresh"] - frame["clim"]).to_numpy(float)   # intensity thresh
    actual = frame["mhw"].to_numpy(bool)
    H, n = a.horizon, len(frame)

    res = make_reservoir(a.kind, n_qubits=5, seed=a.seed)
    ro = fit_readout(x, res, washout=100, train_frac=0.7)
    train_end = int(ro.train_idx[-1]) + 1

    # one drive with checkpoints; rollouts resume from there (O(H) each)
    u_all = np.clip(ro.scaler.to_unit(x), 0.0, 1.0)
    feats, states = drive(res, u_all, checkpoints=True)

    def forecasts(origins, use_persistence=False):
        """(yhat_intensity, theta_at_lead, actual_at_lead) stacked over origins,
        shape (len(origins), H)."""
        yh = np.empty((len(origins), H))
        th = np.empty((len(origins), H))
        ac = np.empty((len(origins), H), dtype=bool)
        for i, o in enumerate(origins):
            if use_persistence:
                yh[i] = persistence_forecast(x, int(o), H)
            else:
                yh[i] = rollout(x, res, ro, int(o), H,
                                state=states[o], feat=feats[o]).yhat
            th[i] = theta[o + 1:o + H + 1]
            ac[i] = actual[o + 1:o + H + 1]
        return yh, th, ac

    train_origins = ro.train_idx[ro.train_idx + H < n]
    test_origins = ro.test_idx[ro.test_idx + H < n]

    # ---- re-calibrate the applied threshold on forecast-of-training ----
    yh_tr, th_tr, ac_tr = forecasts(train_origins)
    cs = np.round(np.linspace(0.2, 1.3, 23), 3)
    best_c, best_f1 = 1.0, -1.0
    for c in cs:
        _, _, f1 = _f1(yh_tr > c * th_tr, ac_tr)
        if f1 > best_f1:
            best_f1, best_c = f1, c
    # same calibration procedure for the persistence baseline (fair)
    yhp_tr, thp_tr, acp_tr = forecasts(train_origins, use_persistence=True)
    best_cp, best_f1p = 1.0, -1.0
    for c in cs:
        _, _, f1 = _f1(yhp_tr > c * thp_tr, acp_tr)
        if f1 > best_f1p:
            best_f1p, best_cp = f1, c

    print(f"=== compose: Forecast-then-Detect on got_sst MHW ({a.kind}) ===")
    print(f"N={n}  H={H}  train_origins={len(train_origins)}  "
          f"test_origins={len(test_origins)}")
    print(f"re-calibration on forecast-of-training: QRC c*={best_c} "
          f"(F1 {best_f1:.3f})   persistence c*={best_cp} (F1 {best_f1p:.3f})")
    print(f"  (raw c=1.0 would fire on {(yh_tr > th_tr).mean():.1%} of forecast "
          f"days vs actual {ac_tr.mean():.1%} -- the under-alarm the calibration fixes)")

    # ---- held-out test: metrics vs lead ----
    yh_te, th_te, ac_te = forecasts(test_origins)
    yhp_te, thp_te, acp_te = forecasts(test_origins, use_persistence=True)

    print(f"\n  {'h':>3} {'QRC prec':>9}{'QRC rec':>8}{'QRC F1':>8}   "
          f"{'pers prec':>9}{'pers rec':>8}{'pers F1':>8}   {'MHW rate':>8}")
    rows = {"qrc": [], "persistence": []}
    for j in range(H):
        pq = _f1(yh_te[:, j] > best_c * th_te[:, j], ac_te[:, j])
        pp = _f1(yhp_te[:, j] > best_cp * thp_te[:, j], acp_te[:, j])
        rows["qrc"].append(pq)
        rows["persistence"].append(pp)
        if j < 5 or (j + 1) % 2 == 0:
            print(f"  {j+1:>3} {pq[0]:>9.2f}{pq[1]:>8.2f}{pq[2]:>8.2f}   "
                  f"{pp[0]:>9.2f}{pp[1]:>8.2f}{pp[2]:>8.2f}   {ac_te[:,j].mean():>8.2f}")

    q = np.array(rows["qrc"]); p = np.array(rows["persistence"])
    # honest summary: where does the QRC-composed warning beat persistence-composed?
    f1_win = [h + 1 for h in range(H) if q[h, 2] > p[h, 2] + 1e-6]
    print(f"\n  QRC-composed F1 beats persistence-composed at leads: "
          f"{f1_win if f1_win else 'never'}")
    print(f"  mean F1 over 1..{H}:  QRC {q[:,2].mean():.3f}  vs  "
          f"persistence {p[:,2].mean():.3f}")
    # the product number: recall of real heatwave days at a usable lead
    for h in (3, 7, 14):
        if h <= H:
            print(f"  lead {h:>2}d: QRC catches {q[h-1,1]:.0%} of real MHW days "
                  f"at precision {q[h-1,0]:.0%}  (persistence {p[h-1,1]:.0%})")

    _plot(q, p, H, a.kind)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"compose_mhw_{a.kind}.json").write_text(json.dumps(dict(
        kind=a.kind, H=H, n=n, n_test_origins=int(len(test_origins)),
        calib=dict(qrc_c=float(best_c), qrc_train_f1=float(best_f1),
                   persistence_c=float(best_cp), persistence_train_f1=float(best_f1p)),
        qrc=dict(precision=q[:, 0].tolist(), recall=q[:, 1].tolist(),
                 f1=q[:, 2].tolist()),
        persistence=dict(precision=p[:, 0].tolist(), recall=p[:, 1].tolist(),
                         f1=p[:, 2].tolist()),
        mhw_rate=[float(ac_te[:, j].mean()) for j in range(H)],
        f1_beats_persistence_at=f1_win,
    ), indent=2))
    print(f"\n  wrote results/compose_mhw_{a.kind}.json + figure")


def _plot(q, p, H, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.arange(1, H + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, lbl, c in [(1, "recall", "#1596a4"), (0, "precision", "#6f8a8c")]:
        ax[0].plot(h, q[:, k], color=c, lw=2, label=f"QRC {lbl}")
        ax[0].plot(h, p[:, k], color=c, lw=1.4, ls="--", label=f"persist {lbl}")
    ax[0].set_title("Composed warning: precision & recall vs lead")
    ax[0].set_xlabel("lead (days ahead)"); ax[0].set_ylabel("score")
    ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1)
    ax[1].plot(h, q[:, 2], color="#e0483a", lw=2, label="QRC-composed F1")
    ax[1].plot(h, p[:, 2], color="#e0483a", lw=1.4, ls="--", label="persistence-composed F1")
    ax[1].set_title("Heatwave-day F1 vs lead"); ax[1].set_xlabel("lead (days ahead)")
    ax[1].set_ylabel("F1"); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1)
    fig.suptitle(f"Forecast-then-Detect end-to-end ({kind}) -- Gulf of Thailand MHW")
    fig.tight_layout()
    fig.savefig(RESULTS / f"compose_mhw_{kind}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
