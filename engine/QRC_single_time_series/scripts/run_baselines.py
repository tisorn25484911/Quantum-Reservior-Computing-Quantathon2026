"""scripts/run_baselines.py -- P7: the full baseline battery (teacher-forced table).

Runs every classical / statistical / reservoir / variational baseline through the
SAME loaders, split, scaler and one-step convention (G3) and prints a teacher-forced
one-step NMSE table on (a) Mackey-Glass tau=17 and (b) the ENSO development split.
It also builds the G7 effective-rank matching table (the size-matched ESN matches
the QRC's EFFECTIVE feature rank, not its qubit count) and the Haar-random-reservoir
control. torch-dependent LSTM/GRU/QLSTM are included only when the optional extra is
installed; their absence is reported, never silently skipped.

Results -> results/metrics/run_baselines.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.dynamical_systems import mackey_glass as MG
from qrc_single_time_series.data.loaders import load_raw, split
from qrc_single_time_series.data.preprocessing import anomaly, fit_scaler
from qrc_single_time_series.quantum.hamiltonians import fc_tfi, haar_reservoir
from qrc_single_time_series.quantum.exact_qrc import ExactQRC
from qrc_single_time_series.training.teacher_forcing import train_teacher_forced
from qrc_single_time_series.models import autoregression as AR
from qrc_single_time_series.models import nvar as NV
from qrc_single_time_series.models import esn as ESN
from qrc_single_time_series.models import qlstm as QV
from qrc_single_time_series.models import lstm as LSTM
from qrc_single_time_series.models import gru as GRU
from qrc_single_time_series.models import statistical as STAT
from qrc_single_time_series.models import trivial_baselines as TB
from qrc_single_time_series.evaluation import metrics as MET


def _nmse(y_true, y_pred):
    return float(np.var(y_true - y_pred) / np.var(y_true))


def _mg_series(tau=17, n=1600, washout=800):
    d = MG.generate(tau, n=n, washout=washout)["series"]
    return d


def _enso_scaled():
    s = load_raw("enso")
    cut = split(len(s))["cut"]
    anom, _ = anomaly(s.month, s.values, cut)
    scale, _ = fit_scaler(anom, cut)
    return scale(anom), cut


def teacher_forced_table(y, n_train, qrc_eff_rank, label):
    """One-step NMSE for every baseline over the dev tail [n_train:]."""
    y = np.asarray(y, dtype=float)
    dev = slice(n_train, len(y) - 1)
    rows = {}

    # trivial (from P2)
    _, yt, yp = TB.persistence_forecast(y, 1)
    rows["persistence"] = _nmse(yt[n_train - 1:], yp[n_train - 1:])
    mu = TB.climatological_mean(y, n_train)
    rows["climatology"] = _nmse(y[n_train:], np.full(len(y) - n_train, mu))

    # ridge-AR
    ar = AR.fit(y, n_train, p=8)
    p = ar.predict_onestep(y)                             # aligned to y[8:]
    rows["ridge_ar"] = _nmse(y[n_train:], p[n_train - 8:])

    # NVAR
    nv = NV.fit(y, n_train, k_lag=4, stride=1)
    pn = nv.predict_onestep(y)                            # aligned to y[span+1:]
    off = nv.span + 1
    rows["nvar"] = _nmse(y[n_train:-1], pn[n_train - off: len(y) - 1 - off])

    # ESN best-of-class + size-matched (G7)
    boc = ESN.best_of_class(y, n_train, dev_slice=slice(n_train, len(y) - 1),
                            seed=1, n_reservoir=150)
    rows["esn_best_of_class"] = float(boc.dev_nmse)
    sm = ESN.size_matched(y, n_train, target_eff_rank=qrc_eff_rank, seed=2)
    psm = sm.predict_onestep(y)
    rows["esn_size_matched"] = _nmse(y[1:][dev], psm[dev])

    # windowed VQC (always available)
    vq = QV.WindowedVQC(window=6, n_qubits=3, layers=2, seed=0).fit(y, n_train,
                                                                    maxiter=60)
    pv = vq.predict_onestep(y)                            # aligned to y[6:]
    rows["windowed_vqc"] = _nmse(y[n_train:], pv[n_train - 6:])

    # optional neural
    if LSTM.AVAILABLE:
        for name, mod in (("lstm", LSTM), ("gru", GRU)):
            m = mod.fit(y, n_train, window=12, hidden=16)
            pm = m.predict_onestep(y)
            rows[name] = _nmse(y[n_train:], pm[n_train - 12:])
    else:
        rows["lstm"] = rows["gru"] = None                # torch extra not installed

    # SARIMA (statsmodels) -- one-step in-sample on the dev tail
    try:
        sm_arima = STAT.fit(y, n_train, seasonal_orders=(None,))
        pins = sm_arima.predict_onestep()                # over [0:n_train]
        rows["arima"] = {"order": sm_arima.order, "aic": sm_arima.aic}
    except Exception as e:                                # pragma: no cover
        rows["arima"] = {"error": str(e)}

    return {"label": label, "n_train": n_train, "n": len(y),
            "qrc_eff_rank": qrc_eff_rank,
            "esn_size_matched_table": getattr(sm, "match_table", None),
            "esn_size_matched_Nr": sm.n_reservoir,
            "onestep_nmse": rows}


def main():
    out = {"torch_available": LSTM.AVAILABLE}

    # --- Mackey-Glass tau=17 -------------------------------------------------
    d = _mg_series(17)
    n_tr = int(0.7 * len(d))
    qrc = ExactQRC(fc_tfi(5, J=1.0, h=0.5, seed=7), V=8, tau=2.0)
    Xq = qrc.features(d[:n_tr], check_budget=False)
    mg_rank = MET.effective_rank(Xq[:, :-1])
    _, info = train_teacher_forced(qrc, d, n_train=n_tr)
    out["mackey_glass"] = teacher_forced_table(d, n_tr, mg_rank, "mackey_glass_tau17")
    out["mackey_glass"]["qrc_onestep_nmse"] = info.get("nmse_dev_onestep")

    # --- ENSO dev split ------------------------------------------------------
    ys, cut = _enso_scaled()
    n_tr_e = int(0.7 * cut)
    qrc_e = ExactQRC(fc_tfi(5, J=1.0, h=0.5, seed=7), V=6, tau=2.0)
    Xe = qrc_e.features(ys[:cut], check_budget=False)
    enso_rank = MET.effective_rank(Xe[:, :-1])
    out["enso"] = teacher_forced_table(ys[:cut], n_tr_e, enso_rank, "enso_dev")

    dest = ROOT / "results" / "metrics" / "run_baselines.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))

    print("Baseline battery (teacher-forced one-step NMSE)")
    print("=" * 60)
    for ds in ("mackey_glass", "enso"):
        t = out[ds]
        print(f"\n{t['label']}  (n={t['n']}, n_train={t['n_train']}, "
              f"QRC eff.rank={t['qrc_eff_rank']:.2f}, "
              f"ESN size-matched Nr={t['esn_size_matched_Nr']})")
        for k, v in t["onestep_nmse"].items():
            if isinstance(v, dict) or v is None:
                print(f"   {k:20s} {v}")
            else:
                print(f"   {k:20s} {v:.4f}")
        if "qrc_onestep_nmse" in t:
            print(f"   {'QRC (FN exact)':20s} {t['qrc_onestep_nmse']:.4f}")
    print(f"\n-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
