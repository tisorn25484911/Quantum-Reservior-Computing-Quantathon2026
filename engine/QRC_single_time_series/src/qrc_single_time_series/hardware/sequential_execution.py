"""sequential_execution.py -- closed-loop sequential totals + strategy comparison A-G.

Two jobs (spec s26):

1. The classical SURROGATE of the quantum feature map (strategy F, and the carrier
   between refreshes in strategy G): a small model fit to (rewind-window -> QRC
   feature) pairs on a dev split. It gives (a) a measured per-step feature-eval cost
   for the ledger and (b) a real held-out accuracy, so the accuracy-vs-latency Pareto
   is not a straw man. This arm doubles as the Gate-7 cost comparator (playbook: a
   model that needs a QPU to match a laptop has answered the business question).

2. Sequential TOTALS: per-step ledgers (from ``latency_model``) multiplied by the
   number of SEQUENTIAL closed-loop steps -- 1 / 12 / H_effective / 100 / 1000 and one
   rolling-origin backtest. Because the rewind loop is inherently sequential, the total
   is n x (one round trip); batched open-loop throughput is NOT used as closed-loop
   evidence (acceptance item 24). Per-deployment-mode verdicts apply the s26.4
   practicality criterion: latency-vs-decision-deadline, per use case. Implemented in P11.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..quantum.hamiltonians import nn_tfi
from ..quantum.rewind_qrc import RewindReservoir
from ..models import readout as R
from . import latency_model as LM

# Named decision deadlines (seconds) used by the s26.4 practicality criterion. The
# operational use case in this project is a MONTHLY climate index, so its deadline is
# generous; the intraday row shows where sequential QPU latency does bite.
DECISION_DEADLINES_S = {
    "monthly_index": 30 * 24 * 3600.0,      # one month between monthly forecasts
    "daily_ops": 24 * 3600.0,               # a daily operational cycle
    "intraday_trading": 60.0,               # a one-minute decision loop
}


# ------------------------------------------------------- classical surrogate (F)
def _minmax01(x):
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


@dataclass
class SurrogateFit:
    """A classical surrogate of the window->feature map, with its measured cost."""

    qrc_test_nmse: float
    surrogate_test_nmse: float
    predict_s: float                        # measured per-window feature-eval cost
    feature_rmse: float                     # surrogate vs true QRC features (test)
    meta: dict


def _load_enso01():
    """ENSO train-anomaly series, min-max scaled to the encoder's [0,1] domain."""
    from ..data.loaders import load_raw, split
    from ..data.preprocessing import anomaly
    s = load_raw("enso")
    cut = split(len(s))["cut"]
    anom, _ = anomaly(s.month, s.values, cut)
    return _minmax01(anom), cut


def fit_surrogate(series=None, N=4, t_w=6, tau=1.0, cut=None, cut_frac=0.6, seed=7,
                  hidden=(64,), predict_repeats=200):
    """Fit a small MLP surrogate of the rewind window->feature map and score it.

    Default series is the real ENSO train-anomaly (the project's actual use case),
    min-max scaled to [0,1]. The surrogate is a genuine classical REPLACEMENT: its
    own ridge readout is fit on surrogate features, so its accuracy is what you would
    actually deploy without a QPU (the Gate-7 comparator), not the brittle QRC readout
    fed dirty features. Also returns the wall-clocked cost of one feature evaluation.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    if series is None:
        u, cut_series = _load_enso01()
    else:
        u, cut_series = _minmax01(series), None

    rw = RewindReservoir(nn_tfi(N, J=1.0, h=0.5, seed=seed), t_w=t_w, tau=tau)
    X = rw.features_series(u, include_bias=False)        # true QRC features
    y = u[t_w:]                                          # one-step-ahead target
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]
    # the surrogate input is the raw rewind window that produced each feature row
    W = np.array([u[t - t_w + 1: t + 1] for t in range(t_w - 1, t_w - 1 + n)])

    # honour the dataset train/test cut when known (row t maps to input index t+t_w-1)
    if cut is None and cut_series is not None:
        cut = max(1, min(n - 1, cut_series - (t_w - 1)))
    if cut is None:
        cut = int(cut_frac * n)
    Xtr, Xte = X[:cut], X[cut:]
    Wtr, Wte = W[:cut], W[cut:]
    ytr, yte = y[:cut], y[cut:]

    # true QRC readout (the quantum-feature ceiling for this config)
    ro = R.fit(np.hstack([Xtr, np.ones((cut, 1))]), ytr, lam="gcv")
    pred_qrc = ro.predict(np.hstack([Xte, np.ones((len(Xte), 1))]))
    qrc_nmse = float(np.var(yte - pred_qrc) / np.var(yte))

    # surrogate: window -> features (MLP), then its OWN ridge readout (real deploy)
    sx = StandardScaler().fit(Wtr)
    surrogate = MLPRegressor(hidden_layer_sizes=hidden, max_iter=2000,
                             random_state=seed).fit(sx.transform(Wtr), Xtr)
    Xtr_hat = surrogate.predict(sx.transform(Wtr))
    Xte_hat = surrogate.predict(sx.transform(Wte))
    ro_sur = R.fit(np.hstack([Xtr_hat, np.ones((cut, 1))]), ytr, lam="gcv")
    pred_sur = ro_sur.predict(np.hstack([Xte_hat, np.ones((len(Xte_hat), 1))]))
    sur_nmse = float(np.var(yte - pred_sur) / np.var(yte))
    feat_rmse = float(np.sqrt(np.mean((Xte_hat - Xte) ** 2)))

    # measured cost of one surrogate feature evaluation (single window)
    w0 = sx.transform(Wte[:1])
    ts = []
    for _ in range(predict_repeats):
        t0 = time.perf_counter()
        surrogate.predict(w0)
        ts.append(time.perf_counter() - t0)
    predict_s = float(np.median(ts))

    return SurrogateFit(
        qrc_test_nmse=qrc_nmse, surrogate_test_nmse=sur_nmse,
        predict_s=predict_s, feature_rmse=feat_rmse,
        meta={"N": N, "t_w": t_w, "tau": tau, "n": int(n), "cut": int(cut),
              "hidden": list(hidden), "series": "enso" if series is None else "custom"},
    )


# -------------------------------------------------------- accuracy assignment
def accuracy_by_strategy(qrc_nmse, surrogate_nmse):
    """One-step held-out NMSE each strategy DEPLOYS (lower is better).

    A/B/C/D/E and the refresh points of G read true (or noiseless-sim) QRC features,
    so they carry the QRC accuracy; F reads surrogate features. G between refreshes
    drifts toward the surrogate -- flagged in the note, reported at the QRC ceiling
    for a one-step-from-observed forecast (its best case).
    """
    return {
        "A_fresh_circuits": qrc_nmse,
        "B_pretranspiled_templates": qrc_nmse,
        "C_dynamic_circuits": qrc_nmse,
        "D_local_noisy_sim": qrc_nmse,
        "E_offline_features_classical_deploy": qrc_nmse,
        "F_classical_surrogate": surrogate_nmse,
        "G_periodic_refresh": qrc_nmse,
    }


# --------------------------------------------------------- strategy comparison
def compare_strategies(measured, asm, shots, surrogate_predict_s):
    """Per-step ledger for every strategy A-G (see ``latency_model.STRATEGIES``)."""
    return {k: LM.step_ledger(k, measured, asm, shots,
                              surrogate_predict_s=surrogate_predict_s)
            for k in LM.STRATEGIES}


def sequential_totals(ledgers, step_counts, backtest_steps):
    """Wall-clock totals for each strategy across the sequential step counts + backtest.

    ``step_counts`` maps a label (e.g. "H_eff") to an integer number of SEQUENTIAL
    closed-loop steps; ``backtest_steps`` is n_origins x H_eff (one rolling-origin
    backtest). Every total is n x (one round trip) -- no batching.
    """
    out = {}
    for key, led in ledgers.items():
        per = led["total_s"]
        totals = {lab: LM.total_time(per, n) for lab, n in step_counts.items()}
        totals["backtest"] = LM.total_time(per, backtest_steps)
        out[key] = {"per_step_s": per, "totals_s": totals,
                    "closed_loop_capable": led["closed_loop_capable"],
                    "remote": led["remote"]}
    return out


def pareto_points(ledgers, accuracy):
    """(strategy, per-step latency, one-step NMSE, closed-loop) for the Pareto plot.

    A point is Pareto-optimal if no other CLOSED-LOOP-CAPABLE strategy is at least as
    fast AND at least as accurate. E (not closed-loop capable) is carried but excluded
    from the frontier flag.
    """
    pts = []
    for k, led in ledgers.items():
        pts.append({
            "strategy": k, "label": led["label"],
            "per_step_s": led["total_s"], "nmse": float(accuracy[k]),
            "closed_loop_capable": led["closed_loop_capable"],
            "remote": led["remote"],
        })
    cl = [p for p in pts if p["closed_loop_capable"]]
    for p in pts:
        if not p["closed_loop_capable"]:
            p["pareto_optimal"] = False
            continue
        dominated = any(
            q is not p and q["per_step_s"] <= p["per_step_s"]
            and q["nmse"] <= p["nmse"]
            and (q["per_step_s"] < p["per_step_s"] or q["nmse"] < p["nmse"])
            for q in cl)
        p["pareto_optimal"] = not dominated
    return pts


# ------------------------------------------------------------- s26.4 verdicts
def deployment_verdicts(totals, deadlines=None, horizon_label="H_eff"):
    """Per-mode practicality (s26.4): latency of a ``horizon_label`` forecast vs deadline.

    Modes:
      research_sim     -- strategy D (local, no QPU): always available for study.
      delayed_batch    -- strategy B on a queued QPU, judged against the daily deadline.
      operational      -- strategy B on a live QPU, judged against each use-case deadline.
    The point (spec soundness note): for a MONTHLY index a 12-step forecast tolerates
    minutes of latency, so "operationally infeasible" is NOT automatic -- the honest
    frame is latency-vs-decision-deadline per use case.
    """
    deadlines = deadlines or DECISION_DEADLINES_S
    d_horizon = totals["D_local_noisy_sim"]["totals_s"][horizon_label]
    b_horizon = totals["B_pretranspiled_templates"]["totals_s"][horizon_label]

    def _feasible(latency_s):
        return {uc: {"deadline_s": dl, "latency_s": latency_s,
                     "feasible": bool(latency_s <= dl)}
                for uc, dl in deadlines.items()}

    return {
        "horizon_label": horizon_label,
        "research_sim": {
            "strategy": "D_local_noisy_sim", "latency_s": d_horizon,
            "verdict": "practical: fully local, seconds; the study path.",
            "by_use_case": _feasible(d_horizon)},
        "delayed_batch": {
            "strategy": "B_pretranspiled_templates", "latency_s": b_horizon,
            "verdict": ("practical for overnight/daily reporting: a whole-horizon "
                        "forecast completes well inside a daily batch window."),
            "by_use_case": _feasible(b_horizon)},
        "operational": {
            "strategy": "B_pretranspiled_templates", "latency_s": b_horizon,
            "verdict": ("latency-vs-deadline: feasible for monthly/daily indices "
                        "(minutes << cycle) but NOT for a sub-minute loop; the binding "
                        "constraint for monthly climate work is QPU access/cost, not "
                        "per-forecast latency."),
            "by_use_case": _feasible(b_horizon)},
    }
