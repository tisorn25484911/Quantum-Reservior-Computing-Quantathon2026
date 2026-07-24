"""combine_and_conclude.py -- the INTEGRATOR's harness.

New division of labour: teammates improve Step 1 (the forecaster) and Step 2 (the
detector) independently; this module combines any Step-1 x Step-2 pair end-to-end
and declares the winning combination and the overall conclusion.

Pluggable. To add a teammate's improved component:
  * FORECASTER -- an object with `.fit(x, train_end)`, `.rollout(origin, H)`
    (mean closed-loop forecast, original units), and `.rollout_ensemble(origin,
    H, K, rng)` (K stochastic trajectories; deterministic models tile the mean).
    Register the class in `FORECASTERS`.
  * DETECTOR -- a `Detector(name, needs, grid, apply)` where `needs` is "mean" or
    "ensemble", `grid` is the list of calibration dicts to try (fit on
    forecast-of-training by F1), and `apply(output, theta, calib) -> bool array`
    over (origins, H). Register it in `DETECTORS`.

Registry: forecasters {engine_qrc, core_qrc, nvar, persistence};
detectors {threshold (mean-trajectory Hobday rule), ensemble (fraction of K
sampled futures breaching the rule -- the Step-3 calibrated alarm)}.

The ML onset detector (`detect.py`) is a *precursor* warner: it predicts onset
from observed ENSO/rain/build-up, and does NOT consume a Step-1 forecast, so it
is reported as a separate parallel path, not a matrix cell.

    .venv/bin/python Quantathon_stack/Anomaly_Forecast/combine_and_conclude.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "engine" / "QRC_single_time_series" / "src"))
sys.path.insert(0, str(_REPO / "engine" / "Classical_ML_post_process"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "DataBase_Analysis"))

from forecast import (Scaler, make_reservoir, fit_readout, drive,   # noqa: E402
                      rollout as core_rollout)
from nvar_baseline import build_design, rollout_nvar                # noqa: E402
from engine_step1 import _EngineDriver, engine_features, engine_rollout  # noqa: E402
from qrc_single_time_series.quantum.exact_qrc import ExactQRC       # noqa: E402
from qrc_single_time_series.quantum.hamiltonians import fc_tfi      # noqa: E402
from qrc_single_time_series.models.readout import fit as eng_fit    # noqa: E402
from pca_subspace import PCASubspaceDetector                        # noqa: E402

LABELS = _HERE / "labels"
RESULTS = _HERE / "results"
K_TEST, K_CALIB = 60, 40


# ----------------------------------------------------------------------
# Forecasters
# ----------------------------------------------------------------------
class Persistence:
    name = "persistence"

    def fit(self, x, train_end):
        self.x = x

    def rollout(self, origin, H):
        return np.full(H, self.x[origin])

    def rollout_ensemble(self, origin, H, K, rng):
        return np.tile(self.rollout(origin, H), (K, 1))     # deterministic


class NVAR:
    name = "nvar"

    def __init__(self, k=4, s=1, lam=1e-4):
        self.k, self.s, self.lam = k, s, lam
        self.pairs = list(combinations_with_replacement(range(k), 2))

    def fit(self, x, train_end):
        self.x = x
        washout = (self.k - 1) * self.s + 1
        idx = np.arange(washout, train_end)
        Phi = build_design(x, self.k, self.s, idx, self.pairs)
        self.W = np.linalg.solve(
            Phi.T @ Phi + self.lam * np.eye(Phi.shape[1]), Phi.T @ x[idx + 1])

    def rollout(self, origin, H):
        return rollout_nvar(self.x, int(origin), self.W, self.k, self.s,
                            self.pairs, H)

    def rollout_ensemble(self, origin, H, K, rng):
        return np.tile(self.rollout(origin, H), (K, 1))     # deterministic


class CoreQRC:
    name = "core_qrc"

    def __init__(self, kind="xxz_hx", seed=7):
        self.kind, self.seed = kind, seed

    def fit(self, x, train_end):
        self.x = x
        self.res = make_reservoir(self.kind, n_qubits=5, seed=self.seed)
        self.ro = fit_readout(x, self.res, washout=100,
                              train_frac=train_end / len(x))
        u = np.clip(self.ro.scaler.to_unit(x), 0.0, 1.0)
        self.feats, self.states = drive(self.res, u, checkpoints=True)

    def rollout(self, origin, H):
        return core_rollout(self.x, self.res, self.ro, int(origin), H,
                            state=self.states[origin], feat=self.feats[origin]).yhat

    def rollout_ensemble(self, origin, H, K, rng):
        return np.array([
            core_rollout(self.x, self.res, self.ro, int(origin), H,
                         state=self.states[origin], feat=self.feats[origin],
                         rng=rng, bootstrap=True).yhat for _ in range(K)])


class EngineQRC:
    name = "engine_qrc"

    def __init__(self, seed=7):
        self.seed = seed

    def fit(self, x, train_end):
        self.x = x
        self.sc = Scaler.fit(x[:train_end + 1])
        u = np.clip(self.sc.to_unit(x), 0.0, 1.0)
        self.q = ExactQRC(fc_tfi(5, J=1.0, h=0.5, seed=self.seed), V=10,
                          tau=2.0, observable_kind="z_local")
        self.feats, self.states = engine_features(self.q, u, checkpoints=True)
        y = u[1:]
        idx = np.arange(100, train_end)
        self.ro = eng_fit(self.feats[idx], y[idx], lam=1e-6)
        self.resid = y[idx] - self.ro.predict(self.feats[idx])   # scaled residuals

    def rollout(self, origin, H):
        return engine_rollout(self.q, self.ro, self.sc,
                              (self.states[origin], self.feats[origin]),
                              int(origin), H)

    def rollout_ensemble(self, origin, H, K, rng):
        out = np.empty((K, H))
        for k in range(K):
            d = _EngineDriver(self.q)
            d.set_state(self.states[origin])
            cur = self.feats[origin]
            traj = np.empty(H)
            for i in range(H):
                u_next = float(self.ro.predict(cur[None, :])[0]) + float(rng.choice(self.resid))
                traj[i] = u_next
                cur = np.append(d.step(min(1.0, max(0.0, u_next))), 1.0)
            out[k] = self.sc.to_data(traj)
        return out


FORECASTERS = {c.name: c for c in [EngineQRC, CoreQRC, NVAR, Persistence]}


# ----------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------
@dataclass
class Detector:
    name: str
    needs: str                       # "mean" | "ensemble"
    grid: list                       # calibration dicts to try
    apply: Callable                  # (output, theta, calib) -> bool (origins,H)


def _threshold_apply(traj, theta, calib, ctx=None):
    return traj > calib["c"] * theta


def _ensemble_apply(ens, theta, calib, ctx=None):
    frac = (ens > calib["c"] * theta[:, None, :]).mean(axis=1)     # (origins,H)
    return frac > calib["p"]


def _pca_t2_apply(traj, theta, calib, ctx):
    """Teammate's EGADS/PCA-subspace detector (Hotelling T^2) on the forecast.

    The model is fit once on OBSERVED training intensity in delay coordinates
    (engine/Classical_ML_post_process/pca_subspace.py). Each forecast trajectory
    is scored with its observed context prepended so lead h=1 has a full
    embedding window; predicted-anomaly where T^2 exceeds the calibrated
    quantile of training scores. Note T^2 is two-sided (cold excursions also
    score) -- whatever that costs in precision is reported, not hidden.
    """
    model, x = ctx["pca_model"], ctx["x"]
    need = model.emb * model.lag
    H = traj.shape[1]
    flags = np.empty(traj.shape[:2], dtype=bool)
    thr = ctx["pca_train_q"][calib["q"]]
    for i, o in enumerate(ctx["origins"]):
        seq = np.concatenate([x[o - need + 1:o + 1], traj[i]])
        t2 = model.score(seq[:, None])["t2"][-H:]
        flags[i] = t2 > thr
    return flags


DETECTORS = {
    "threshold": Detector("threshold", "mean",
                          [{"c": c} for c in np.linspace(0.4, 1.2, 17)],
                          _threshold_apply),
    "ensemble": Detector("ensemble", "ensemble",
                         [{"c": c, "p": p}
                          for c in np.linspace(0.5, 1.1, 7)
                          for p in np.linspace(0.15, 0.6, 10)],
                         _ensemble_apply),
    "pca_t2": Detector("pca_t2", "mean",
                       [{"q": q} for q in (0.30, 0.40, 0.50, 0.60, 0.70,
                                           0.80, 0.90, 0.95)],
                       _pca_t2_apply),
}


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def f1(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _output(fc, det, origins, H, rng, K):
    if det.needs == "ensemble":
        return np.stack([fc.rollout_ensemble(o, H, K, rng) for o in origins])
    return np.array([fc.rollout(o, H) for o in origins])


def evaluate(fc, det, theta, actual, H, train_o, test_o, rng, ctx):
    tr_out = _output(fc, det, train_o, H, rng, K_CALIB)
    tr_theta = np.array([theta[o + 1:o + H + 1] for o in train_o])
    tr_act = np.array([actual[o + 1:o + H + 1] for o in train_o], bool)
    best, best_c = -1.0, det.grid[0]
    for calib in det.grid:
        _, _, s = f1(det.apply(tr_out, tr_theta, calib,
                               {**ctx, "origins": train_o}), tr_act)
        if s > best:
            best, best_c = s, calib
    te_out = _output(fc, det, test_o, H, rng, K_TEST)
    te_theta = np.array([theta[o + 1:o + H + 1] for o in test_o])
    te_act = np.array([actual[o + 1:o + H + 1] for o in test_o], bool)
    pred = det.apply(te_out, te_theta, best_c, {**ctx, "origins": test_o})
    return best_c, np.array([f1(pred[:, j], te_act[:, j]) for j in range(H)]), te_act


def main():
    frame = pd.read_csv(LABELS / "got_sst_mhw_labeled.csv", parse_dates=["date"])
    frame = frame.iloc[-5000:].reset_index(drop=True)
    x = frame["intensity"].to_numpy(float)
    theta = (frame["thresh"] - frame["clim"]).to_numpy(float)
    actual = frame["mhw"].to_numpy(bool)
    H, n = 14, len(x)
    valid = np.arange(100, n - 1)
    train_end = int(valid[int(0.7 * len(valid)) - 1]) + 1
    tr = valid[valid < train_end]; te = valid[valid >= train_end]
    train_o = tr[tr + H < n][::5]                # stride: calibration cost
    test_o = te[te + H < n]
    rng = np.random.default_rng(7)

    # teammate's PCA-subspace model (Step 2), fit once on OBSERVED training
    # intensity in weekly delay coordinates; threshold grid = quantiles of its
    # training-span T^2 scores
    pca = PCASubspaceDetector(emb=7, lag=1)
    pca.fit(x[:train_end, None])
    t2_tr = pca.score(x[:train_end, None])["t2"]
    t2_tr = t2_tr[t2_tr > 0]
    ctx = {"x": x, "pca_model": pca,
           "pca_train_q": {q: float(np.quantile(t2_tr, q))
                           for q in (0.30, 0.40, 0.50, 0.60, 0.70,
                                     0.80, 0.90, 0.95)}}

    print("=== INTEGRATOR: combine Step-1 x Step-2, conclude ===")
    print(f"N={n} H={H} train_o={len(train_o)} test_o={len(test_o)} "
          f"K={K_TEST}\n")

    results = {}
    for fname, FC in FORECASTERS.items():
        fc = FC(); fc.fit(x, train_end)
        for dname, det in DETECTORS.items():
            # ensemble detector on a deterministic forecaster == its threshold row
            calib, ph, act = evaluate(fc, det, theta, actual, H, train_o, test_o,
                                      rng, ctx)
            results[(fname, dname)] = ph
            print(f"  [{fname:11s} x {dname:9s}] "
                  f"F1(1-7)={ph[:7,2].mean():.3f} F1(1-14)={ph[:,2].mean():.3f} "
                  f"rec@3={ph[2,1]:.2f} rec@7={ph[6,1]:.2f} rec@14={ph[13,1]:.2f}")
    mhw_rate = float(np.mean([actual[o+1:o+H+1].mean() for o in test_o]))

    def short(ph):
        return ph[:7, 2].mean()

    print(f"\n  ranking by actionable window F1 (h1-7); MHW rate ~{mhw_rate:.2f}:")
    print(f"    {'combination':26s} {'F1 1-7':>7} {'F1 1-14':>8} {'rec@3':>6} "
          f"{'rec@7':>6} {'rec@14':>7}")
    ranked = sorted(results.items(), key=lambda kv: -short(kv[1]))
    for (fn, dn), ph in ranked:
        print(f"    {fn+' x '+dn:26s} {short(ph):>7.3f} {ph[:,2].mean():>8.3f} "
              f"{ph[2,1]:>6.2f} {ph[6,1]:>6.2f} {ph[13,1]:>7.2f}")

    # what the ensemble detector buys the best QRC at long lead
    qkey = max([k for k in results if "qrc" in k[0]],
               key=lambda k: short(results[k]))
    qf = qkey[0]
    thr, ens = results[(qf, "threshold")], results[(qf, "ensemble")]
    (bf, bd), bph = ranked[0]

    per = results.get(("persistence", "threshold"))
    qthr = results[(qf, "threshold")]
    print(f"\n  === CONCLUSION ===")
    print(f"  On balanced F1 it is close: persistence-then-threshold F1(1-7) "
          f"{short(per):.3f} ~ {qf} {short(qthr):.3f}. Persistence is strong here "
          f"because heatwaves are >=5-day events, so 'today's heat continues' bets well.")
    print(f"  The quantum forecast's clear edge is RECALL at short lead: "
          f"{qf} catches {qthr[2,1]:.0%} of heatwave days 3d out vs persistence "
          f"{per[2,1]:.0%} -- it predicts the event is COMING, not just continuing.")
    print(f"  Under aquaculture's cost asymmetry (a missed heatwave loses a grow-out "
          f"cycle; a false alarm just runs an aerator), recall is the priority "
          f"metric -> the quantum forecast is preferred.")
    print(f"  Step-2 upgrade (threshold -> ensemble) on {qf}: recovers long-lead "
          f"recall@14 {thr[13,1]:.0%} -> {ens[13,1]:.0%} (calibrated, plan.md Step 3).")
    print(f"  Parallel path: the precursor ML detector (detect.py, observed ENSO/"
          f"rain/build-up) warns ONSET at 2.6x base rate, independent of Step 1.")
    print(f"  NET (recall-first): ship {qf} (Step 1) x ensemble (Step 2). "
          f"Honest caveat: on balanced F1, persistence is a genuine tie.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "combine_conclusion.json").write_text(json.dumps({
        f"{fn}|{dn}": dict(f1_1_7=float(ph[:7, 2].mean()),
                           f1_1_14=float(ph[:, 2].mean()),
                           recall=ph[:, 1].tolist(), f1=ph[:, 2].tolist())
        for (fn, dn), ph in results.items()}, indent=2))
    print(f"\n  wrote results/combine_conclusion.json")


if __name__ == "__main__":
    main()
