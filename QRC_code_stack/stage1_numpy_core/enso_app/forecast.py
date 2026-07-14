"""forecast.py -- CLI entry point: train the QRC at the selected operating
point, forecast the test span (+ 3 genuinely future months), write
outputs/forecasts.csv, outputs/metrics.json, and a trajectory figure.

Defaults are the Phase-5 selected config (gamma=pi/4, ent_scale=0.5,
chosen by inner-validation sweep in code/experiments.py -- the test span
played no part in that choice).

Prediction bands: residual quantiles (default 10/90) from an INNER
VALIDATION block within the train span (fit on inner-train, residuals on
inner-val -- out-of-sample, leak-free), applied around predictions of a
model refit on the full train span. Coverage on the test span is
reported honestly in metrics.json.

Modes: exact (NumPy reference, default) | sampled (noiseless Aer,
finite shots) | noisy (Aer + depolarizing/readout, ECR/RZ/SX/X, ring).
Backend 'ibm' exists but is DORMANT (needs IBM_QUANTUM_TOKEN).
All results are simulation; the ENSO data is real.

Usage:
    python forecast.py                          # exact, full battery
    python forecast.py --mode noisy --shots 4096 --tag noisy_S4096
    python forecast.py --no-battery --bands 5,95
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
# qrc-stack layout shim (Phase 2): sibling modules live in this directory;
# the Qiskit port lives in stage2_circuits_exact/. Old flat-layout insert
# of REPO/"code" replaced by the stage-2 path.
sys.path.insert(0, str(REPO.parents[1] / "stage2_circuits_exact"))

import datasets                                            # noqa: E402
import figstyle                                            # noqa: E402
from baselines import (EVENT_THRESHOLD, HORIZON, L, chrono_split,  # noqa: E402
                       make_targets, run_battery, score)
from qrc_core import (SEED, WindowedReservoir, add_bias,   # noqa: E402
                      ridge_gcv)
from qrc_qiskit import (NOISE_P1, NOISE_P2, NOISE_RO,      # noqa: E402
                        QiskitReservoir, get_backend)

import matplotlib.pyplot as plt                            # noqa: E402

LAMS = np.logspace(-8, 4, 25)
INNER_TRAIN_FRAC = 0.8


# ------------------------------------------------------------------ fitting
class RidgeModel:
    """Standardise by the fit rows, GCV ridge; predicts arbitrary rows."""

    def fit(self, X, y):
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.w, self.lam = ridge_gcv(add_bias((X - self.mu) / self.sd),
                                     y, LAMS)
        return self

    def predict(self, X):
        return add_bias((X - self.mu) / self.sd) @ self.w


def compute_features(u_scaled, cfg):
    """Feature rows for EVERY window k = L-1 .. T-1 (incl. future windows).
    Returns (Xall, meta_str)."""
    res = WindowedReservoir(gamma=cfg.gamma, seed=cfg.seed,
                            ent_scale=cfg.ent_scale)
    if cfg.mode == "exact":
        return res.feature_matrix(u_scaled, L), "exact expectation (S=inf)"
    qr = QiskitReservoir(res=res)
    be = get_backend(cfg.backend, noise=(cfg.mode == "noisy"), seed=cfg.seed)
    Xall, meta = qr.sampled_feature_matrix(u_scaled, L, shots=cfg.shots,
                                           backend=be)
    lab = (f"{cfg.mode}, S={meta['shots_per_basis']}/basis, "
           f"{meta['circuits_per_window']} circuits/window")
    if cfg.mode == "noisy":
        lab += f", noise p1={NOISE_P1} p2={NOISE_P2} ro={NOISE_RO}"
    return Xall, lab


# ------------------------------------------------------------------ pipeline
def run(cfg):
    figstyle.apply()
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (REPO / "figures").mkdir(exist_ok=True)
    tag = f"_{cfg.tag}" if cfg.tag else ""

    print(f"forecast config: mode={cfg.mode} backend={cfg.backend} "
          f"gamma={cfg.gamma:.4f} ent_scale={cfg.ent_scale} "
          f"shots={cfg.shots if cfg.mode != 'exact' else 'inf'} "
          f"bands={cfg.bands} seed={cfg.seed}")
    d = datasets.prepare()
    y, scaler, dates = d["y"], d["scaler"], d["df"].date
    u_scaled = scaler(y)
    T = len(y)

    ks, yt = make_targets(y)
    tr, te = chrono_split(len(ks))
    icut = int(len(tr) * INNER_TRAIN_FRAC)
    itr, ival = tr[:icut], tr[icut:]

    t0 = time.time()
    Xall, feat_label = compute_features(u_scaled, cfg)
    print(f"features: {feat_label} ({time.time() - t0:.1f} s, "
          f"{Xall.shape[0]} windows x {Xall.shape[1]})")
    Xs = Xall[ks - (L - 1)]                       # rows aligned with samples

    # bands: out-of-sample residual quantiles from the inner split
    band_model = RidgeModel().fit(Xs[itr], yt[itr])
    resid = yt[ival] - band_model.predict(Xs[ival])
    q_lo, q_hi = np.percentile(resid, cfg.bands)
    print(f"bands: inner-val residual quantiles p{cfg.bands[0]:g}/"
          f"p{cfg.bands[1]:g} = [{q_lo:+.3f}, {q_hi:+.3f}] C "
          f"(n_val={len(ival)}, band-model lambda={band_model.lam:.1e})")

    # final model on the full train span
    model = RidgeModel().fit(Xs[tr], yt[tr])
    pred_te = model.predict(Xs[te])
    sc = score(yt[te], pred_te, yt[tr])
    lo, hi = pred_te + q_lo, pred_te + q_hi
    coverage = float(np.mean((yt[te] >= lo) & (yt[te] <= hi)))
    nominal = (cfg.bands[1] - cfg.bands[0]) / 100
    ev = f"{sc['nmse_event']:.3f}" if sc["nmse_event"] else "n/a"
    print(f"test: NMSE={sc['nmse']:.3f} event-NMSE={ev} "
          f"(n={sc['n']}, n_event={sc['n_event']}, lambda={model.lam:.1e})")
    print(f"band coverage on test: {coverage:.3f} (nominal {nominal:.2f})")

    # genuinely future months: windows ending at T-H .. T-1
    ks_f = np.arange(T - HORIZON, T)
    pred_f = model.predict(Xall[ks_f - (L - 1)])
    dates_f = [dates.iloc[-1] + pd.DateOffset(months=int(k + HORIZON - (T - 1)))
               for k in ks_f]
    for dt_, p in zip(dates_f, pred_f):
        print(f"future forecast {dt_:%Y-%m}: {p:+.3f} C "
              f"[{p + q_lo:+.3f}, {p + q_hi:+.3f}]")

    # ---------------- outputs/forecasts.csv
    rows = pd.DataFrame({
        "date": [f"{t:%Y-%m}" for t in dates.iloc[ks[te] + HORIZON]]
                + [f"{t:%Y-%m}" for t in dates_f],
        "y_true": np.concatenate([yt[te], np.full(HORIZON, np.nan)]),
        "y_pred": np.concatenate([pred_te, pred_f]),
        "lo": np.concatenate([lo, pred_f + q_lo]),
        "hi": np.concatenate([hi, pred_f + q_hi]),
    })
    csv_path = outdir / f"forecasts{tag}.csv"
    rows.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"wrote {csv_path} ({len(rows)} rows: {len(te)} test + "
          f"{HORIZON} future)")

    # ---------------- outputs/metrics.json
    metrics = {
        "config": {
            "mode": cfg.mode, "backend": cfg.backend,
            "gamma": cfg.gamma, "ent_scale": cfg.ent_scale,
            "n_qubits": 5, "L": L, "H": HORIZON, "seed": cfg.seed,
            "shots_per_basis": None if cfg.mode == "exact" else cfg.shots,
            "noise": ({"p1": NOISE_P1, "p2": NOISE_P2, "ro": NOISE_RO}
                      if cfg.mode == "noisy" else None),
            "bands_percentiles": list(cfg.bands),
            "selection": "gamma, ent_scale by inner-val sweep "
                         "(code/experiments.py); test span untouched",
            "ridge_lambda": {"band_model": float(band_model.lam),
                             "final_model": float(model.lam)},
        },
        "data": {"source": "statsmodels elnino (NOAA monthly Nino SST, REAL)",
                 "months": int(T), "train_months": int(d["n_train_months"]),
                 "first_test_target": f"{dates.iloc[d['n_train_months']]:%Y-%m}"},
        "qrc_test": sc,
        "band_coverage_test": coverage,
        "future_forecasts": {f"{t:%Y-%m}": float(p)
                             for t, p in zip(dates_f, pred_f)},
        "scope": (("REAL HARDWARE (IBM)"
                   if cfg.backend == "ibm" and cfg.mode != "exact"
                   else "all results simulation") + "; ENSO data real"),
    }
    if not cfg.no_battery:
        print("running classical battery...")
        battery, _ = run_battery(y, scaler, seed=cfg.seed)
        metrics["battery"] = battery
        for name, s in battery.items():
            e = f"{s['nmse_event']:.3f}" if s["nmse_event"] else "n/a"
            print(f"  {name:12s} NMSE={s['nmse']:.3f} event={e}")
    json_path = outdir / f"metrics{tag}.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {json_path}")

    # ---------------- figure
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    t_te = dates.iloc[ks[te] + HORIZON]
    ax.fill_between(t_te, lo, hi, color="C0", alpha=0.2,
                    label=f"p{cfg.bands[0]:g}-p{cfg.bands[1]:g} band")
    ax.plot(t_te, yt[te], color="0.2", lw=1.0, label="true")
    ax.plot(t_te, pred_te, color="C0", lw=1.0, label="QRC forecast")
    ax.plot(dates_f, pred_f, "C3o", ms=4, label="future (no truth)")
    for s_ in (-EVENT_THRESHOLD, EVENT_THRESHOLD):
        ax.axhline(s_, color="C3", ls="--", lw=0.7)
    ax.set(xlabel="target month", ylabel="SST anomaly [C]",
           title=f"ENSO {HORIZON}-month-ahead, {cfg.mode} "
                 f"(test NMSE {sc['nmse']:.3f}) -- simulation")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig_path = REPO / "figures" / f"forecast_test_span{tag}.png"
    figstyle.save(fig, fig_path)
    return {"csv": csv_path, "json": json_path, "figure": fig_path,
            "metrics": metrics}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["exact", "sampled", "noisy"],
                   default="exact")
    p.add_argument("--backend", choices=["aer", "ibm"], default="aer",
                   help="'ibm' is a dormant real-hardware hook")
    p.add_argument("--shots", type=int, default=4096,
                   help="shots per basis circuit (sampled/noisy)")
    p.add_argument("--gamma", type=float, default=float(np.pi / 4),
                   help="encoding gain (default: Phase-5 selected)")
    p.add_argument("--ent-scale", type=float, default=0.5,
                   help="entangler scale s (default: Phase-5 selected)")
    p.add_argument("--bands", type=lambda s: tuple(float(x) for x in
                   s.split(",")), default=(10.0, 90.0),
                   help="residual-quantile percentiles, e.g. 10,90")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--outdir", default=str(REPO / "outputs"))
    p.add_argument("--tag", default="",
                   help="suffix for output filenames, e.g. noisy_S4096")
    p.add_argument("--no-battery", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
