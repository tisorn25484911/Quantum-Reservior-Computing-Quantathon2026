"""engine.py -- the forecasting pipeline the demo actually runs.

One walk of a time series through the project's reference quantum reservoir:

    series -> train-span scaler -> QuantumReservoir -> ridge read-out
           -> split-conformal band -> held-out metrics + baselines

The reservoir is ``stage1_numpy_core.QuantumReservoir``, the exact
density-matrix simulation that repository law R2 designates as the semantic
definition for the whole programme. Nothing is mocked: driving it here runs the
same propagator the handbook's numbers came from. It is *classical simulation
of* a quantum system, not quantum hardware, and the UI says so.

Three things in here exist purely to stop the demo flattering itself:

* **The scaler is fitted on the training span only.** Fitting min/max over the
  whole series leaks the test range backwards into training and inflates skill.
* **Baselines run on the identical split.** Persistence and a size-matched ESN
  are computed from the same feature-count budget and scored on the same rows,
  so the comparison is like-for-like.
* **Bands are split-conformal**, calibrated on a held-out slice that the ridge
  read-out never saw, so the coverage number means something.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np

from . import metrics
from .catalog import series_values
from ..config import SEED

from qrc_core import QuantumReservoir, ESN, ridge_fit, ridge_predict  # noqa: E402


ProgressFn = Callable[[float, str], None]


@dataclass
class ForecastConfig:
    """Everything that determines a run. Echoed into the model card verbatim."""

    dataset: str = "solar"
    horizon: int = 1                 # steps ahead
    n_qubits: int = 5
    dt: float = 2.0                  # input interval J*dt (the temporal edge)
    virtual_nodes: int = 4
    use_zz: bool = True
    seed: int = SEED
    max_points: int = 2000
    train_frac: float = 0.6
    calib_frac: float = 0.2          # remainder is test
    washout: int = 100
    lam: float = 1e-6
    alpha: float = 0.1               # 1 - alpha nominal band coverage

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1 step")
        if not 1 <= self.n_qubits <= 8:
            raise ValueError("n_qubits must be between 1 and 8 "
                             "(exact simulation cost is 4^n)")
        if self.virtual_nodes < 1:
            raise ValueError("virtual_nodes must be at least 1")
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError("train_frac must lie in (0, 1)")
        if not 0.0 < self.calib_frac < 1.0:
            raise ValueError("calib_frac must lie in (0, 1)")
        if self.train_frac + self.calib_frac >= 1.0:
            raise ValueError("train_frac + calib_frac must leave a test split")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.max_points < 300:
            raise ValueError("max_points must be at least 300 to leave a split")


@dataclass
class Split:
    """Chronological index ranges. Never shuffled -- this is a time series."""

    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict:
        return {"train": int(len(self.train)), "calib": int(len(self.calib)),
                "test": int(len(self.test))}


def _chronological_split(valid: np.ndarray, train_frac: float,
                         calib_frac: float) -> Split:
    n = len(valid)
    n_tr = int(train_frac * n)
    n_ca = int(calib_frac * n)
    if min(n_tr, n_ca, n - n_tr - n_ca) < 10:
        raise ValueError(
            f"series too short for a 3-way split: {n} usable steps gives "
            f"train={n_tr}, calib={n_ca}, test={n - n_tr - n_ca}")
    return Split(valid[:n_tr], valid[n_tr:n_tr + n_ca], valid[n_tr + n_ca:])


def _fit_predict_conformal(X: np.ndarray, y: np.ndarray, split: Split,
                           lam: float, alpha: float):
    """Ridge on train, conformal radius on calib, prediction on test.

    The radius is the ``1 - alpha`` empirical quantile of absolute calibration
    residuals, which gives a constant-width band with approximately ``1 - alpha``
    marginal coverage under exchangeability. That assumption is not exactly true
    for a time series, which is why the realised coverage is measured and shown
    rather than asserted.
    """
    w = ridge_fit(X[split.train], y[split.train], lam=lam)
    resid = np.abs(y[split.calib] - ridge_predict(X[split.calib], w))
    # Finite-sample conformal quantile.
    n = len(resid)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    radius = float(np.quantile(resid, q_level))
    pred = ridge_predict(X[split.test], w)
    return pred, radius, w


def run_forecast(cfg: ForecastConfig, progress: ProgressFn | None = None
                 ) -> dict:
    """Execute one forecast run and return a fully JSON-serialisable result."""
    cfg.validate()
    say = progress or (lambda frac, msg: None)

    say(0.02, "loading series")
    x, meta = series_values(cfg.dataset, max_points=cfg.max_points)
    T = len(x)
    h = cfg.horizon

    # ---- valid target range, then the split -------------------------------
    valid = np.arange(cfg.washout, T - h)
    if len(valid) < 60:
        raise ValueError(
            f"after washout={cfg.washout} and horizon={h}, only {len(valid)} "
            "usable steps remain; shorten the washout or the horizon")
    split = _chronological_split(valid, cfg.train_frac, cfg.calib_frac)

    # ---- scaler fitted on the TRAIN SPAN ONLY (leakage guard) -------------
    # The reservoir is driven by the whole series (it is causal, so that is
    # fine), but the scaling constants may only see training-era data.
    say(0.08, "scaling on train span")
    train_end = int(split.train[-1]) + 1
    lo, hi = float(np.min(x[:train_end])), float(np.max(x[:train_end]))
    span = hi - lo
    if span < 1e-12:
        raise ValueError("training span is constant; nothing to learn from")
    u = np.clip((x - lo) / span, 0.0, 1.0)

    # ---- quantum reservoir ------------------------------------------------
    say(0.15, f"driving {cfg.n_qubits}-qubit reservoir ({T} steps)")
    qr = QuantumReservoir(n_qubits=cfg.n_qubits, dt=cfg.dt,
                          virtual_nodes=cfg.virtual_nodes,
                          use_zz=cfg.use_zz, seed=cfg.seed)
    X_q = qr.run(u)

    y = np.empty(T)
    y[:T - h] = x[h:]
    y[T - h:] = np.nan                      # never indexed: valid stops at T-h

    say(0.55, "fitting read-out + conformal band")
    pred_q, rad_q, _ = _fit_predict_conformal(X_q, y, split, cfg.lam, cfg.alpha)
    y_test = y[split.test]
    lo_q, hi_q = pred_q - rad_q, pred_q + rad_q
    m_q = metrics.summarise(y_test, pred_q, lo_q, hi_q)

    # ---- baselines on the identical split ---------------------------------
    say(0.7, "persistence baseline")
    pred_p = x[split.test]                  # last observed value, h steps back
    m_p = metrics.summarise(y_test, pred_p)

    say(0.78, f"size-matched ESN baseline ({qr.n_features} nodes)")
    esn = ESN(qr.n_features, seed=cfg.seed + 4)
    X_e = esn.run(u)
    pred_e, rad_e, _ = _fit_predict_conformal(X_e, y, split, cfg.lam, cfg.alpha)
    m_e = metrics.summarise(y_test, pred_e, pred_e - rad_e, pred_e + rad_e)

    say(0.9, "scoring")
    for m, p in ((m_q, pred_q), (m_e, pred_e)):
        m["skill_vs_persistence"] = metrics.skill(m["rmse"], m_p["rmse"])
    m_p["skill_vs_persistence"] = 0.0

    verdict = _verdict(m_q, m_e, m_p, cfg.alpha)

    say(1.0, "done")
    return {
        "config": asdict(cfg),
        "dataset": meta,
        "n_features": int(qr.n_features),
        "split": split.sizes(),
        "scaler": {"lo": lo, "hi": hi, "fitted_on": "train span only"},
        "conformal": {"nominal": 1.0 - cfg.alpha, "radius_qrc": rad_q,
                      "radius_esn": float(rad_e)},
        "metrics": {"qrc": m_q, "esn": m_e, "persistence": m_p},
        "verdict": verdict,
        # Absolute index boundaries of each chronological zone, so a plot can
        # shade exactly what the read-out was trained on (train), what set the
        # conformal radius (calib), and what was scored (test). These are ORIGIN
        # indices; each origin's target sits `horizon` steps later.
        "zones": {
            "washout": int(cfg.washout),
            "train_end": int(split.train[-1]) + 1,
            "calib_end": int(split.calib[-1]) + 1,
            "test_end": int(split.test[-1]) + 1,
            "T": int(T),
            "horizon": int(h),
        },
        "series": {
            "test_index": split.test.tolist(),
            "y_true": y_test.tolist(),
            "qrc": pred_q.tolist(),
            "qrc_lo": lo_q.tolist(),
            "qrc_hi": hi_q.tolist(),
            "esn": pred_e.tolist(),
            "persistence": pred_p.tolist(),
            # Whole observed series (already capped at max_points) so the plot
            # can draw the train/calib/test context, not just the test window.
            "observed_full": x.tolist(),
        },
    }


def _verdict(m_q: dict, m_e: dict, m_p: dict, alpha: float) -> dict:
    """The plain-language reading of the run, including the unflattering cases.

    Written so the honest outcome is the default one. A demo that only has
    language for success will describe a failure as a success.
    """
    notes: list[str] = []
    beat_mean = m_q["nmse"] < 1.0
    beat_persist = m_q["rmse"] < m_p["rmse"]
    vs_esn = m_q["rmse"] / m_e["rmse"] if m_e["rmse"] > 0 else float("nan")

    if not beat_mean:
        headline = "Reservoir failed: worse than predicting the mean"
        notes.append(
            "NMSE >= 1 means the read-out carries no usable signal on this "
            "split. Treat every other number on this page as void.")
    elif not beat_persist:
        headline = "Reservoir learned, but does not beat persistence"
        notes.append(
            "Persistence (repeat the last observation) is the honest floor "
            "for a smooth series. Losing to it means the model adds nothing "
            "operationally, however good its NMSE looks.")
    else:
        headline = "Reservoir beats persistence on held-out data"

    if np.isfinite(vs_esn):
        if vs_esn <= 0.97:
            notes.append(
                f"QRC edges the size-matched ESN (RMSE ratio {vs_esn:.3f}). "
                "With one seed and one split this is not a significant "
                "difference -- it is parity, not advantage.")
        elif vs_esn >= 1.03:
            notes.append(
                f"The size-matched ESN beats QRC (RMSE ratio {vs_esn:.3f}). "
                "This is the expected outcome at these sizes and is reported "
                "rather than hidden.")
        else:
            notes.append(
                f"QRC and the size-matched ESN are within 3% (ratio "
                f"{vs_esn:.3f}) -- parity on matched trained parameters.")

    # Conformal coverage assumes exchangeability, which a time series violates;
    # a large shortfall is the signal that the assumption broke here.
    cov, nominal = m_q.get("coverage"), 1.0 - alpha
    calibrated = True
    if cov is not None:
        if cov < nominal - 0.10:
            calibrated = False
            notes.append(
                f"Band under-covers: {cov:.0%} of outcomes fell inside a "
                f"{nominal:.0%} interval. Conformal calibration assumes "
                "exchangeability, which a drifting time series breaks -- the "
                "interval is optimistic.")
        elif cov > nominal + 0.10:
            notes.append(
                f"Band over-covers ({cov:.0%} vs {nominal:.0%} nominal): "
                "intervals are wider than they need to be, so coverage is "
                "being bought with width.")

    return {"headline": headline, "notes": notes,
            "beat_mean_predictor": bool(beat_mean),
            "beat_persistence": bool(beat_persist),
            "band_calibrated": bool(calibrated),
            "rmse_ratio_vs_esn": (float(vs_esn) if np.isfinite(vs_esn)
                                  else None)}


# ----------------------------------------------------------------------
# Horizon sweep
# ----------------------------------------------------------------------
def run_horizon_sweep(cfg: ForecastConfig, horizons: list[int] | None = None,
                      progress: ProgressFn | None = None) -> dict:
    """Re-run the forecast at each horizon to find where skill runs out.

    The operationally important question is not "how good is the forecast" but
    "how far ahead is it still worth anything", and that is a curve, not a
    number. Reading it against the mean-predictor line at NRMSE = 1 gives the
    usable horizon directly; for the chaotic tier it can be compared with the
    Lyapunov horizon that ``DataBase_Analysis`` estimates independently.
    """
    cfg.validate()
    say = progress or (lambda frac, msg: None)
    hs = sorted(set(int(h) for h in (horizons or [1, 2, 3, 6, 12, 24])))
    if any(h < 1 for h in hs):
        raise ValueError("horizons must all be >= 1")

    out = {"qrc": [], "esn": [], "persistence": []}
    rmse_out = {"qrc": [], "esn": [], "persistence": []}
    done_h, verdicts = [], []

    for i, h in enumerate(hs):
        say(i / len(hs), f"horizon {h} ({i + 1} of {len(hs)})")
        sub = ForecastConfig(**{**asdict(cfg), "horizon": h})
        try:
            r = run_forecast(sub)
        except ValueError:
            # A horizon too long for the series is a legitimate stop, not a
            # crash: record what completed and move on.
            continue
        done_h.append(h)
        for k in out:
            out[k].append(r["metrics"][k]["nrmse"])
            rmse_out[k].append(r["metrics"][k]["rmse"])
        verdicts.append(r["verdict"]["headline"])

    if not done_h:
        raise ValueError("no horizon in the requested set fits this series")

    # First horizon at which QRC no longer beats the mean predictor.
    lost = next((h for h, v in zip(done_h, out["qrc"]) if v >= 1.0), None)
    _, meta = series_values(cfg.dataset, max_points=cfg.max_points)

    say(1.0, "done")
    return {"config": asdict(cfg), "dataset": meta, "horizons": done_h,
            "nrmse": out, "rmse": rmse_out, "verdicts": verdicts,
            "usable_horizon": (None if lost is None else int(lost)),
            "usable_note": (
                f"QRC still beats the mean predictor at every horizon tested "
                f"(up to {done_h[-1]})." if lost is None else
                f"QRC stops beating the mean predictor at horizon {lost}; "
                f"beyond that the forecast carries no usable signal.")}


# ----------------------------------------------------------------------
# Reservoir diagnostic: memory capacity
# ----------------------------------------------------------------------
def run_memory_capacity(cfg: ForecastConfig, max_delay: int = 25,
                        n_steps: int = 2500,
                        progress: ProgressFn | None = None) -> dict:
    """Linear memory capacity of the configured reservoir.

    Deliberately driven by an **i.i.d. uniform stream, not the dataset**.
    Memory capacity is only defined for an i.i.d. drive: an autocorrelated
    series lets a memoryless read-out infer past inputs from the present one,
    which inflates the number without the reservoir remembering anything. So
    this is a property of the reservoir, measured on its own terms, and it does
    not change when you switch datasets.
    """
    cfg.validate()
    say = progress or (lambda frac, msg: None)

    from memory_capacity import linear_memory_capacity, esn_features

    say(0.05, "generating i.i.d. drive")
    rng = np.random.default_rng(cfg.seed)
    u = rng.uniform(0.0, 1.0, size=int(n_steps))

    say(0.15, f"driving {cfg.n_qubits}-qubit reservoir")
    qr = QuantumReservoir(n_qubits=cfg.n_qubits, dt=cfg.dt,
                          virtual_nodes=cfg.virtual_nodes,
                          use_zz=cfg.use_zz, seed=cfg.seed)
    X_q = qr.run(u)

    say(0.5, "measuring memory capacity (QRC)")
    res_q = linear_memory_capacity(X_q, u, max_delay=max_delay)

    say(0.75, "measuring memory capacity (size-matched ESN)")
    X_e = esn_features(u, n_nodes=qr.n_features, seed=cfg.seed + 4)
    res_e = linear_memory_capacity(X_e, u, max_delay=max_delay)

    say(1.0, "done")

    def pack(r):
        return {"mc": r.mc, "mc_raw": r.mc_raw, "noise_floor": r.noise_floor,
                "n_features": r.n_features, "rank": r.rank,
                "effective_rank": r.effective_rank, "saturated": r.saturated,
                "warnings": r.warnings, "delays": r.delays.tolist(),
                "mf": r.mf.tolist()}

    return {"config": asdict(cfg), "max_delay": max_delay, "n_steps": n_steps,
            "drive": "i.i.d. uniform [0,1] (required for MC to be defined)",
            "qrc": pack(res_q), "esn": pack(res_e)}
