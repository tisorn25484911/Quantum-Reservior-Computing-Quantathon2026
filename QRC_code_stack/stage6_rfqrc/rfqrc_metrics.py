"""rfqrc_metrics.py -- extreme-event evaluation battery (stage 6 Phase 3).

Conventions frozen in Part X (verified against Ahmed et al., PRR 6 043082
(2024), sec. on predictability horizon; live-checked 2026-07-14):

  * PH (predictability horizon), Racca-Magri convention: the prediction
    fails at the first time the normalised error
        E(t) = |k_pred(t) - k_true(t)| / (k_e - k_bar)
    exceeds 0.2, where k_bar is the time-averaged kinetic energy of the
    truth series. The ladder starts at 10 LT and decrements by 0.5 LT:
    PH(delta) is accepted at the largest delta for which the prediction
    survives to delta. (Implemented directly as first-crossing time,
    then snapped DOWN to the ladder grid -- equivalent and simpler; the
    anchor test pins the equivalence on a hand-computed series.)
  * Event F-score versus prediction-time offset: an extreme event is
    k >= k_e. For each offset bin, precision/recall/F1 of predicted
    event occurrence within the bin, over >= many starting points.
  * VPT (valid prediction time): first crossing of a normalised
    state-error threshold (Lorenz-style tasks).
  * Effective rank: participation ratio of the feature covariance --
    imported from stage 5 (single implementation).

Every function is pure numpy on arrays; no model code imported.
"""

from __future__ import annotations

import numpy as np

from qrc_experiment import effective_rank  # single implementation (stage 5)

__all__ = ["nmse", "predictability_horizon", "ph_ladder",
           "event_scores_vs_offset", "valid_prediction_time",
           "effective_rank"]


def nmse(y_true: np.ndarray, y_pred: np.ndarray,
         var_ref: float | None = None) -> float:
    """Variance-normalised MSE; mean predictor = 1.0 by construction."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    v = float(np.var(y_true)) if var_ref is None else float(var_ref)
    return float(np.mean((y_true - y_pred) ** 2) / v)


def predictability_horizon(k_true: np.ndarray, k_pred: np.ndarray,
                           k_e: float, dt_lt: float,
                           threshold: float = 0.2) -> float:
    """PH in Lyapunov times: first time E(t) exceeds `threshold`.

    dt_lt = time-step size expressed in Lyapunov times (dt / LT).
    k_bar is the mean of the TRUE series (Racca-Magri convention).
    Returns the full series length (in LT) if the error never crosses.
    """
    k_true = np.asarray(k_true, dtype=float)
    k_pred = np.asarray(k_pred, dtype=float)
    denom = k_e - float(np.mean(k_true))
    if abs(denom) < 1e-12:
        return 0.0
    err = np.abs(k_pred - k_true) / abs(denom)
    bad = np.nonzero(err > threshold)[0]
    n_ok = bad[0] if len(bad) else len(k_true)
    return float(n_ok * dt_lt)


def ph_ladder(ph_raw: float, start: float = 10.0, step: float = 0.5) -> float:
    """Snap a raw first-crossing PH DOWN onto the Part X ladder
    {start, start-step, ..., 0}: the largest ladder rung not exceeding
    ph_raw, capped at `start`. The ladder terminates at 0."""
    if ph_raw >= start:
        return start
    if ph_raw <= 0.0:
        return 0.0
    return float(np.floor(ph_raw / step) * step)


def event_scores_vs_offset(k_true: np.ndarray, k_pred: np.ndarray,
                           k_e: float, bin_edges_steps: list[int]):
    """Precision/recall/F1 of extreme-event occurrence per offset bin.

    For each bin [lo, hi) in steps from the prediction start: the truth
    label is 'an event occurs in the bin' (any k_true >= k_e), likewise
    for the prediction. Aggregation over starting points is the caller's
    job (arrays here are (n_starts, T)). Degenerate cases (no positive
    truth or no positive predictions) return 0 for the undefined ratio
    and are flagged -- anchor-tested."""
    k_true = np.atleast_2d(k_true)
    k_pred = np.atleast_2d(k_pred)
    out = []
    for lo, hi in zip(bin_edges_steps[:-1], bin_edges_steps[1:]):
        t_ev = np.any(k_true[:, lo:hi] >= k_e, axis=1)
        p_ev = np.any(k_pred[:, lo:hi] >= k_e, axis=1)
        tp = int(np.sum(t_ev & p_ev))
        fp = int(np.sum(~t_ev & p_ev))
        fn = int(np.sum(t_ev & ~p_ev))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out.append({"bin": (lo, hi), "precision": prec, "recall": rec,
                    "f1": f1, "tp": tp, "fp": fp, "fn": fn,
                    "degenerate": (tp + fn) == 0})
    return out


def valid_prediction_time(x_true: np.ndarray, x_pred: np.ndarray,
                          dt_lt: float, threshold: float = 0.4) -> float:
    """VPT in Lyapunov times: first time the normalised state error
        ||x_pred(t) - x_true(t)|| / sqrt(mean ||x_true||^2 over time)
    exceeds `threshold` (Lorenz-63 convention)."""
    x_true = np.atleast_2d(x_true)
    x_pred = np.atleast_2d(x_pred)
    scale = np.sqrt(np.mean(np.sum(x_true ** 2, axis=1)))
    err = np.linalg.norm(x_pred - x_true, axis=1) / max(scale, 1e-12)
    bad = np.nonzero(err > threshold)[0]
    n_ok = bad[0] if len(bad) else x_true.shape[0]
    return float(n_ok * dt_lt)
