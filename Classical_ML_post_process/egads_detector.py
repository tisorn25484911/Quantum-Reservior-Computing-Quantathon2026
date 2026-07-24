"""Multivariate anomalous-series detection driver (EGADS Section 3.3).

EGADS' third anomaly class (Section 3, item (c)) is the *anomalous time-series*:
given a *set* of series X = {x^(i)}, flag the members whose behaviour differs
from the majority. Section 3.3 does this by describing each series with the
Table-1 features, clustering the population, and scoring each series by its
deviation from the cluster centroids.

Here we apply that machine to a handful of long climate channels (e.g. the
Nino-region SST anomaly, the Southern Oscillation Index, the Pacific Decadal
Oscillation). To turn "one anomalous series among a population" into a temporal
detector, we use the standard *bag-of-windows* reduction: slide a fixed window
along the record and treat each window as one member x^(i) of the population X.

    * multivariate input:  the feature vector of a window concatenates the
      Table-1 features of *every channel* over that window, so the PC space
      jointly describes all series at once -- this is how multiple time series
      are combined to predict the anomaly.
    * a window whose joint feature profile is unusual relative to the other
      windows (large centroid deviation in PC space) is flagged; mapped back to
      the calendar it marks an anomalous stretch of time.

The module exposes:

    sliding_windows          index bookkeeping for the bag of windows
    build_feature_matrix     (T, C) series -> (n_windows, C * n_features) matrix
    AnomalousSeriesDetector  fit the PC-cluster model, flag windows, expand to
                             a per-timestamp score
    evaluate                 precision / recall / F1 vs a boolean label series
                             (the F1 metric EGADS reports in Section 6.3)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from egads_features import FEATURE_NAMES, feature_vector
from egads_pc_cluster import PCClusterAnomalyModel


# ------------------------------------------------------------------- windowing
def sliding_windows(n, win, step=1):
    """Yield (start, center, end) index triples for windows over [0, n).

    `end` is exclusive. The center is the integer midpoint, used when a window
    verdict is attributed to a single representative timestamp.
    """
    out = []
    start = 0
    while start + win <= n:
        end = start + win
        out.append((start, start + win // 2, end))
        start += step
    return out


def build_feature_matrix(X, win, step=1, period=12, channel_names=None):
    """Turn a (T, C) multivariate array into the EGADS window feature matrix.

    Returns
    -------
    F : ndarray (n_windows, C * len(FEATURE_NAMES))
        Row w = concatenation over channels of the Table-1 feature vector of
        window w.
    windows : list[(start, center, end)]
    columns : list[str]   "<channel>::<feature>" labels for F's columns.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T, C = X.shape
    if channel_names is None:
        channel_names = [f"ch{c}" for c in range(C)]
    windows = sliding_windows(T, win, step)
    columns = [f"{ch}::{feat}" for ch in channel_names for feat in FEATURE_NAMES]

    rows = []
    for (s, _, e) in windows:
        vec = []
        for c in range(C):
            vec.append(feature_vector(X[s:e, c], period=period))
        rows.append(np.concatenate(vec))
    F = np.asarray(rows, dtype=float)
    return F, windows, columns


# ------------------------------------------------------------------- detector
@dataclass
class DetectionResult:
    """Container for a detector run."""
    windows: list = field(default_factory=list)
    window_flags: np.ndarray = None       # (n_windows,) bool
    window_scores: np.ndarray = None      # (n_windows,) float deviation metric
    point_score: np.ndarray = None        # (T,) float, aggregated to timestamps
    point_flags: np.ndarray = None        # (T,) bool
    feature_matrix: np.ndarray = None     # (n_windows, n_feat)
    columns: list = field(default_factory=list)
    model: PCClusterAnomalyModel = None


@dataclass
class AnomalousSeriesDetector:
    """EGADS Section-3.3 anomalous-series detector over sliding windows.

    Parameters
    ----------
    win : int
        Window length in samples (>= 2*period + 1 to allow STL seasonality).
    step : int
        Hop between consecutive windows.
    period : int
        Seasonal period passed to the feature extractor (12 monthly, ~365 daily).
    rule : {"ksigma", "density"}
        Section 4.1 threshold rule used to binarise the deviation metric.
    expand : {"cover", "center"}
        How a window verdict maps onto timestamps. "cover" marks every
        timestamp inside a flagged window (natural for events that span time);
        "center" marks only the window midpoint.
    model_kwargs : dict
        Extra keyword args forwarded to ``PCClusterAnomalyModel``.
    """

    win: int = 36
    step: int = 1
    period: int = 12
    rule: str = "ksigma"
    expand: str = "cover"
    model_kwargs: dict = field(default_factory=dict)

    def fit_predict(self, X, channel_names=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        T = X.shape[0]
        F, windows, columns = build_feature_matrix(
            X, self.win, self.step, self.period, channel_names)

        model = PCClusterAnomalyModel(**self.model_kwargs)
        flags, scores = model.fit_flag(F, rule=self.rule)

        # aggregate window verdicts/scores back onto the timeline
        point_score = np.zeros(T)
        cover_count = np.zeros(T)
        point_flags = np.zeros(T, dtype=bool)
        # normalise scores to [0, 1] for a comparable per-point score
        s = scores.astype(float)
        s_norm = (s - s.min()) / (np.ptp(s) + 1e-12)
        for w, (start, center, end) in enumerate(windows):
            if self.expand == "center":
                point_score[center] = max(point_score[center], s_norm[w])
                if flags[w]:
                    point_flags[center] = True
            else:  # cover
                point_score[start:end] = np.maximum(point_score[start:end],
                                                     s_norm[w])
                cover_count[start:end] += 1
                if flags[w]:
                    point_flags[start:end] = True

        return DetectionResult(
            windows=windows, window_flags=flags, window_scores=scores,
            point_score=point_score, point_flags=point_flags,
            feature_matrix=F, columns=columns, model=model)


# ------------------------------------------------------------------ evaluation
def evaluate(pred_flags, true_labels, tolerance=0):
    """Point-wise precision / recall / F1 (the metric EGADS reports, Sec 6.3).

    Parameters
    ----------
    pred_flags, true_labels : bool arrays of equal length.
    tolerance : int
        If > 0, a predicted positive counts as a true positive when a real
        event lies within +/- `tolerance` samples (and vice-versa for recall).
        This is the usual event-detection slack; 0 gives strict point overlap.
    """
    pred = np.asarray(pred_flags, dtype=bool)
    true = np.asarray(true_labels, dtype=bool)
    n = min(pred.size, true.size)
    pred, true = pred[:n], true[:n]

    if tolerance > 0:
        true_dil = _dilate(true, tolerance)
        pred_dil = _dilate(pred, tolerance)
        tp_p = int(np.sum(pred & true_dil))          # preds near a true event
        tp_r = int(np.sum(true & pred_dil))          # events near a pred
        precision = tp_p / max(1, int(pred.sum()))
        recall = tp_r / max(1, int(true.sum()))
    else:
        tp = int(np.sum(pred & true))
        precision = tp / max(1, int(pred.sum()))
        recall = tp / max(1, int(true.sum()))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": int(pred.sum()),
        "n_true": int(true.sum()),
        "n": int(n),
    }


def _dilate(mask, k):
    """Boolean dilation by +/- k samples."""
    m = np.asarray(mask, dtype=bool)
    out = m.copy()
    for s in range(1, k + 1):
        out[s:] |= m[:-s]
        out[:-s] |= m[s:]
    return out
