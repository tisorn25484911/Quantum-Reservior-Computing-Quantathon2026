"""EGADS Table-1 time-series feature extraction.

This module implements the per-series feature catalogue that EGADS uses to
*characterise* a time-series before deciding whether it is anomalous relative
to its peers.

    Laptev, Amizadeh & Flint, "Generic and Scalable Framework for Automated
    Time-series Anomaly Detection", KDD 2015 (DOI 10.1145/2783258.2788611).
    See Table 1 ("Time-series features used by EGADS") and Section 3.3
    ("Detecting Anomalous Time-series"). EGADS reference [29] is the feature
    catalogue this builds on: Wang, Smith-Miles & Hyndman, "Rule induction for
    forecasting method selection", Neurocomputing 72 (2009).

Every function here takes a 1-D numpy array (one channel of one window) and
returns a scalar. `extract_features` bundles them into an ordered dict so a set
of series can be stacked into the feature matrix consumed by
``egads_pc_cluster``.

The features implemented (Table 1 rows, plus spectral entropy which Section 3.3
names explicitly):

    periodicity     dominant seasonal period estimated from the spectrum
    trend           STL trend strength           F_T = 1 - Var(R)/Var(T+R)
    seasonality     STL seasonal strength        F_S = 1 - Var(R)/Var(S+R)
    spectral_entropy Shannon entropy of the normalised periodogram
    acf1            lag-1 autocorrelation (long-range dependence proxy)
    acf10           sum of squares of the first 10 autocorrelations
    nonlinearity    Terasvirta-style neural-network nonlinearity statistic
    skewness        lack of symmetry
    kurtosis        peakedness relative to a normal (Fisher / excess)
    hurst           Hurst exponent (rescaled-range, long-term memory)
    lyapunov        largest Lyapunov exponent (divergence of trajectories)

All estimators are guarded so that short or degenerate windows return a finite
value (0.0 by default) rather than raising, because the sliding-window harness
in ``egads_detector`` evaluates them on every window of the record.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import STL

# The ordered list of feature names produced by ``extract_features``. Kept as a
# module constant so downstream code (PCA column labels, plots) can rely on it.
FEATURE_NAMES = (
    "periodicity",
    "trend",
    "seasonality",
    "spectral_entropy",
    "acf1",
    "acf10",
    "nonlinearity",
    "skewness",
    "kurtosis",
    "hurst",
    "lyapunov",
)


# --------------------------------------------------------------------- helpers
def _clean(x):
    """Return a finite float64 1-D copy, dropping NaN/Inf."""
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def _acf(x, nlags):
    """Biased sample autocorrelations r_1..r_nlags (r_0 == 1 dropped)."""
    x = x - x.mean()
    n = x.size
    denom = np.dot(x, x)
    if denom <= 0:
        return np.zeros(nlags)
    out = np.empty(nlags)
    for k in range(1, nlags + 1):
        out[k - 1] = np.dot(x[:-k], x[k:]) / denom
    return out


# ------------------------------------------------------------------- spectrum
def spectral_features(x, sampling_period=1):
    """Dominant period and normalised spectral (Shannon) entropy.

    The one-sided periodogram is normalised to a probability distribution over
    frequency; its Shannon entropy (divided by log of the number of bins) is a
    flatness measure in [0, 1] -- low entropy means the energy is concentrated
    at a few frequencies (strongly periodic), high entropy means white-noise
    like. The dominant period is the reciprocal of the frequency carrying the
    most energy (ignoring the zero-frequency / DC term).
    """
    x = _clean(x)
    n = x.size
    if n < 4 or np.allclose(x, x[0]):
        return 0.0, 0.0
    x = x - x.mean()
    freqs = np.fft.rfftfreq(n, d=sampling_period)
    power = np.abs(np.fft.rfft(x)) ** 2
    # drop the DC bin: it carries no periodicity information
    freqs, power = freqs[1:], power[1:]
    total = power.sum()
    if total <= 0:
        return 0.0, 0.0
    p = power / total
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size))
    dom_freq = freqs[np.argmax(power)]
    period = float(1.0 / dom_freq) if dom_freq > 0 else 0.0
    return period, entropy


# --------------------------------------------------------- trend & seasonality
def stl_strengths(x, period):
    """STL trend and seasonal strength, following Wang/Hyndman (EGADS [29]).

    Decompose x = trend + seasonal + remainder via STL, then

        F_trend    = max(0, 1 - Var(remainder) / Var(trend    + remainder))
        F_seasonal = max(0, 1 - Var(remainder) / Var(seasonal + remainder))

    Each lies in [0, 1]; 1 means the component dominates what is left after
    removing the others, 0 means it explains no more than the noise. STL needs
    at least two full periods, so shorter windows fall back to a trend-only
    proxy and zero seasonality.
    """
    x = _clean(x)
    n = x.size
    period = int(period)
    if n < 2 * period + 1 or period < 2:
        # not enough data for a seasonal decomposition
        if n >= 5:
            t = np.arange(n)
            slope = np.polyfit(t, x, 1)
            detr = x - np.polyval(slope, t)
            vt = np.var(x)
            f_trend = max(0.0, 1.0 - np.var(detr) / vt) if vt > 0 else 0.0
            return float(f_trend), 0.0
        return 0.0, 0.0
    try:
        res = STL(x, period=period, robust=True).fit()
    except Exception:
        return 0.0, 0.0
    rem = res.resid
    var_rem = np.var(rem)
    var_tr = np.var(res.trend + rem)
    var_se = np.var(res.seasonal + rem)
    f_trend = max(0.0, 1.0 - var_rem / var_tr) if var_tr > 1e-12 else 0.0
    f_seas = max(0.0, 1.0 - var_rem / var_se) if var_se > 1e-12 else 0.0
    return float(f_trend), float(f_seas)


# ------------------------------------------------------------- autocorrelation
def acf_features(x, nlags=10):
    """Lag-1 autocorrelation and the sum of squares of the first `nlags`."""
    x = _clean(x)
    if x.size < nlags + 2 or np.allclose(x, x[0]):
        return 0.0, 0.0
    r = _acf(x, nlags)
    return float(r[0]), float(np.sum(r ** 2))


# --------------------------------------------------------------- nonlinearity
def nonlinearity(x):
    """Terasvirta-style neural-network test statistic for nonlinearity.

    We fit a linear AR(1) model, then regress its residuals on the lagged level
    together with its square and cube. Under linearity the added quadratic and
    cubic terms explain nothing, so the Lagrange-multiplier statistic
    ``T * R^2`` of that auxiliary regression is small; genuine nonlinear
    structure makes it large. This is the reduced (lag-1) form of the
    Terasvirta test used by Wang/Hyndman (EGADS [29]).
    """
    x = _clean(x)
    n = x.size
    if n < 12 or np.allclose(x, x[0]):
        return 0.0
    x = (x - x.mean()) / (x.std() + 1e-12)
    y, lag = x[1:], x[:-1]
    # linear AR(1) fit
    a = np.vstack([np.ones_like(lag), lag]).T
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ beta
    # auxiliary regression of residuals on lag, lag^2, lag^3
    z = np.vstack([np.ones_like(lag), lag, lag ** 2, lag ** 3]).T
    coef, *_ = np.linalg.lstsq(z, resid, rcond=None)
    fitted = z @ coef
    ss_tot = np.sum((resid - resid.mean()) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    r2 = 1.0 - np.sum((resid - fitted) ** 2) / ss_tot
    return float(max(0.0, (n - 1) * r2))


# ------------------------------------------------------------ shape statistics
def shape_features(x):
    """Fisher skewness and excess kurtosis."""
    x = _clean(x)
    if x.size < 3 or np.allclose(x, x[0]):
        return 0.0, 0.0
    return float(stats.skew(x)), float(stats.kurtosis(x))  # excess (Fisher)


# --------------------------------------------------------------------- Hurst
def hurst_exponent(x, min_chunk=8):
    """Hurst exponent by rescaled-range (R/S) analysis.

    For a set of chunk sizes n, split the series into non-overlapping chunks,
    compute the rescaled range R/S per chunk, average, and take the slope of
    log(R/S) versus log(n). H ~ 0.5 is a random walk, H > 0.5 persistent
    (long memory), H < 0.5 anti-persistent -- the "long-term memory" row of
    Table 1.
    """
    x = _clean(x)
    n = x.size
    if n < 2 * min_chunk:
        return 0.5
    scales = []
    rs = []
    size = min_chunk
    while size <= n // 2:
        n_chunks = n // size
        vals = []
        for c in range(n_chunks):
            chunk = x[c * size:(c + 1) * size]
            z = chunk - chunk.mean()
            cumdev = np.cumsum(z)
            r = cumdev.max() - cumdev.min()
            s = chunk.std()
            if s > 1e-12:
                vals.append(r / s)
        if vals:
            scales.append(size)
            rs.append(np.mean(vals))
        size *= 2
    if len(scales) < 2:
        return 0.5
    slope = np.polyfit(np.log(scales), np.log(rs), 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


# ------------------------------------------------------------------ Lyapunov
def lyapunov_exponent(x, emb_dim=2, lag=1):
    """Largest Lyapunov exponent, a light Rosenstein-style estimate.

    Delay-embed the (standardised) series, and for each embedded point find its
    nearest neighbour. The average log-divergence of initially-close pairs after
    one step estimates the rate at which nearby trajectories separate -- the
    "rate of divergence of nearby trajectories" row of Table 1. Positive values
    indicate chaotic / unpredictable dynamics. Kept deliberately short-window
    safe; degenerate windows return 0.
    """
    x = _clean(x)
    n = x.size
    m = n - (emb_dim - 1) * lag
    if m < 6 or np.allclose(x, x[0]):
        return 0.0
    x = (x - x.mean()) / (x.std() + 1e-12)
    emb = np.column_stack([x[i * lag:i * lag + m] for i in range(emb_dim)])
    # pairwise distances between embedded points
    diff = emb[:, None, :] - emb[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    # exclude self and temporally-adjacent points (Theiler window = 1)
    big = dist.max() + 1.0
    idx = np.arange(m)
    mask = np.abs(idx[:, None] - idx[None, :]) <= 1
    dist[mask] = big
    nn = np.argmin(dist, axis=1)
    divs = []
    for i in range(m - 1):
        j = nn[i]
        if j >= m - 1:
            continue
        d0 = np.linalg.norm(emb[i] - emb[j])
        d1 = np.linalg.norm(emb[i + 1] - emb[j + 1])
        if d0 > 1e-12 and d1 > 1e-12:
            divs.append(np.log(d1 / d0))
    if not divs:
        return 0.0
    return float(np.mean(divs))


# ---------------------------------------------------------------- entry point
def extract_features(x, period=12):
    """Return the ordered EGADS Table-1 feature dict for one series/window.

    Parameters
    ----------
    x : array-like
        One channel of one window (raw values or anomalies).
    period : int
        Known seasonal period in samples (12 for monthly data, ~365 for daily).

    Returns
    -------
    collections.OrderedDict
        ``FEATURE_NAMES`` -> float, in a fixed, stable order.
    """
    x = _clean(x)
    dom_period, spec_entropy = spectral_features(x)
    f_trend, f_seas = stl_strengths(x, period)
    r1, r10 = acf_features(x)
    skew, kurt = shape_features(x)
    feats = OrderedDict()
    feats["periodicity"] = dom_period
    feats["trend"] = f_trend
    feats["seasonality"] = f_seas
    feats["spectral_entropy"] = spec_entropy
    feats["acf1"] = r1
    feats["acf10"] = r10
    feats["nonlinearity"] = nonlinearity(x)
    feats["skewness"] = skew
    feats["kurtosis"] = kurt
    feats["hurst"] = hurst_exponent(x)
    feats["lyapunov"] = lyapunov_exponent(x)
    return feats


def feature_vector(x, period=12):
    """`extract_features` as a plain numpy array ordered by ``FEATURE_NAMES``."""
    f = extract_features(x, period=period)
    return np.array([f[k] for k in FEATURE_NAMES], dtype=float)
