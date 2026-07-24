"""eda.py -- exploratory diagnostics for the observational indices (spec s14).

Train-only diagnostic statistics used to READ the climate results correctly (not to
select models): stationarity (ADF, KPSS), autocorrelation structure (ACF/PACF, the
decorrelation time that sets the bootstrap block length), the dominant spectral
period, and a seasonal-strength summary. Everything here is computed on the
development span only so it never touches the untouched test window. Implemented in
P9.
"""
from __future__ import annotations

import numpy as np

try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    _SM = True
except Exception:                                         # pragma: no cover
    _SM = False


def decorrelation_time(x, max_lag=120):
    """First lag where the ACF drops below 1/e (fallback: first zero crossing)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = np.dot(x, x)
    if var == 0:
        return 1
    r = np.array([np.dot(x[:len(x) - k], x[k:]) / var for k in range(max_lag + 1)])
    below = np.where(r < np.exp(-1.0))[0]
    if below.size:
        return int(below[0])
    zero = np.where(r < 0)[0]
    return int(zero[0]) if zero.size else max_lag


def dominant_period(x):
    """Period (in samples) of the largest non-DC periodogram peak."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x))
    p[0] = 0.0
    k = int(np.argmax(p))
    return float(1.0 / freqs[k]) if freqs[k] > 0 else float("inf")


def seasonal_strength(x, period=12):
    """Fraction of variance explained by the mean seasonal cycle (0..1)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    cyc = np.array([x[j::period].mean() for j in range(period)])
    seasonal = np.array([cyc[t % period] for t in range(n)])
    return float(np.var(seasonal) / (np.var(x) + 1e-12))


def eda_report(x, n_train, period=12):
    """Stationarity + ACF-structure + spectral summary on the TRAIN span.

    Returns a JSON-safe dict. ADF small p ⇒ stationary; KPSS small p ⇒ NON-stationary
    (the two are complementary null hypotheses, reported together per the checklist).
    """
    xt = np.asarray(x, dtype=float)[:n_train]
    out = {
        "n_train": int(n_train),
        "mean": float(xt.mean()), "std": float(xt.std()),
        "decorrelation_time": decorrelation_time(xt),
        "dominant_period": dominant_period(xt),
        "seasonal_strength": seasonal_strength(xt, period),
    }
    if _SM:
        try:
            adf = adfuller(xt, autolag="AIC")
            out["adf_stat"], out["adf_p"] = float(adf[0]), float(adf[1])
        except Exception:                                 # pragma: no cover
            out["adf_p"] = None
        try:
            k = kpss(xt, regression="c", nlags="auto")
            out["kpss_stat"], out["kpss_p"] = float(k[0]), float(k[1])
        except Exception:                                 # pragma: no cover
            out["kpss_p"] = None
        out["acf_10"] = [float(v) for v in acf(xt, nlags=10, fft=True)]
        out["pacf_10"] = [float(v) for v in pacf(xt, nlags=10)]
    return out
