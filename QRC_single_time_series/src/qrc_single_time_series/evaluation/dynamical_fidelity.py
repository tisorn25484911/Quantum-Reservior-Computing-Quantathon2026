"""dynamical_fidelity.py -- distributional / spectral / ACF / recurrence fidelity.

Autonomous forecasting must be judged on DYNAMICAL fidelity, not only pointwise
error (spec s18.1, acceptance item 16): a trajectory can diverge pointwise from the
truth while still living on the right attractor. This module scores whether a
generated series reproduces the true series' invariant measure, power spectrum,
autocorrelation, and recurrence structure -- the statistics that survive after the
pointwise valid time is exhausted. Implemented in P8.
"""
from __future__ import annotations

import numpy as np


def autocorrelation(x, max_lag=60):
    """Biased sample ACF rho(k), k=0..max_lag (rho(0)=1)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = np.dot(x, x)
    if var == 0:
        return np.concatenate([[1.0], np.zeros(max_lag)])
    acf = np.array([np.dot(x[:len(x) - k], x[k:]) / var for k in range(max_lag + 1)])
    return acf


def power_spectrum(x, detrend=True):
    """One-sided periodogram of ``x`` (normalised to unit total power)."""
    x = np.asarray(x, dtype=float)
    if detrend:
        x = x - x.mean()
    f = np.fft.rfft(x)
    p = np.abs(f) ** 2
    tot = p.sum()
    return p / tot if tot > 0 else p


def invariant_measure_distance(true, pred, bins=40, range_=(0.0, 1.0)):
    """L1 distance between the histograms (empirical invariant measures)."""
    ht, edges = np.histogram(np.asarray(true, float), bins=bins, range=range_,
                             density=True)
    hp, _ = np.histogram(np.asarray(pred, float), bins=bins, range=range_,
                         density=True)
    width = edges[1] - edges[0]
    return float(0.5 * np.sum(np.abs(ht - hp)) * width)


def recurrence_rate(x, m=3, lag=1, threshold=0.1):
    """Recurrence rate: fraction of embedded-state pairs within ``threshold``.

    A cheap recurrence-quantification summary (Marwan et al.): the density of the
    recurrence matrix on a delay embedding, on the state range normalised to [0,1].
    """
    from .lyapunov import delay_embed
    emb = delay_embed(np.asarray(x, float), m, lag)
    # subsample for O(M^2) tractability
    M = len(emb)
    step = max(1, M // 400)
    emb = emb[::step]
    d = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
    scale = d.max() if d.max() > 0 else 1.0
    return float(np.mean((d / scale) < threshold))


def spectral_distance(true, pred):
    """L1 distance between normalised power spectra (truncated to the shorter)."""
    pt = power_spectrum(true)
    pp = power_spectrum(pred)
    n = min(len(pt), len(pp))
    return float(0.5 * np.sum(np.abs(pt[:n] - pp[:n])))


def acf_distance(true, pred, max_lag=60):
    """RMS difference of the two ACFs over lags 1..max_lag."""
    at = autocorrelation(true, max_lag)
    ap = autocorrelation(pred, max_lag)
    return float(np.sqrt(np.mean((at[1:] - ap[1:]) ** 2)))


def fidelity_report(true, pred, hist_range=(0.0, 1.0), max_lag=60):
    """All four fidelity distances in one dict (0 == perfect match)."""
    return {
        "invariant_measure_L1": invariant_measure_distance(true, pred,
                                                            range_=hist_range),
        "spectral_L1": spectral_distance(true, pred),
        "acf_rms": acf_distance(true, pred, max_lag),
        "recurrence_rate_true": recurrence_rate(true),
        "recurrence_rate_pred": recurrence_rate(pred),
        "std_ratio": float(np.std(pred) / (np.std(true) + 1e-12)),
    }
