"""Baseline battery anchors."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from baselines import (ESN, HORIZON, L, chrono_split, haar_reservoir,
                       lag_matrix, make_targets, run_battery)


def ar1(T=600, phi=0.9, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + rng.normal(scale=sigma)
    return y


def make_scaler(y):
    lo, hi = y[:480].min(), y[:480].max()
    return lambda s: np.clip((s - lo) / (hi - lo), 0, 1)


def test_alignment_convention():
    y = np.arange(100, dtype=float)
    ks, yt = make_targets(y)
    assert ks[0] == L - 1
    assert yt[0] == y[L - 1 + HORIZON]          # target H ahead of k
    assert ks[-1] + HORIZON == len(y) - 1        # last target inside series


def test_mean_baseline_nmse_close_to_one():
    y = ar1()
    results, _ = run_battery(y, make_scaler(y))
    # exactly 1.0 iff train mean == test-span reference; close for AR(1)
    assert 0.9 < results["mean"]["nmse"] < 1.2


def test_linear_lags_nails_deterministic_sine():
    """A sinusoid obeys a 2-term linear recurrence; 24 lags must nail it."""
    t = np.arange(400)
    y = np.sin(2 * np.pi * t / 12.0)
    results, _ = run_battery(y, make_scaler(y))
    assert results["linear_lags"]["nmse"] < 1e-6


def test_battery_ordering_on_ar1():
    """On AR(1), linear lags ~ persistence << mean; nothing is > 1.5."""
    y = ar1()
    results, _ = run_battery(y, make_scaler(y))
    assert results["persistence"]["nmse"] < results["mean"]["nmse"]
    assert results["linear_lags"]["nmse"] <= results["persistence"]["nmse"] * 1.1
    for name, s in results.items():
        assert s["nmse"] < 1.5, (name, s)


def test_esn_echo_state_property():
    """Two different initial states converge after a washout."""
    y = ar1(T=400)
    e = ESN(seed=7)
    xa = e.states(y, x0=np.zeros(20))
    xb = e.states(y, x0=np.ones(20))
    assert np.max(np.abs(xa[300:] - xb[300:])) < 1e-8


def test_haar_control_differs_from_structured():
    from qrc_core import WindowedReservoir
    w = np.linspace(0, 1, L)
    f_haar = haar_reservoir(seed=7).features(w)
    f_struct = WindowedReservoir(seed=7).features(w)
    assert len(f_haar) == 20
    assert not np.allclose(f_haar, f_struct)
    assert np.array_equal(f_haar, haar_reservoir(seed=7).features(w))


def test_chrono_split_is_chronological():
    tr, te = chrono_split(100)
    assert tr.max() < te.min()


def test_lag_matrix_rows_end_at_k():
    y = np.arange(60, dtype=float)
    ks, _ = make_targets(y)
    M = lag_matrix(y, ks)
    assert M.shape[1] == L
    assert M[0, -1] == y[ks[0]]
    assert M[0, 0] == y[ks[0] - L + 1]
