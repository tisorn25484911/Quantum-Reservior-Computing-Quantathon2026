"""Dataset anchors: loader integrity + the leak rule can never drift."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from baselines import HORIZON, L, chrono_split, make_targets
from datasets import (anomaly, fit_scaler, load_enso, monthly_climatology,
                      train_month_cutoff)


def test_loader_shape_gaps_cache(tmp_path):
    cache = tmp_path / "enso.csv"
    df = load_enso(cache)                       # fetch + pin
    assert cache.exists()
    df2 = load_enso(cache)                      # reload from pinned copy
    assert len(df) == 732 and len(df2) == 732
    assert df.sst.notna().all()
    assert np.allclose(df.sst.to_numpy(), df2.sst.to_numpy())
    mi = df.date.dt.year * 12 + df.date.dt.month
    assert (mi.diff().dropna() == 1).all()


def test_cutoff_aligns_with_harness():
    """Train span must end exactly at the last training target month."""
    T = 732
    ks, _ = make_targets(np.zeros(T))
    tr, te = chrono_split(len(ks))
    n_train = train_month_cutoff(T)
    assert ks[tr][-1] + HORIZON == n_train - 1   # last train target inside
    assert ks[te][0] + HORIZON == n_train        # first test target outside
    assert n_train == L + HORIZON + len(tr) - 1


def test_climatology_and_scaler_are_leak_free():
    """Perturbing the test span must not move any train-fit statistic."""
    rng = np.random.default_rng(3)
    T = 240
    month = np.tile(np.arange(1, 13), T // 12)
    sst = 24 + 2 * np.sin(2 * np.pi * np.arange(T) / 12) + rng.normal(0, 0.5, T)
    n_train = train_month_cutoff(T)

    y1, clim1 = anomaly(month, sst, n_train)
    s1, r1 = fit_scaler(y1, n_train)

    sst_p = sst.copy()
    sst_p[n_train:] += 10.0                      # corrupt test span only
    y2, clim2 = anomaly(month, sst_p, n_train)
    s2, r2 = fit_scaler(y2, n_train)

    assert np.allclose(clim1, clim2)
    assert r1 == r2
    assert np.allclose(y1[:n_train], y2[:n_train])
    assert np.allclose(s1(y1[:n_train]), s2(y2[:n_train]))


def test_anomaly_roundtrip_and_train_mean_zeroish():
    rng = np.random.default_rng(4)
    T = 240
    month = np.tile(np.arange(1, 13), T // 12)
    sst = 24 + 2 * np.sin(2 * np.pi * np.arange(T) / 12) + rng.normal(0, 0.5, T)
    n_train = train_month_cutoff(T)
    y, clim = anomaly(month, sst, n_train)
    assert np.allclose(y + clim[month - 1], sst)
    # exact zero only if n_train is a whole number of years; else near-zero
    assert abs(y[:n_train].mean()) < 0.05


def test_scaler_maps_train_extremes_and_clips():
    y = np.array([-2.0, 0.0, 1.0, 3.0, -5.0, 9.0])
    scaler, (lo, hi) = fit_scaler(y, n_train=4)   # train = first 4
    assert (lo, hi) == (-2.0, 3.0)
    assert scaler(np.array([lo]))[0] == 0.0
    assert scaler(np.array([hi]))[0] == 1.0
    assert scaler(np.array([-5.0]))[0] == 0.0     # clipped
    assert scaler(np.array([9.0]))[0] == 1.0      # clipped
