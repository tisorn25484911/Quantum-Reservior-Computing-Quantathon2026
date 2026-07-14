"""Validation anchors. Must pass before and after any reservoir change."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from qrc_core import (WindowedReservoir, add_bias, nmse, ridge_fit,
                      ridge_gcv)


def test_ridge_recovers_known_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    w_true = rng.normal(size=10)
    y = X @ w_true
    w = ridge_fit(X, y, lam=1e-12)
    assert np.max(np.abs(w - w_true)) < 1e-10


def test_gcv_recovers_known_linear_map():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 10))
    w_true = rng.normal(size=10)
    y = X @ w_true
    w, lam = ridge_gcv(X, y, lams=np.logspace(-14, 2, 17))
    assert np.max(np.abs(w - w_true)) < 1e-6


def test_nmse_of_mean_predictor_is_exactly_one():
    rng = np.random.default_rng(2)
    y_train = rng.normal(size=300)
    y_test = rng.normal(size=100)
    m = y_train.mean()
    assert nmse(y_test, np.full_like(y_test, m), m) == 1.0


def test_entangler_is_unitary_and_state_normalised():
    r = WindowedReservoir()
    err = np.max(np.abs(r.W.conj().T @ r.W - np.eye(2 ** r.n)))
    assert err < 1e-12
    psi = r.state(np.linspace(0, 1, 24))
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_features_deterministic_at_fixed_seed():
    w = np.linspace(0, 1, 24)
    f1 = WindowedReservoir(seed=7).features(w)
    f2 = WindowedReservoir(seed=7).features(w)
    assert np.array_equal(f1, f2)
    f3 = WindowedReservoir(seed=8).features(w)
    assert not np.allclose(f1, f3)


def test_feature_count_and_range():
    r = WindowedReservoir()
    f = r.features(np.linspace(0, 1, 24))
    assert len(f) == 20 == r.n_features()
    assert np.all(np.abs(f) <= 1.0 + 1e-12)  # Pauli expectations


def test_shot_noise_scales_as_inverse_sqrt_shots():
    r = WindowedReservoir()
    w = np.linspace(0, 1, 24)
    exact = r.features(w)[: r.n]  # <Z_i> block
    rng = np.random.default_rng(3)

    def spread(shots, reps=60):
        ests = np.array([r.sampled_z_features(w, shots, rng)
                         for _ in range(reps)])
        return np.mean(np.std(ests, axis=0))

    s1, s2 = spread(250), spread(4000)
    ratio = s1 / s2                      # expect sqrt(4000/250) = 4
    assert 3.0 < ratio < 5.0
    big = np.mean([r.sampled_z_features(w, 4000, rng) for _ in range(60)], axis=0)
    assert np.max(np.abs(big - exact)) < 0.05
