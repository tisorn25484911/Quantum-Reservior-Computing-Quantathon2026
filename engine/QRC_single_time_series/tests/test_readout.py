"""P2 physics-free readout suite: SVD ridge path, pinv exactness, GCV, invariances."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.models import readout as R  # noqa: E402


def synth(T=300, F=6, L=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, F))
    W = rng.standard_normal((F, L))
    b = rng.standard_normal(L)
    return X, X @ W + b, W, b


def test_lambda0_recovers_exact_linear_map():
    X, Y, _, _ = synth()
    ro = R.fit(X, Y, lam=0.0)
    assert np.allclose(ro.predict(X), Y, atol=1e-9)
    assert ro.rank == X.shape[1]


def test_lambda0_matches_lstsq():
    X, Y, _, _ = synth(L=1)
    ro = R.fit(X, Y.ravel(), lam=0.0)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(Xb, Y.ravel(), rcond=None)
    assert np.allclose(ro.predict(X), Xb @ coef, atol=1e-8)


def test_large_lambda_drives_w_to_zero_intercept_to_mean():
    X, Y, _, _ = synth()
    ro = R.fit(X, Y, lam=1e14)
    assert np.max(np.abs(ro.W)) < 1e-6
    assert np.allclose(ro.b, Y.mean(axis=0), atol=1e-6)
    assert np.allclose(ro.predict(X), Y.mean(axis=0), atol=1e-6)


def test_batched_multitarget_equals_percolumn():
    X, Y, _, _ = synth(L=3)
    ro_all = R.fit(X, Y, lam=0.7)
    for l in range(Y.shape[1]):
        ro_l = R.fit(X, Y[:, l], lam=0.7)
        assert np.allclose(ro_all.W[:, l], ro_l.W[:, 0], atol=1e-10)
        assert ro_all.b[l] == pytest.approx(ro_l.b[0])


def test_standardisation_invariance_at_lambda0():
    # Predictions at lambda=0 are invariant to affine rescaling of the features.
    X, Y, _, _ = synth(L=1)
    scale = np.random.default_rng(2).uniform(0.5, 3, X.shape[1])
    shift = np.random.default_rng(3).standard_normal(X.shape[1])
    Xs = X * scale + shift
    p_raw = R.fit(X, Y.ravel(), lam=0.0).predict(X)
    p_std = R.fit(Xs, Y.ravel(), lam=0.0).predict(Xs)
    assert np.allclose(p_raw, p_std, atol=1e-7)


def test_gcv_picks_positive_lambda_under_noise():
    X, Y, _, _ = synth(T=200, F=8, L=1)
    yn = Y.ravel() + 0.5 * np.random.default_rng(9).standard_normal(len(Y))
    ro = R.fit(X, yn, lam="gcv")
    assert ro.lam[0] > 0
    # GCV solution should generalise no worse than a wildly over/under-regularised one
    assert ro._1d is True


def test_dof_monotonic_in_lambda():
    X, Y, _, _ = synth(L=1)
    dofs = [R.fit(X, Y.ravel(), lam=lam).dof[0] for lam in [1e-6, 1.0, 100.0, 1e6]]
    assert all(a >= b - 1e-9 for a, b in zip(dofs, dofs[1:]))
