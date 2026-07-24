"""P2 physics-free metrics suite: NMSE conventions, capacity, effective rank, MASE."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.evaluation import metrics as M  # noqa: E402
from qrc_single_time_series.models import readout as R  # noqa: E402


def test_mean_predictor_nmse_is_one():
    y = np.random.default_rng(0).standard_normal(500)
    assert M.nmse_mean_predictor(y) == pytest.approx(1.0)
    assert M.nmse_variance(y, np.full_like(y, y.mean())) == pytest.approx(1.0)


def test_perfect_prediction_scores():
    y = np.random.default_rng(1).standard_normal(200)
    assert M.mse(y, y) == pytest.approx(0.0)
    assert M.r2(y, y) == pytest.approx(1.0)
    assert M.pearson(y, y) == pytest.approx(1.0)
    assert M.capacity(y, y) == pytest.approx(1.0)


def test_both_nmse_conventions_are_distinct_and_correct():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(1000) + 5.0        # nonzero mean -> conventions differ
    yhat = y + 0.3 * rng.standard_normal(1000)
    sse = np.sum((y - yhat) ** 2)
    assert M.nmse_variance(y, yhat) == pytest.approx(sse / (len(y) * np.var(y)))
    assert M.nmse_fn(y, yhat) == pytest.approx(sse / np.sum(y ** 2))
    assert M.nmse_variance(y, yhat) != pytest.approx(M.nmse_fn(y, yhat))


def test_effective_rank_bounds():
    rng = np.random.default_rng(3)
    assert M.effective_rank(rng.standard_normal((500, 5))) == pytest.approx(5.0, abs=0.2)
    col = rng.standard_normal((500, 1))
    rank1 = np.hstack([col, 2 * col, -col])    # rank-1 matrix
    assert M.effective_rank(rank1) == pytest.approx(1.0, abs=1e-6)


def test_capacity_of_constant_is_zero():
    y = np.random.default_rng(4).standard_normal(300)
    assert M.capacity(y, np.zeros_like(y)) == pytest.approx(0.0)


def test_mase_scales_by_naive():
    y_train = np.cumsum(np.random.default_rng(5).standard_normal(300))
    y = np.cumsum(np.random.default_rng(6).standard_normal(50))
    scale = np.mean(np.abs(np.diff(y_train)))
    assert M.mase(y, y + 1.0, y_train) == pytest.approx(1.0 / scale)


def test_persistence_skill_sign():
    assert M.persistence_skill(0.5, 1.0) > 0     # model beats persistence
    assert M.persistence_skill(2.0, 1.0) < 0     # model worse than persistence


def test_capacity_floor_of_random_features():
    # White-noise features vs a random target: total captured "capacity" over
    # independent delays approximates the finite-length floor ~ F/L_test.
    rng = np.random.default_rng(11)
    T, F = 4000, 20
    X = rng.standard_normal((T, F))
    y = rng.standard_normal(T)               # unrelated target
    ntr = 3000
    ro = R.fit(X[:ntr], y[:ntr], lam=1e-10)
    pred = ro.predict(X[ntr:])
    cap = M.capacity(y[ntr:], pred)
    floor = F / (T - ntr)
    assert cap < 5 * floor                   # spurious capacity stays near the floor
