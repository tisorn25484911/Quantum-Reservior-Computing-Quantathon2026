"""P10: shot emulation, feature-space diagnostics, and the three-arm adjudication."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.evaluation import diagnostics as D  # noqa: E402
from qrc_single_time_series.quantum import mitigation as MIT  # noqa: E402


# --------------------------------------------------------------- diagnostics
def test_effective_rank_hhi_rank_one_is_small():
    v = np.random.default_rng(0).standard_normal(200)
    X = np.outer(v, np.array([1.0, 2.0, -1.0, 0.5]))     # rank 1
    assert D.effective_rank_hhi(X) == pytest.approx(1.0, abs=0.05)


def test_effective_rank_hhi_full_rank_is_large():
    X = np.random.default_rng(1).standard_normal((500, 6))
    assert D.effective_rank_hhi(X) > 5.0                  # near 6 for iid columns


def test_feature_space_report_keys_and_cosine():
    rng = np.random.default_rng(2)
    Xe = 0.5 + 0.2 * rng.standard_normal((300, 8))
    Xn = Xe + 0.01 * rng.standard_normal((300, 8))
    rep = D.feature_space_report(Xe, Xn)
    assert rep["rmse"] > 0 and rep["cosine_rows"] > 0.99
    for k in ("eff_rank_exact", "eff_rank_noisy", "condition_exact", "covariance_frobenius"):
        assert k in rep


# ------------------------------------------------------------- shot emulation
def test_emulate_shots_more_shots_less_deviation():
    rng = np.random.default_rng(3)
    X = np.clip(0.5 + 0.3 * rng.standard_normal((400, 5)), 0, 1)
    lo = MIT.emulate_shots(X, 128, np.random.default_rng(0))
    hi = MIT.emulate_shots(X, 8192, np.random.default_rng(0))
    assert np.sqrt(((hi - X) ** 2).mean()) < np.sqrt(((lo - X) ** 2).mean())


def test_emulate_shots_preserves_bias_column():
    rng = np.random.default_rng(4)
    X = np.hstack([np.clip(0.5 + 0.2 * rng.standard_normal((200, 4)), 0, 1),
                   np.ones((200, 1))])                    # last col = bias
    Xn = MIT.emulate_shots(X, 256, np.random.default_rng(0))
    assert np.allclose(Xn[:, -1], 1.0)


# ------------------------------------------------------- three-arm adjudication
def test_svd_truncate_full_rank_matches_lstsq():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((200, 4))
    w = np.array([1.0, -2.0, 0.5, 3.0])
    y = X @ w + 0.3
    W, b = MIT.svd_truncate_fit(X, y, rank=4)
    assert np.allclose(W, w, atol=1e-6) and b == pytest.approx(0.3, abs=1e-6)


def test_three_arm_ridge_beats_noise_on_clean_linear_problem():
    # a well-conditioned linear map: ridge (arm B) should not be beaten by training
    # on shot noise (arm A) -> H5-style outcome on a clean problem.
    rng = np.random.default_rng(6)
    X = np.clip(0.5 + 0.2 * rng.standard_normal((400, 6)), 0, 1)
    w = rng.standard_normal(6)
    y = X @ w
    r = MIT.three_arm_adjudication(X[:300], y[:300], X[300:], y[300:],
                                   shots=512, ridge_lam=1e-4, trunc_rank=6)
    assert not r["noise_helps_beyond_matched_regulariser"]
    assert r["arm_B_ridge_nmse"] <= r["arm_A_noise_nmse"] + 1e-6
