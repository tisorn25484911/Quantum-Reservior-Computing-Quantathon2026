"""P5 (fast, small-scale): MG gate mechanics -- generation, lambda estimators,
stateful adapter one-step, noise~ridge equivalence. The full GATE is the script
scripts/reproduce_mackey_glass.py; this guards its building blocks in CI."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.dynamical_systems import mackey_glass as MG  # noqa: E402
from qrc_single_time_series.quantum.hamiltonians import fc_tfi  # noqa: E402
from qrc_single_time_series.quantum.exact_qrc import ExactQRC  # noqa: E402
from qrc_single_time_series.training.teacher_forcing import train_teacher_forced  # noqa: E402
from qrc_single_time_series.training.noise_augmentation import (  # noqa: E402
    lambda_eff, sigma_for_lambda, add_feature_noise)
from qrc_single_time_series.models.readout import fit as fit_readout  # noqa: E402
from qrc_single_time_series.models.recursive_forecaster import StatefulFN  # noqa: E402
from qrc_single_time_series.evaluation.autonomous import autonomous_from_series  # noqa: E402
from qrc_single_time_series.data.windows import align  # noqa: E402
from qrc_single_time_series.evaluation.lyapunov import (  # noqa: E402
    rosenstein, logistic_series, delay_embed)


def test_mg_generation_is_deterministic_and_scaled():
    a = MG.generate(17, n=500, washout=800)["series"]
    b = MG.generate(17, n=500, washout=800)["series"]
    assert np.array_equal(a, b)
    assert a.min() == pytest.approx(0.0) and a.max() == pytest.approx(1.0)
    assert len(a) == 500


def test_rosenstein_recovers_logistic_exponent():
    lam, _ = rosenstein(logistic_series(n=3000), m=3, lag=1, mean_period=1, max_t=15)
    assert lam == pytest.approx(0.494, abs=0.06)


def test_delay_embed_shape():
    emb = delay_embed(np.arange(100.0), m=3, lag=5)
    assert emb.shape == (100 - 2 * 5, 3)


def test_tau17_more_chaotic_than_tau16():
    s16 = MG.generate(16, n=1500, washout=1200)["series"]
    s17 = MG.generate(17, n=1500, washout=1200)["series"]
    l16, _ = rosenstein(s16, m=4, lag=6, mean_period=12, max_t=40)
    l17, _ = rosenstein(s17, m=4, lag=6, mean_period=12, max_t=40)
    assert l17 > l16
    assert l17 > 0.001                       # chaotic, FN band ~0.002-0.007
    assert abs(l16) < 0.001                   # limit cycle ~ 0


def test_perturbation_pair_sign_matches():
    assert MG.lyapunov_perturbation_pair(17, n=1500, washout=1000) > 0
    assert MG.lyapunov_perturbation_pair(16, n=1500, washout=1000) < 0.002


def test_stateful_fn_one_step_learns_map():
    d = MG.generate(16, n=1200, washout=800)["series"]
    qrc = ExactQRC(fc_tfi(5, J=1.0, h=0.5, seed=7), V=8, tau=4.0)
    ro, info = train_teacher_forced(qrc, d, n_train=800)
    assert info["nmse_dev_onestep"] < 1e-2
    # adapter drives without error and stays finite for a short rollout
    r = autonomous_from_series(StatefulFN(qrc, ro), d, 900, 30)
    assert np.all(np.isfinite(r["predictions"]))


def test_noise_equals_ridge():
    d = MG.generate(16, n=1400, washout=800)["series"]
    qrc = ExactQRC(fc_tfi(5, J=1.0, h=0.5, seed=7), V=8, tau=4.0)
    X = qrc.features(d[:-1], check_budget=False)
    Xa, ya = align(X, d[1:], horizon=0)
    Xtr, ytr = Xa[100:1000], ya[100:1000]
    L = Xtr.shape[0]
    sigma = 0.02
    lam = lambda_eff(sigma, L)
    assert sigma_for_lambda(lam, L) == pytest.approx(sigma)
    W_ridge = fit_readout(Xtr, ytr, lam=lam).W
    Wsum = np.zeros_like(W_ridge)
    for seed in range(30):
        Wsum += fit_readout(add_feature_noise(Xtr, sigma, np.random.default_rng(seed)),
                            ytr, lam=0.0).W
    Wmean = Wsum / 30
    corr = np.corrcoef(W_ridge.ravel(), Wmean.ravel())[0, 1]
    assert corr > 0.95
