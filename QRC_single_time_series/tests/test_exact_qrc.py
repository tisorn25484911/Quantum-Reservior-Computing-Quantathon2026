"""P2 anchors: exact FN reservoir, G3 STM tau_B=0 exactness, features==run, budget."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.quantum.hamiltonians import fc_tfi  # noqa: E402
from qrc_single_time_series.quantum.exact_qrc import ExactQRC  # noqa: E402
from qrc_single_time_series.quantum.pauli import site_expect  # noqa: E402
from qrc_single_time_series.data.windows import align  # noqa: E402
from qrc_single_time_series.models import readout as R  # noqa: E402
from qrc_single_time_series.evaluation import metrics as M  # noqa: E402

NS = [2, 3, 5]


def make_qrc(N, V=10, tau=2.0):
    return ExactQRC(fc_tfi(N, J=1.0, h=0.5, seed=7), V=V, tau=tau)


@pytest.mark.parametrize("N", NS)
def test_feature_dim_and_range(N):
    qrc = make_qrc(N)
    s = np.random.default_rng(0).uniform(0, 1, 600)
    X = qrc.features(s)
    assert X.shape == (600, qrc.feature_dim)
    assert qrc.feature_dim == N * qrc.V + 1
    # x' = (1+<Z>)/2 in [0,1]; bias column is exactly 1.
    assert X[:, :-1].min() >= -1e-12 and X[:, :-1].max() <= 1 + 1e-12
    assert np.allclose(X[:, -1], 1.0)


@pytest.mark.parametrize("N", NS)
def test_features_match_run_generator(N):
    qrc = make_qrc(N, V=6)
    s = np.random.default_rng(1).uniform(0, 1, 40)
    X = qrc.features(s, check_budget=False)
    z0 = defaultdict(dict)
    for k, v, rho in qrc.run(s):
        z0[k][v] = 0.5 * (1 + site_expect(rho, "Z", 0, N))
    for k in range(len(s)):
        for v in range(qrc.V):
            assert X[k, v * qrc.M + 0] == pytest.approx(z0[k][v], abs=1e-10)


def test_g3_stm_tau0_exactness():
    # STM tau_B = 0 with the debug post-injection sample: <Z_inj> = 1 - 2 s_k
    # exactly, so a linear lambda=0 readout recovers the input -> C(0) = 1.0.
    qrc = make_qrc(5, V=10)
    s = np.random.default_rng(7).uniform(0, 1, 800)
    X = qrc.features(s, sample="postinjection", check_budget=False)
    Xa, ya = align(X, s, horizon=0)
    ntr = 500
    ro = R.fit(Xa[:ntr], ya[:ntr], lam=0.0)
    pred = ro.predict(Xa[ntr:])
    assert M.capacity(ya[ntr:], pred) == pytest.approx(1.0, abs=1e-9)
    assert M.nmse_variance(ya[ntr:], pred) < 1e-12


def test_budget_assert_refuses_short_series():
    qrc = make_qrc(5, V=10)          # feature dim 51 -> needs L > 255
    with pytest.raises(ValueError, match="overdetermination"):
        qrc.features(np.random.default_rng(0).uniform(0, 1, 100))


def test_reservoir_is_deterministic():
    s = np.random.default_rng(3).uniform(0, 1, 600)
    assert np.allclose(make_qrc(5).features(s), make_qrc(5).features(s))


def test_float32_cache_within_one_percent():
    # float32 feature cache vs float64 recompute: NMSE within 1% (G4 cache test).
    qrc = make_qrc(5, V=10)
    s = np.random.default_rng(5).uniform(0, 1, 800)
    X64 = qrc.features(s)
    X32 = X64.astype(np.float32).astype(np.float64)
    ntr = 500
    _, tgt = align(X64, np.roll(s, -1), 0)  # arbitrary smooth-ish target
    ro64 = R.fit(X64[:ntr], s[:ntr], lam=1e-6)
    ro32 = R.fit(X32[:ntr], s[:ntr], lam=1e-6)
    p64 = ro64.predict(X64[ntr:])
    p32 = ro32.predict(X32[ntr:])
    n64 = M.nmse_variance(s[ntr:], p64)
    n32 = M.nmse_variance(s[ntr:], p32)
    assert abs(n32 - n64) / max(n64, 1e-12) < 0.01


def test_memory_capacity_grows_with_virtual_nodes():
    # FN Fig.5 direction: short-term memory capacity rises with V.
    def total_mc(V):
        qrc = make_qrc(5, V=V)
        s = np.random.default_rng(0).uniform(0, 1, 1500)
        X = qrc.features(s, check_budget=False)
        ntr = 1000
        mc = 0.0
        for d in range(1, 8):
            Xa, _ = align(X, s, 0)
            tgt = s.copy()
            Xd, yd = X[d:], tgt[:len(tgt) - d]   # feature at k vs input d steps back
            ro = R.fit(Xd[:ntr], yd[:ntr], lam=1e-8)
            pred = ro.predict(Xd[ntr:])
            mc += M.capacity(yd[ntr:], pred)
        return mc
    mc1, mc10 = total_mc(1), total_mc(10)
    assert mc10 > mc1
