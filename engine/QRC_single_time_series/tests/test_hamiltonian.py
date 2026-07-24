"""P1 anchors: FC/NN disordered TFI construction, exact evolution, chaos diagnostic."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.quantum import hamiltonians as ham  # noqa: E402

NS = [2, 3, 5]
TAUS = [0.3, 1.0, 2.7]


@pytest.mark.parametrize("N", NS)
def test_hamiltonian_is_hermitian(N):
    H = ham.fc_tfi(N, seed=7).H
    assert np.allclose(H, H.conj().T)


@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("tau", TAUS)
def test_expm_matches_eigendecomposition(N, tau):
    res = ham.fc_tfi(N, seed=7)
    assert np.allclose(res.evolution(tau), res.evolution_expm(tau), atol=1e-12)


@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("tau", TAUS)
def test_evolution_is_unitary(N, tau):
    U = ham.fc_tfi(N, seed=7).evolution(tau)
    assert np.allclose(U.conj().T @ U, np.eye(2 ** N), atol=1e-12)


@pytest.mark.parametrize("N", NS)
def test_couplings_frozen_per_seed(N):
    a = ham.fc_tfi(N, seed=7)
    b = ham.fc_tfi(N, seed=7)
    c = ham.fc_tfi(N, seed=8)
    assert np.array_equal(a.J, b.J) and np.allclose(a.H, b.H)
    if N >= 2:
        assert not np.array_equal(a.J, c.J)


@pytest.mark.parametrize("N", NS)
def test_coupling_bounds_and_upper_triangular(N):
    res = ham.fc_tfi(N, J=1.0, seed=7)
    assert np.all(np.abs(res.J) <= 0.5 + 1e-12)
    assert np.allclose(np.tril(res.J), 0.0)     # stored strictly upper-triangular


def test_nn_is_sparser_than_fc():
    # NN keeps only chain bonds; FC keeps all pairs.
    N = 5
    nn = ham.nn_tfi(N, seed=7)
    fc = ham.fc_tfi(N, seed=7)
    assert np.count_nonzero(nn.J) == N - 1
    assert np.count_nonzero(fc.J) == N * (N - 1) // 2


def test_r_value_signals_chaos_at_n5():
    # Atas anchors: Poisson 0.386, GOE 0.531. A generic disordered FC-TFI at N=5
    # should be well away from the Poisson (integrable) value.
    r = ham.fc_tfi(5, seed=7).r_value()
    assert 0.40 < r < 0.65
