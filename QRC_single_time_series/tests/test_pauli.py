"""P1 anchors: Pauli algebra, G2 site embedding, expectations, Pauli strings."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.quantum import pauli as P  # noqa: E402
from qrc_single_time_series.quantum.tensor import embed, kron_all  # noqa: E402

NS = [2, 3, 5]


def test_pauli_algebra():
    assert np.allclose(P.X @ P.X, P.I)
    assert np.allclose(P.Y @ P.Y, P.I)
    assert np.allclose(P.Z @ P.Z, P.I)
    assert np.allclose(P.X @ P.Y, 1j * P.Z)      # cyclic
    assert np.allclose(P.Y @ P.Z, 1j * P.X)
    assert np.allclose(P.Z @ P.X, 1j * P.Y)


@pytest.mark.parametrize("N", NS)
def test_embed_is_leftmost_site0(N):
    # G2: site 0 is the leftmost tensor factor.
    manual = kron_all([P.Z] + [P.I] * (N - 1))
    assert np.allclose(P.site_operator("Z", 0, N), manual)
    # site N-1 is the rightmost factor.
    manual_last = kron_all([P.I] * (N - 1) + [P.Z])
    assert np.allclose(P.site_operator("Z", N - 1, N), manual_last)


@pytest.mark.parametrize("N", NS)
def test_embed_dimension_and_hermiticity(N):
    for c in "XYZ":
        op = P.site_operator(c, 0, N)
        assert op.shape == (2 ** N, 2 ** N)
        assert np.allclose(op, op.conj().T)


def test_embed_out_of_range():
    with pytest.raises(IndexError):
        embed(P.Z, 3, 3)


@pytest.mark.parametrize("N", NS)
def test_expect_on_basis_states(N):
    # |0...0><0...0|: <Z_i> = +1 on every site.
    zero = np.zeros((2 ** N, 2 ** N), dtype=complex)
    zero[0, 0] = 1.0
    for i in range(N):
        assert P.site_expect(zero, "Z", i, N) == pytest.approx(1.0)


def test_pauli_string_matches_kron():
    assert np.allclose(P.pauli_string("ZIX"), kron_all([P.Z, P.I, P.X]))


def test_pauli_string_rejects_garbage():
    with pytest.raises(ValueError):
        P.pauli_string("ZQX")


def test_expect_rejects_non_real():
    # X has zero diagonal -> <X> = 0 on |0><0|; a non-Hermitian op gives complex.
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    with pytest.raises(ValueError):
        P.expect(rho, np.array([[1j, 0], [0, 0]], dtype=complex))
