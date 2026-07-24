"""P1 anchors: partial trace, site re-insertion (G2), CPTP injection invariants."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.quantum import pauli as P  # noqa: E402
from qrc_single_time_series.quantum import partial_trace as pt  # noqa: E402
from qrc_single_time_series.quantum import input_channels as ic  # noqa: E402
from qrc_single_time_series.quantum import hamiltonians as ham  # noqa: E402
from qrc_single_time_series.quantum.tensor import kron_all  # noqa: E402

NS = [2, 3, 5]


def random_dm(N, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((2 ** N, 2 ** N)) + 1j * rng.standard_normal((2 ** N, 2 ** N))
    rho = A @ A.conj().T
    return rho / np.trace(rho).real


@pytest.mark.parametrize("N", NS)
def test_partial_trace_of_product_state(N):
    # rho = |0><0| (x) rest;  tracing to site 0 recovers |0><0|.
    factors = [P.KET0] + [P.PLUS] * (N - 1)
    rho = kron_all(factors)
    assert np.allclose(pt.partial_trace(rho, [0], N), P.KET0)
    assert np.allclose(pt.partial_trace(rho, [N - 1], N), P.PLUS)


@pytest.mark.parametrize("N", NS)
def test_partial_trace_preserves_trace(N):
    rho = random_dm(N, seed=N)
    for keep in ([0], [N - 1], list(range(N - 1))):
        red = pt.partial_trace(rho, keep, N)
        assert np.trace(red).real == pytest.approx(1.0)
        assert np.allclose(red, red.conj().T)


def test_g2_reinjection_anchor():
    # Gotcha anchor: |0><0| (x) |1><1| (x) |+><+|, trace out site 1, re-inject |+>.
    rho = kron_all([P.KET0, P.KET1, P.PLUS])
    new = pt.reinsert(pt.trace_out(rho, [1], 3), P.PLUS, 1, 3)
    zs = [P.site_expect(new, "Z", i, 3) for i in range(3)]
    assert zs == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


@pytest.mark.parametrize("N", NS)
def test_reinsert_is_trace_out_inverse_on_product(N):
    # For a state that is a product across the cut, trace-out then reinsert the
    # original single-qubit state returns the exact same density matrix.
    for site in range(N):
        factors = [P.PLUS] * N
        factors[site] = P.KET1
        rho = kron_all(factors)
        back = pt.reinsert(pt.trace_out(rho, [site], N), P.KET1, site, N)
        assert np.allclose(back, rho, atol=1e-12)


@pytest.mark.parametrize("N", NS)
def test_injection_then_evolution_is_cptp(N):
    # Positivity / hermiticity / unit trace preserved by injection then evolution.
    rho = random_dm(N, seed=100 + N)
    U = ham.fc_tfi(N, seed=7).evolution(0.9)
    inj = ic.inject(rho, 0.42, N, site=0)
    out = U @ inj @ U.conj().T
    assert np.trace(out).real == pytest.approx(1.0)
    assert np.allclose(out, out.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(out).min() >= -1e-12
