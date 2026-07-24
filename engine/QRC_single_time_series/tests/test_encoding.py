"""P1 anchors: injection state rho_s, <Z_inj>/<X_inj>, untouched sites, purity decay."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.quantum import pauli as P  # noqa: E402
from qrc_single_time_series.quantum import input_channels as ic  # noqa: E402
from qrc_single_time_series.quantum import hamiltonians as ham  # noqa: E402
from qrc_single_time_series.quantum.tensor import kron_all  # noqa: E402

NS = [2, 3, 5]
SVALS = [0.0, 0.25, 0.5, 0.73, 1.0]


@pytest.mark.parametrize("s", SVALS)
def test_rho_s_is_valid_state(s):
    r = ic.rho_s(s)
    assert np.trace(r).real == pytest.approx(1.0)
    assert np.allclose(r, r.conj().T)
    assert np.linalg.eigvalsh(r).min() >= -1e-12
    assert ic.purity(r) == pytest.approx(1.0)     # rho_s is pure


@pytest.mark.parametrize("s", SVALS)
def test_rho_s_expectations(s):
    r = ic.rho_s(s)
    assert P.expect(r, P.Z) == pytest.approx(1 - 2 * s)
    assert P.expect(r, P.X) == pytest.approx(2 * np.sqrt(s * (1 - s)))


def test_ry_density_matches_rho_s():
    theta = 0.9
    assert np.allclose(ic.ry_density(theta), ic.rho_s(np.sin(theta / 2) ** 2))


@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("s", [0.2, 0.6])
def test_injection_sets_inj_qubit_leaves_others(N, s):
    # Start from |+> on every site, evolve, then inject at site 0.
    rho = kron_all([P.PLUS] * N)
    U = ham.fc_tfi(N, seed=7).evolution(0.5)
    rho = U @ rho @ U.conj().T
    before = [P.site_expect(rho, "Z", i, N) for i in range(N)]
    inj = ic.inject(rho, s, N, site=0)
    assert P.site_expect(inj, "Z", 0, N) == pytest.approx(1 - 2 * s)
    assert P.site_expect(inj, "X", 0, N) == pytest.approx(2 * np.sqrt(s * (1 - s)))
    # Reduced state of the other sites is untouched by injecting on site 0.
    for i in range(1, N):
        assert P.site_expect(inj, "Z", i, N) == pytest.approx(before[i], abs=1e-12)


@pytest.mark.parametrize("N", [2, 3])
def test_purity_falls_from_one_to_plateau(N):
    # Repeated inject-then-evolve mixes the reservoir: purity drops from 1 and
    # settles to a plateau above the maximally-mixed floor 1/2**N.
    rho = kron_all([P.KET0] * N)
    res = ham.fc_tfi(N, seed=7)
    U = res.evolution(1.0)
    purities = [ic.purity(rho)]
    rng = np.random.default_rng(0)
    for _ in range(40):
        rho = ic.inject(rho, float(rng.uniform(0, 1)), N, site=0)
        rho = U @ rho @ U.conj().T
        purities.append(ic.purity(rho))
    assert purities[0] == pytest.approx(1.0)
    assert purities[-1] < 0.95
    assert purities[-1] >= 1.0 / 2 ** N - 1e-9
    # late-time purity is more stable than the initial transient (plateau).
    late = np.array(purities[-10:])
    assert late.std() < 0.15
