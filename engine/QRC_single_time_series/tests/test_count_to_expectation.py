"""P3 anchor (G2): counts -> <Z_i> through the ONE endianness function."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qiskit import QuantumCircuit  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

from qrc_single_time_series.quantum import endianness as E  # noqa: E402

SIM = AerSimulator()
NS = [3, 5, 8]


def test_wire_map_is_involution():
    for N in NS:
        for i in range(N):
            assert E.wire_to_logical(E.logical_to_wire(i, N), N) == i


def test_reverse_key():
    assert E.reverse_key("100") == "001"


def test_counts_to_z_noiseless_ground_state():
    counts = {"000": 1000}
    assert np.allclose(E.counts_to_z(counts, 3), [1, 1, 1])


@pytest.mark.parametrize("N", NS)
def test_x_on_each_logical_site_gives_minus_one(N):
    # X on logical qubit j -> <Z_j> = -1, all others +1 (the G2 anchor).
    for j in range(N):
        qc = QuantumCircuit(N)
        qc.x(E.logical_to_wire(j, N))
        qc.measure_all()
        counts = SIM.run(qc, shots=4000, seed_simulator=1).result().get_counts()
        z = E.counts_to_z(counts, N)
        expected = np.ones(N)
        expected[j] = -1.0
        assert np.allclose(z, expected), (j, z)


def test_probabilities_consistent_with_z():
    counts = {"010": 700, "000": 300}      # logical qubit 1 is 1 in 70% of shots
    p = E.counts_to_probabilities(counts, 3)
    z = E.counts_to_z(counts, 3)
    assert np.allclose(p, 0.5 * (1 - z))
    assert p[1] == pytest.approx(0.7)
