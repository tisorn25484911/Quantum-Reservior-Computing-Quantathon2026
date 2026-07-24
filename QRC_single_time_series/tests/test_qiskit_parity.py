"""P3 validation ladder: angle conventions, L1..L4 parity (spec s12), N in {3,5,8}."""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qiskit.circuit.library import RXXGate, RZGate  # noqa: E402
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, Pauli  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

from qrc_single_time_series.quantum.hamiltonians import fc_tfi  # noqa: E402
from qrc_single_time_series.quantum.pauli import X, Z  # noqa: E402
from qrc_single_time_series.quantum import endianness as E  # noqa: E402
from qrc_single_time_series.quantum.schedule import build_block_schedule  # noqa: E402
from qrc_single_time_series.quantum.exact_qrc import ExactQRC  # noqa: E402
from qrc_single_time_series.quantum.exact_trotter_qrc import (  # noqa: E402
    ExactTrotterQRC, block_unitary)
from qrc_single_time_series.quantum.qiskit_qrc import (  # noqa: E402
    QiskitReservoir, block_circuit, block_unitary_logical, shot_emulate)

NS = [3, 5, 8]


def _res(N):
    return fc_tfi(N, J=1.0, h=0.5, seed=7)


# -- angle conventions (kills the factor-of-2 bug class) ---------------------
def test_rxx_angle_convention():
    J, t = 0.37, 1.9
    assert np.allclose(Operator(RXXGate(2 * J * t)).data,
                       sla.expm(-1j * J * t * np.kron(X, X)), atol=1e-12)


def test_rz_angle_convention():
    h, t = 0.5, 1.9
    assert np.allclose(Operator(RZGate(2 * h * t)).data,
                       sla.expm(-1j * h * t * Z), atol=1e-12)


# -- L1 exact dense vs L2 exact matrix Trotter: Trotter error only -----------
@pytest.mark.parametrize("N", NS)
def test_L1_vs_L2_trotter_error_decreases(N):
    res = _res(N)
    s = np.random.default_rng(0).uniform(0, 1, 12)
    l1 = ExactQRC(res, V=3, tau=2.0).features(s, check_budget=False)
    errs = [np.max(np.abs(l1 - ExactTrotterQRC(res, V=3, tau=2.0, kappa=k)
                          .features(s, check_budget=False))) for k in (1, 4, 16)]
    assert errs[0] > errs[1] > errs[2]          # first-order Trotter -> 0 as kappa grows
    assert errs[2] < 0.01


# -- L2 matrix Trotter vs L3 Aer density_matrix: same Schedule => 1e-10 ------
@pytest.mark.parametrize("N", NS)
def test_L2_vs_L3_block_unitary(N):
    sch = build_block_schedule(_res(N), 0.2, kappa=2, order=1)
    assert np.allclose(block_unitary(sch, N), block_unitary_logical(sch, N), atol=1e-10)


@pytest.mark.parametrize("N", NS)
def test_L2_vs_L3_features(N):
    res = _res(N)
    s = np.random.default_rng(1).uniform(0, 1, 10)
    l2 = ExactTrotterQRC(res, V=3, tau=2.0, kappa=2).features(s, check_budget=False)
    l3 = QiskitReservoir(res, V=3, tau=2.0, kappa=2).features(s, check_budget=False)
    assert np.allclose(l2, l3, atol=1e-10)


# -- L3 density_matrix vs statevector stochastic-reset trajectory average ----
def test_L3_vs_statevector_trajectory_within_3SE():
    N = 3
    res = _res(N)
    sch = build_block_schedule(res, 0.2, kappa=2)
    s = np.random.default_rng(0).uniform(0, 1, 6)
    from qrc_single_time_series.quantum.qiskit_qrc import _theta
    inj = E.logical_to_wire(0, N)

    def build(save):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(N)
        for k in range(len(s)):
            qc.reset(inj)
            qc.ry(_theta(s[k]), inj)
            for _ in range(4):
                block_circuit(sch, N, qc)
        (qc.save_density_matrix if save == "dm" else qc.save_statevector)(label="f")
        return qc

    dm = DensityMatrix(AerSimulator(method="density_matrix")
                       .run(build("dm")).result().data(0)["f"])
    zdm = np.array([dm.expectation_value(Pauli("Z"), [E.logical_to_wire(i, N)]).real
                    for i in range(N)])
    sim = AerSimulator(method="statevector")
    qc = build("sv")
    M = 300
    acc = np.zeros(N)
    for m in range(M):
        sv = Statevector(sim.run(qc, shots=1, seed_simulator=1000 + m)
                         .result().data(0)["f"])
        acc += [sv.expectation_value(Pauli("Z"), [E.logical_to_wire(i, N)]).real
                for i in range(N)]
    zmean = acc / M
    assert np.all(np.abs(zdm - zmean) <= 3.0 / np.sqrt(M) + 1e-9)


# -- L3 ideal vs L4 finite shots (shot emulation): binomial scaling ----------
def test_L4_shot_emulation_scaling():
    rng = np.random.default_rng(0)
    z = rng.uniform(-0.9, 0.9, 5000)
    # empirical spread of the emulated estimator shrinks like 1/sqrt(S)
    spreads = []
    for S in (128, 2048):
        dev = shot_emulate(z, S, np.random.default_rng(1)) - z
        spreads.append(np.std(dev))
    ratio = spreads[0] / spreads[1]
    assert 2.5 < ratio < 5.5                    # ~ sqrt(2048/128) = 4
    # infinite shots -> exact
    assert np.max(np.abs(shot_emulate(z, 10 ** 12, rng) - z)) < 1e-4
