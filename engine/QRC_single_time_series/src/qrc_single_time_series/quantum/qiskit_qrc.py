"""qiskit_qrc.py -- ideal Qiskit reservoir (statevector / density_matrix / shots).

Builds the FN reservoir circuit from the SAME ``Schedule`` the matrix Trotter
mirror uses, so parity with ``exact_trotter_qrc`` is structural (validation ladder
L2 vs L3). Injection is a mid-circuit ``reset`` followed by ``ry(theta)`` with
theta = 2 arcsin(sqrt(s)) (prepares rho_s(s)); virtual nodes are read WITHOUT
destructive measurement via saved density matrices (simulation-only, G8-7).
Wire/counts ordering crosses to Qiskit through ``endianness`` only (G2).

Shot emulation (the consistent cheap path used across ALL models on the FN track,
G8-7): for an exact expectation z in [-1,1], p=(1+z)/2, sigma=2 sqrt(p(1-p)/S),
emulate clip(z + xi sigma, -1, 1), xi ~ N(0,1), seeded per feature per step.
Implemented in P3.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Pauli
from qiskit_aer import AerSimulator

from . import endianness as E
from .observables import build_observables
from .schedule import build_block_schedule


def block_circuit(schedule, N, qc=None):
    """Append one Schedule block to ``qc`` (or a fresh circuit), wires via endianness."""
    if qc is None:
        qc = QuantumCircuit(N)
    for name, sites, angle in schedule:
        wires = [E.logical_to_wire(i, N) for i in sites]
        if name == "RXX":
            qc.rxx(angle, wires[0], wires[1])
        elif name == "RZ":
            qc.rz(angle, wires[0])
        else:
            raise ValueError(f"unknown gate {name}")
    return qc


def block_unitary_logical(schedule, N):
    """Dense block unitary in the LOGICAL (big-endian) basis, from the Qiskit circuit.

    ``block_circuit`` already places logical site ``i`` on wire ``N-1-i`` (the G2
    endianness map), so the plain ``Operator`` is already in the project's logical
    MSB-first basis and equals ``exact_trotter_qrc.block_unitary`` (L2 vs L3).
    """
    from qiskit.quantum_info import Operator
    return Operator(block_circuit(schedule, N)).data


def _theta(s):
    return 2.0 * np.arcsin(np.sqrt(np.clip(s, 0.0, 1.0)))


def shot_emulate(z, shots, rng):
    """Apply the consistent finite-shot noise model to exact expectations ``z``."""
    z = np.asarray(z, dtype=float)
    p = 0.5 * (1.0 + z)
    sigma = 2.0 * np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / shots)
    return np.clip(z + rng.standard_normal(z.shape) * sigma, -1.0, 1.0)


class QiskitReservoir:
    """Ideal FN reservoir on Aer's density_matrix method (deterministic reference)."""

    def __init__(self, reservoir, V=10, tau=2.0, kappa=1, order=1,
                 observable_kind="z_local", inject_site=0, ordering=None):
        self.res = reservoir
        self.N = reservoir.N
        self.V = int(V)
        self.tau = float(tau)
        self.inject_site = inject_site
        self.ops, self.obs_labels = build_observables(self.N, observable_kind)
        self.M = len(self.ops)
        if observable_kind != "z_local":
            raise NotImplementedError("QiskitReservoir currently reads the Z-family only")
        self.schedule = build_block_schedule(
            reservoir, block_time=self.tau / self.V, kappa=kappa, order=order,
            ordering=ordering)
        self.sim = AerSimulator(method="density_matrix")

    @property
    def feature_dim(self):
        return self.M * self.V + 1

    def _z_from_dm(self, dm):
        return np.array([dm.expectation_value(Pauli("Z"),
                        [E.logical_to_wire(i, self.N)]).real for i in range(self.N)])

    def features(self, inputs, include_bias=True, check_budget=True,
                 shots=None, sampling_seed=101):
        s = np.asarray(inputs, dtype=float)
        L = len(s)
        per_step = self.M * self.V
        if check_budget and per_step + (1 if include_bias else 0) > L / 5:
            raise ValueError("feature count exceeds L/5 (FN overdetermination)")
        inj_wire = E.logical_to_wire(self.inject_site, self.N)

        qc = QuantumCircuit(self.N)
        for k in range(L):
            qc.reset(inj_wire)
            qc.ry(_theta(s[k]), inj_wire)
            for v in range(self.V):
                block_circuit(self.schedule, self.N, qc)
                qc.save_density_matrix(label=f"n_{k}_{v}")
        data = self.sim.run(qc).result().data(0)

        rows = np.empty((L, per_step))
        for k in range(L):
            block = np.empty((self.V, self.M))
            for v in range(self.V):
                z = self._z_from_dm(DensityMatrix(data[f"n_{k}_{v}"]))
                block[v] = 0.5 * (1.0 + z)
            rows[k] = block.reshape(-1)

        if shots is not None:
            rng = np.random.default_rng(sampling_seed)
            # features live in [0,1] = (1+z)/2; emulate on z then map back
            z_feat = 2.0 * rows - 1.0
            rows = 0.5 * (1.0 + shot_emulate(z_feat, shots, rng))
        if include_bias:
            rows = np.hstack([rows, np.ones((L, 1))])
        return rows
