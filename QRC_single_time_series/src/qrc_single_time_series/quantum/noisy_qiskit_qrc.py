"""noisy_qiskit_qrc.py -- noisy Aer reservoir (synthetic + snapshot noise).

Same circuit as ``QiskitReservoir`` (built from the shared ``Schedule``) run on
Aer's density_matrix method WITH a ``NoiseModel``, so the deviation from the ideal
QiskitReservoir is attributable to the injected channels alone (validation ladder
L4 vs L5). Reads virtual nodes from the noisy density matrix (deterministic in the
channel, no shot noise); an optional finite-shot layer uses the same shot
emulation as the ideal path. Implemented in P3.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Pauli
from qiskit_aer import AerSimulator

from . import endianness as E
from .observables import build_observables
from .qiskit_qrc import block_circuit, shot_emulate, _theta
from .schedule import build_block_schedule


class NoisyQiskitReservoir:
    """FN reservoir on Aer density_matrix with a NoiseModel (L5 of the ladder)."""

    def __init__(self, reservoir, noise_model, V=10, tau=2.0, kappa=1, order=1,
                 observable_kind="z_local", inject_site=0, ordering=None):
        if observable_kind != "z_local":
            raise NotImplementedError("Z-family only")
        self.res = reservoir
        self.N = reservoir.N
        self.V = int(V)
        self.tau = float(tau)
        self.inject_site = inject_site
        self.ops, self.obs_labels = build_observables(self.N, observable_kind)
        self.M = len(self.ops)
        self.schedule = build_block_schedule(
            reservoir, block_time=self.tau / self.V, kappa=kappa, order=order,
            ordering=ordering)
        self.sim = AerSimulator(method="density_matrix", noise_model=noise_model)

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
            rows = 0.5 * (1.0 + shot_emulate(2.0 * rows - 1.0, shots, rng))
        if include_bias:
            rows = np.hstack([rows, np.ones((L, 1))])
        return rows
