"""exact_trotter_qrc.py -- exact matrix first-order Trotter mirror of the circuit.

Same FN reservoir as ``exact_qrc`` but the per-virtual-node evolution is the
Trotterised block from ``schedule.build_block_schedule`` applied as dense matrix
products (closed-form single/two-qubit exponentials), rather than the exact
matrix exponential. The gap between this and ``ExactQRC`` at the same seed is the
Trotter error and nothing else (validation ladder L1 vs L2). The SAME Schedule
drives the Qiskit circuit (L2 vs L3), so those agree by construction.
Implemented in P3.
"""
from __future__ import annotations

import numpy as np

from .observables import build_observables
from .input_channels import inject
from .pauli import X, Z
from .schedule import build_block_schedule
from .tensor import embed, embed_pair


def gate_unitary(name, sites, angle, N):
    """Dense N-qubit unitary for one schedule gate (closed form; XX/Z have eigvals +-1)."""
    c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
    dim = 2 ** N
    I = np.eye(dim, dtype=complex)
    if name == "RXX":
        G = embed_pair(X, sites[0], X, sites[1], N)      # X_i X_j (eigvals +-1)
    elif name == "RZ":
        G = embed(Z, sites[0], N)
    else:
        raise ValueError(f"unknown gate {name}")
    return c * I - 1j * s * G


def block_unitary(schedule, N):
    """Dense unitary for a whole block: gates applied in listed order (left-multiply)."""
    U = np.eye(2 ** N, dtype=complex)
    for name, sites, angle in schedule:
        U = gate_unitary(name, sites, angle, N) @ U
    return U


class ExactTrotterQRC:
    """FN reservoir evolved by matrix Trotter blocks (mirror of ExactQRC)."""

    def __init__(self, reservoir, V=10, tau=2.0, kappa=1, order=1,
                 observable_kind="z_local", observable_strings=None,
                 inject_site=0, ordering=None):
        self.res = reservoir
        self.N = reservoir.N
        self.V = int(V)
        self.tau = float(tau)
        self.kappa = int(kappa)
        self.inject_site = inject_site
        self.ops, self.obs_labels = build_observables(
            self.N, observable_kind, observable_strings)
        self.M = len(self.ops)
        # one virtual-node interval tau/V, Trotterised with `kappa` substeps
        self.schedule = build_block_schedule(
            reservoir, block_time=self.tau / self.V, kappa=self.kappa,
            order=order, ordering=ordering)
        self.U_unit = block_unitary(self.schedule, self.N)

    @property
    def feature_dim(self):
        return self.M * self.V + 1

    def initial_state(self):
        dim = 2 ** self.N
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0
        return rho

    def _obs(self, rho):
        return np.array([np.trace(rho @ op).real for op in self.ops])

    def features(self, inputs, x0=None, include_bias=True, check_budget=True):
        s = np.asarray(inputs, dtype=float)
        L = len(s)
        per_step = self.M * self.V
        if check_budget:
            fc = per_step + (1 if include_bias else 0)
            if fc > L / 5:
                raise ValueError(
                    f"feature count {fc} exceeds L/5 = {L/5:.1f} (FN overdetermination)")
        rho = self.initial_state() if x0 is None else np.array(x0, dtype=complex)
        Ud = self.U_unit.conj().T
        rows = np.empty((L, per_step))
        for k in range(L):
            rho = inject(rho, float(s[k]), self.N, site=self.inject_site)
            block = np.empty((self.V, self.M))
            for v in range(self.V):
                rho = self.U_unit @ rho @ Ud
                block[v] = 0.5 * (1.0 + self._obs(rho))
            rows[k] = block.reshape(-1)
        if include_bias:
            rows = np.hstack([rows, np.ones((L, 1))])
        return rows
