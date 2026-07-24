"""schedule.py -- Schedule dataclass consumed by BOTH matrix Trotter and Qiskit.

A ``Schedule`` is the ordered list of ``(gate, sites, angle)`` for ONE evolution
block (advancing time ``block_time`` by ``kappa`` first-order Trotter substeps).
It is produced ONCE per config and consumed by both ``exact_trotter_qrc`` (matrix
products) and ``qiskit_qrc`` (circuit builder), so their parity is structural, not
coincidental. Sites are LOGICAL indices (Qiskit wires are resolved via
``endianness`` at circuit-build time only). Implemented in P3.

Angle convention (pinned by ``test_qiskit_parity``): a Trotter substep of duration
``dt`` for H = sum J_ij X_iX_j + sum h Z_i uses

    RXX(2 J_ij dt)   ==  exp(-i J_ij dt X_i X_j)
    RZ (2 h    dt)   ==  exp(-i h    dt Z_i)

matching Qiskit's RXX(theta)=exp(-i theta/2 XX), RZ(theta)=exp(-i theta/2 Z).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Schedule:
    N: int
    block_time: float
    kappa: int
    order: int
    topology: str
    gates: Tuple[Tuple[str, Tuple[int, ...], float], ...]

    def __iter__(self):
        return iter(self.gates)


def _bonds(reservoir):
    """(i, j, J_ij) for every nonzero coupling in the reservoir (i<j)."""
    J = reservoir.J
    out = []
    N = reservoir.N
    for i in range(N):
        for j in range(i + 1, N):
            if J[i, j] != 0.0:
                out.append((i, j, float(J[i, j])))
    return out


def build_block_schedule(reservoir, block_time, kappa=1, order=1, ordering=None):
    """First-order (or 2nd-order Suzuki) Trotter schedule for one evolution block.

    ``ordering`` optionally permutes the two-qubit bonds (Opt-NN gate ordering,
    Hamhoum Fig. 3/4); default keeps the natural (i<j) order.
    """
    N = reservoir.N
    h = reservoir.h
    dt = block_time / kappa
    bonds = _bonds(reservoir)
    if ordering is not None:
        bonds = [bonds[k] for k in ordering]

    def xx_layer(scale):
        return [("RXX", (i, j), 2.0 * Jij * dt * scale) for (i, j, Jij) in bonds]

    def z_layer(scale):
        return [("RZ", (i,), 2.0 * h * dt * scale) for i in range(N)]

    gates: List[Tuple[str, Tuple[int, ...], float]] = []
    for _ in range(kappa):
        if order == 1:
            gates += xx_layer(1.0)
            gates += z_layer(1.0)
        elif order == 2:                      # symmetric Strang split
            gates += z_layer(0.5)
            gates += xx_layer(1.0)
            gates += z_layer(0.5)
        else:
            raise ValueError(f"unsupported Trotter order {order}")
    return Schedule(N=N, block_time=block_time, kappa=kappa, order=order,
                    topology=reservoir.topology, gates=tuple(gates))
