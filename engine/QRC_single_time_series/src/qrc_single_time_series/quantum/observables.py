"""observables.py -- observable set (Pauli strings) for reservoir readout features.

The reservoir feature at virtual node v is x'_{n,v} = (1 + <O_n>)/2 for each
observable O_n in the configured set. The default FN set is the local Z on every
qubit (``z_local``); ``z_and_zz`` adds the two-body ZZ correlators; ``pauli``
takes an explicit list of Pauli strings. Every operator is returned as a dense
Hermitian matrix plus a human label; ``is_diagonal`` flags the Z-family (all
returned operators diagonal in the computational basis), which a later fast path
can exploit. Implemented in P2.
"""
from __future__ import annotations

import numpy as np

from .pauli import Z, pauli_string, site_operator


def build_observables(N, kind="z_local", strings=None):
    """Return ``(ops, labels)`` for the requested observable set on ``N`` qubits.

    - ``z_local``: [Z_0, ..., Z_{N-1}]                       (M = N)
    - ``z_and_zz``: local Z plus every ZZ correlator Z_iZ_j  (M = N + N(N-1)/2)
    - ``pauli``: explicit ``strings`` list, e.g. ["ZII", "IXI"]
    """
    ops, labels = [], []
    if kind == "z_local":
        for i in range(N):
            ops.append(site_operator("Z", i, N))
            labels.append(f"Z{i}")
    elif kind == "z_and_zz":
        for i in range(N):
            ops.append(site_operator("Z", i, N))
            labels.append(f"Z{i}")
        for i in range(N):
            for j in range(i + 1, N):
                ops.append(site_operator("Z", i, N) @ site_operator("Z", j, N))
                labels.append(f"Z{i}Z{j}")
    elif kind == "pauli":
        if not strings:
            raise ValueError("kind='pauli' requires a non-empty `strings` list")
        for s in strings:
            if len(s) != N:
                raise ValueError(f"Pauli string {s!r} length != N={N}")
            ops.append(pauli_string(s))
            labels.append(s)
    else:
        raise ValueError(f"unknown observable kind {kind!r}")
    return ops, labels


def is_diagonal(ops, tol=1e-12):
    """True if every operator in ``ops`` is diagonal in the computational basis."""
    return all(np.max(np.abs(op - np.diag(np.diag(op)))) <= tol for op in ops)
