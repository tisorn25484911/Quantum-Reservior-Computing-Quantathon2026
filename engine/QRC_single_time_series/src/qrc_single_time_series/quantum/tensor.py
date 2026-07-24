"""tensor.py -- kron/tensor helpers with the injection-qubit-leftmost contract (G2).

Convention (G2, IMPLEMENTATION_PHASES.md s0): in all NumPy/SciPy matrix code the
qubit with logical index 0 is the *leftmost* tensor factor in ``np.kron`` order
(the most-significant bit of the computational-basis index). The injection qubit
is always logical index 0, so re-injection needs no permutation. A single-site
operator ``op`` acting on logical site ``i`` of an ``N``-qubit register embeds as

    I(2**i)  (x)  op  (x)  I(2**(N - 1 - i)).

Implemented in P1.
"""
from __future__ import annotations

from functools import reduce

import numpy as np

I2 = np.eye(2, dtype=complex)


def kron_all(ops):
    """Kronecker product of a sequence of matrices, left-to-right (G2 order).

    ``kron_all([A, B, C]) == np.kron(np.kron(A, B), C)`` so the first element is
    the leftmost / most-significant tensor factor.
    """
    ops = list(ops)
    if not ops:
        raise ValueError("kron_all requires at least one operator")
    return reduce(np.kron, ops)


def embed(op, site, N):
    """Embed a single-qubit operator ``op`` at logical ``site`` of ``N`` qubits.

    Site 0 is the leftmost tensor factor (G2). Identity pads both sides.
    """
    op = np.asarray(op, dtype=complex)
    if op.shape != (2, 2):
        raise ValueError(f"embed expects a 2x2 operator, got {op.shape}")
    if not 0 <= site < N:
        raise IndexError(f"site {site} out of range for N={N}")
    left = np.eye(2 ** site, dtype=complex)
    right = np.eye(2 ** (N - 1 - site), dtype=complex)
    return kron_all([left, op, right])


def embed_pair(op_a, site_a, op_b, site_b, N):
    """Embed two single-qubit operators at distinct sites (product of embeddings).

    ``op_a`` and ``op_b`` commute as embedded operators (disjoint supports), so
    the product is order-independent.
    """
    if site_a == site_b:
        raise ValueError("embed_pair requires distinct sites")
    return embed(op_a, site_a, N) @ embed(op_b, site_b, N)
