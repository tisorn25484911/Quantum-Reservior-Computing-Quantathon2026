"""endianness.py -- the ONE logical<->Qiskit-wire + counts-key reversal (G2).

The single place the code crosses between the NumPy/SciPy convention and Qiskit.

Conventions
-----------
- NumPy/SciPy (this project): logical qubit 0 is the *leftmost* tensor factor, i.e.
  the MOST-significant bit of the computational-basis index (see ``tensor.embed``).
- Qiskit: qubit 0 is the LEAST-significant bit; a printed counts key is MSB-first,
  so its leftmost character is the highest-indexed wire.

To make a Qiskit measurement come back already in logical order we map logical
site ``i`` to wire ``N-1-i``. Then a measure-all counts key, read left to right,
is indexed directly by logical qubit: ``key[i]`` is the outcome of logical qubit
``i``. This is the ONLY function pair allowed to reverse the ordering (G2).
Implemented in P3.
"""
from __future__ import annotations

import numpy as np


def logical_to_wire(i, N):
    """Logical qubit index -> Qiskit wire index (the single reversal)."""
    if not 0 <= i < N:
        raise IndexError(f"logical index {i} out of range for N={N}")
    return N - 1 - i


def wire_to_logical(q, N):
    """Qiskit wire index -> logical qubit index (inverse of ``logical_to_wire``)."""
    if not 0 <= q < N:
        raise IndexError(f"wire index {q} out of range for N={N}")
    return N - 1 - q


def reverse_key(key):
    """Reverse a counts bitstring (Qiskit little-endian <-> logical big-endian)."""
    return key[::-1]


def _logical_bits(key, N):
    """Outcome bits indexed by logical qubit, given a measure-all key on wires N-1..0.

    With the ``logical_to_wire`` mapping, the printed key (MSB-first over wires) is
    already logical-ordered, so ``key[i]`` is logical qubit ``i``.
    """
    key = key.replace(" ", "")
    if len(key) != N:
        raise ValueError(f"counts key {key!r} width != N={N}")
    return np.array([int(c) for c in key], dtype=int)


def counts_to_z(counts, N):
    """<Z_i> = P(bit_i = 0) - P(bit_i = 1) per logical qubit, from a counts dict.

    Assumes the circuit measured all wires and was built with ``logical_to_wire``.
    Returns a length-N real array in logical order.
    """
    total = sum(counts.values())
    if total == 0:
        raise ValueError("empty counts")
    z = np.zeros(N)
    for key, n in counts.items():
        bits = _logical_bits(key, N)
        z += n * (1.0 - 2.0 * bits)      # +1 for bit 0, -1 for bit 1
    return z / total


def counts_to_probabilities(counts, N):
    """Per-logical-qubit P(bit = 1), logical order (convenience for readout error)."""
    return 0.5 * (1.0 - counts_to_z(counts, N))
