"""pauli.py -- Pauli operators, strings, tensor embedding at arbitrary site (G2 order).

Single-qubit Paulis, computational-basis states used by the tests, embedding of a
Pauli at a logical site (via ``tensor.embed``, G2: site 0 leftmost), full Pauli
strings, and expectation values Tr(rho . O). Implemented in P1.
"""
from __future__ import annotations

import numpy as np

from .tensor import embed, kron_all

# --- single-qubit operators (dense, complex) -------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}

# --- single-qubit states as density matrices (fixtures for the anchors) ----
KET0 = np.array([[1, 0], [0, 0]], dtype=complex)          # |0><0|
KET1 = np.array([[0, 0], [0, 1]], dtype=complex)          # |1><1|
PLUS = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)    # |+><+|
MINUS = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)  # |-><-|


def site_operator(pauli_char, site, N):
    """Embed a single Pauli (``'I' 'X' 'Y' 'Z'``) at logical ``site`` of ``N`` qubits."""
    return embed(PAULI[pauli_char.upper()], site, N)


def pauli_string(string):
    """Dense operator for a Pauli string, e.g. ``"ZIX"`` (leftmost char = site 0).

    Length of ``string`` is the qubit count N; identity chars are allowed.
    """
    chars = string.upper().strip()
    if not chars or any(c not in PAULI for c in chars):
        raise ValueError(f"invalid Pauli string {string!r}")
    return kron_all([PAULI[c] for c in chars])


def expect(rho, op):
    """Expectation value Tr(rho . op); returns a real float (imag part asserted ~0)."""
    val = np.trace(rho @ op)
    if abs(val.imag) > 1e-9:
        raise ValueError(f"non-real expectation (imag={val.imag:.2e}); check operator/state")
    return float(val.real)


def site_expect(rho, pauli_char, site, N):
    """Single-site Pauli expectation <P_site> for an N-qubit density matrix ``rho``."""
    return expect(rho, site_operator(pauli_char, site, N))
