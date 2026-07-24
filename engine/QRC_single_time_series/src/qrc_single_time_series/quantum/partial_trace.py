"""partial_trace.py -- partial trace and site re-insertion via reshape/einsum (G2).

The register is ``N`` qubits with logical site 0 as the leftmost tensor factor
(G2). A density matrix ``rho`` of shape (2**N, 2**N) is viewed as a rank-2N tensor
with ket axes 0..N-1 and bra axes N..2N-1. ``partial_trace`` contracts the ket/bra
pair of every traced site; ``reinsert`` is its structural inverse, tensoring a
single-qubit state back in at a chosen site. Together they implement the CPTP
injection channel used by ``input_channels`` (the replacement is partial-trace-then
-tensor, never a unitary on a statevector -- spec s8.3). Implemented in P1.
"""
from __future__ import annotations

import string

import numpy as np


def _check(rho, N):
    dim = 2 ** N
    if rho.shape != (dim, dim):
        raise ValueError(f"rho shape {rho.shape} inconsistent with N={N} (expected {(dim, dim)})")


def partial_trace(rho, keep, N):
    """Partial trace of ``rho`` down to the qubits in ``keep`` (ascending order).

    Returns the reduced density matrix on ``len(keep)`` qubits, ordered by
    ascending logical index. Traced sites are the complement of ``keep``.
    """
    _check(rho, N)
    keep = sorted(keep)
    if any(not 0 <= k < N for k in keep) or len(set(keep)) != len(keep):
        raise ValueError(f"invalid keep set {keep} for N={N}")
    if 2 * N > len(string.ascii_letters):
        raise ValueError("N too large for einsum-based partial trace")

    ket = list(string.ascii_lowercase[:N])
    bra = list(string.ascii_uppercase[:N])
    traced = [i for i in range(N) if i not in keep]
    for t in traced:                # share a label -> summed (traced) axis
        bra[t] = ket[t]

    t = rho.reshape([2] * (2 * N))
    subscript = "".join(ket) + "".join(bra)
    out_labels = "".join(ket[k] for k in keep) + "".join(bra[k] for k in keep)
    reduced = np.einsum(f"{subscript}->{out_labels}", t)
    d = 2 ** len(keep)
    return reduced.reshape(d, d)


def trace_out(rho, sites, N):
    """Convenience: trace *out* the given ``sites``, keeping the rest (ascending)."""
    keep = [i for i in range(N) if i not in set(sites)]
    return partial_trace(rho, keep, N)


def reinsert(reduced, rho1, site, N):
    """Tensor a single-qubit state ``rho1`` back in at logical ``site``.

    ``reduced`` is a density matrix on the other ``N-1`` qubits (ascending order),
    as produced by ``trace_out(rho, [site], N)``. Returns an ``N``-qubit density
    matrix with ``rho1`` occupying ``site`` and ``reduced`` filling the rest in
    their original relative order. This is the structural inverse of tracing out
    ``site`` for a state that was a product across that cut.
    """
    if not 0 <= site < N:
        raise IndexError(f"site {site} out of range for N={N}")
    _check(reduced, N - 1)
    rho1 = np.asarray(rho1, dtype=complex)
    if rho1.shape != (2, 2):
        raise ValueError(f"rho1 must be 2x2, got {rho1.shape}")
    if 2 * N > len(string.ascii_letters):
        raise ValueError("N too large for einsum-based reinsert")

    keep = [i for i in range(N) if i != site]     # sites carried by ``reduced``
    ket = list(string.ascii_lowercase[:N])
    bra = list(string.ascii_uppercase[:N])

    red_t = reduced.reshape([2] * (2 * (N - 1)))
    red_sub = "".join(ket[k] for k in keep) + "".join(bra[k] for k in keep)
    one_sub = ket[site] + bra[site]
    out_sub = "".join(ket) + "".join(bra)
    full = np.einsum(f"{red_sub},{one_sub}->{out_sub}", red_t, rho1)
    d = 2 ** N
    return full.reshape(d, d)
