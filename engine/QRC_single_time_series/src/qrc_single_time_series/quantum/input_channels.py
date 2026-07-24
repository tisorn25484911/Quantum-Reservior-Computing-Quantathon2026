"""input_channels.py -- injection channel rho -> rho_s (x) Tr_inj(rho); Ry encoding.

Fujii-Nakajima sequential input replacement (Paper A): at each step the injection
qubit (logical site 0, G2) is discarded and re-prepared in the input-dependent
state rho_s, leaving the rest of the reservoir entangled/evolving. Implemented as
partial-trace-then-tensor (CPTP by construction, spec s8.3), never as a unitary on
a statevector. Implemented in P1.

    rho_s = 1/2 ( I + 2 sqrt(s(1-s)) X + (1-2s) Z ),   s in [0, 1]

which is the pure state Ry(theta)|0> with s = sin^2(theta/2); it satisfies
<Z> = 1 - 2s and <X> = 2 sqrt(s(1-s)).
"""
from __future__ import annotations

import numpy as np

from .pauli import I, X, Z
from .partial_trace import reinsert, trace_out


def rho_s(s):
    """Injection density matrix for input value ``s`` in [0, 1] (pure)."""
    if not -1e-12 <= s <= 1 + 1e-12:
        raise ValueError(f"s={s} outside [0, 1]")
    s = float(np.clip(s, 0.0, 1.0))
    return 0.5 * (I + 2.0 * np.sqrt(s * (1.0 - s)) * X + (1.0 - 2.0 * s) * Z)


def ry_density(theta):
    """Density matrix of Ry(theta)|0>; equals ``rho_s(sin^2(theta/2))``."""
    return rho_s(np.sin(theta / 2.0) ** 2)


def inject(rho, s, N, site=0):
    """Apply the injection channel at ``site`` (default 0): discard, re-prepare rho_s(s).

    ``rho`` is an N-qubit density matrix. Returns ``rho_s(s) (x) Tr_site(rho)`` with
    ``rho_s`` placed at ``site`` (G2: site 0 is the leftmost factor). CPTP.
    """
    reduced = trace_out(rho, [site], N)
    return reinsert(reduced, rho_s(s), site, N)


def purity(rho):
    """Tr(rho^2) -- 1 for a pure state, 1/d for the maximally mixed state."""
    return float(np.trace(rho @ rho).real)
