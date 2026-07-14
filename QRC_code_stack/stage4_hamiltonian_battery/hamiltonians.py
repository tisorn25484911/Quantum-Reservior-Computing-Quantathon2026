"""hamiltonians.py -- dense Hamiltonians + RMT chaos diagnostics (numpy-only).

Merged module (Phase 3): the spectral-diagnostics side of the stage-4
battery. The circuit side (Pauli-string specs, Trotter emission, reuse
benchmark) lives in benchmark_hamiltonians.py; this module provides

  * dense_from_terms(terms, n)   -- bridge: the benchmark's PauliTerm lists
                                    -> dense Hermitian matrix, so all seven
                                    families feed the same diagnostics;
  * mixed_field_ising(...)       -- the disordered tilted-field Ising chain
                                    with the on-site disorder dial W
                                    (ergodic/GOE at small W -> MBL/Poisson
                                    at large W);
  * level_spacing_ratio(H)       -- mean <r> (Atas et al., PRL 110, 084101
                                    (2013)): Poisson 0.386 / GOE 0.531;
  * chaos_dial_scan(...)         -- <r> versus W, averaged over disorder
                                    realisations.

Ported from the qrc-edge-of-chaos project's qutip implementation; the
qutip dependency is dropped (project decision, Phase 3): all operators are
built with np.kron and diagonalised with np.linalg.eigvalsh. Qubit-order
convention matches stage 1 (qubit 0 = leftmost tensor factor).

Chaos is diagnosed *spectrally* because quantum mechanics admits no
phase-space trajectory: at weak disorder the chain is ergodic/chaotic with
Wigner-Dyson (GOE) level statistics, at strong disorder many-body
localised with Poisson statistics. Disorder also breaks parity/reflection
symmetries, so no symmetry-resolution bookkeeping is needed at N ~ 8-12.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "POISSON_R",
    "GOE_R",
    "GUE_R",
    "PAULI",
    "dense_from_terms",
    "IsingModel",
    "mixed_field_ising",
    "level_spacing_ratio",
    "chaos_dial_scan",
]

# Reference mean level-spacing ratios (Atas et al., PRL 110, 084101, 2013).
POISSON_R = 0.3863
GOE_R = 0.5307
GUE_R = 0.5996

I2 = np.eye(2, dtype=complex)
PAULI = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _op_on_site(op: np.ndarray, site: int, n: int) -> np.ndarray:
    """Embed a single-qubit operator on `site` of an n-qubit register.

    Qubit 0 is the LEFTMOST tensor factor (stage-1 convention).
    """
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, op if i == site else I2)
    return out


def dense_from_terms(terms, n: int) -> np.ndarray:
    """Dense Hermitian matrix from a benchmark PauliTerm list.

    `terms` is a list of (coeff, {qubit_index: 'X'|'Y'|'Z'}) exactly as the
    spec_* functions of benchmark_hamiltonians.py produce, so every family
    in the circuit battery can be diagnosed spectrally with no second
    definition of the Hamiltonian.
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for coeff, sites in terms:
        term = np.array([[1.0 + 0j]])
        for i in range(n):
            term = np.kron(term, PAULI[sites.get(i, "I")])
        H += coeff * term
    if not np.allclose(H, H.conj().T, atol=1e-12):
        raise ValueError("term list produced a non-Hermitian matrix")
    return H


@dataclass
class IsingModel:
    """A constructed mixed-field Ising Hamiltonian and its arrays."""

    H: np.ndarray
    n_qubits: int
    couplings: np.ndarray
    hx: np.ndarray
    hz: np.ndarray
    terms: list = field(default_factory=list)


def mixed_field_ising(n_qubits: int, J: float = 1.0, hx: float = 1.0,
                      hz: float = 0.4, topology: str = "chain",
                      disorder: float = 0.0,
                      seed: int | None = 0) -> IsingModel:
    """Disordered tilted-field Ising chain: H = sum J Z_i Z_j + hx X_i + hz_i Z_i.

    `disorder` is the chaos dial W: on-site longitudinal fields get an
    additive U(-W, W) term per site. Small W -> ergodic/GOE; large W ->
    localised/Poisson. `topology` is 'chain' (nearest-neighbour) or
    'all-to-all'. Returns the dense H plus the PauliTerm list so circuit
    code can Trotterise the identical Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    n = int(n_qubits)

    if topology == "chain":
        pairs = [(i, i + 1) for i in range(n - 1)]
    elif topology == "all-to-all":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError(f"unknown topology {topology!r}")

    onsite = rng.uniform(-disorder, disorder, n) if disorder > 0 else np.zeros(n)
    hx_arr = np.full(n, hx, dtype=float)
    hz_arr = hz + onsite

    Jij = np.zeros((n, n))
    terms = []
    for (i, j) in pairs:
        Jij[i, j] = J
        terms.append((J, {i: "Z", j: "Z"}))
    for i in range(n):
        terms.append((float(hx_arr[i]), {i: "X"}))
        terms.append((float(hz_arr[i]), {i: "Z"}))

    return IsingModel(H=dense_from_terms(terms, n), n_qubits=n,
                      couplings=Jij, hx=hx_arr, hz=hz_arr, terms=terms)


def level_spacing_ratio(H: np.ndarray, central_fraction: float = 0.5) -> float:
    """Mean level-spacing ratio <r> over the central fraction of the spectrum.

    For ordered eigenvalues E_n with gaps s_n = E_{n+1} - E_n,
    r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}). Compare against POISSON_R
    (integrable/localised) and GOE_R / GUE_R (chaotic).
    """
    evals = np.sort(np.real(np.linalg.eigvalsh(np.asarray(H))))

    n = len(evals)
    lo = int((1 - central_fraction) / 2 * n)
    hi = n - lo
    evals = evals[lo:hi]

    gaps = np.diff(evals)
    gaps = gaps[gaps > 1e-12]  # drop (near-)degeneracies
    if len(gaps) < 2:
        return float("nan")
    s1, s2 = gaps[:-1], gaps[1:]
    return float(np.mean(np.minimum(s1, s2) / np.maximum(s1, s2)))


def chaos_dial_scan(n_qubits: int, disorder_values, J: float = 1.0,
                    hx: float = 1.0, hz: float = 0.4,
                    topology: str = "chain", n_realizations: int = 8,
                    base_seed: int = 0) -> np.ndarray:
    """<r> versus the disorder dial W, averaged over disorder realisations."""
    Ws = np.asarray(disorder_values, dtype=float)
    out = np.zeros_like(Ws)
    for idx, W in enumerate(Ws):
        rs = []
        for rep in range(int(n_realizations)):
            model = mixed_field_ising(
                n_qubits, J=J, hx=hx, hz=hz, topology=topology,
                disorder=float(W), seed=base_seed + rep,
            )
            rs.append(level_spacing_ratio(model.H))
        out[idx] = np.nanmean(rs)
    return out
