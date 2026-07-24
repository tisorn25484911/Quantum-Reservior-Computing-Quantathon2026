"""hamiltonians.py -- FC-TFI (FN) and NN-TFI (Hamhoum) disordered Ising Hamiltonians.

Fujii-Nakajima (Paper A) fully-connected transverse-field Ising model

    H = sum_{i<j} J_ij X_i X_j + sum_i h Z_i,   J_ij ~ U[-J/2, J/2]

with couplings frozen once per ``reservoir_seed`` (spec s8.2). Hamhoum (Paper B)
uses the same form restricted to nearest neighbours on a chain. Exact evolution
U_tau = exp(-i H tau) is obtained by eigendecomposition (H Hermitian -> eigh); a
scipy.linalg.expm path is provided and matched to it in the tests. The mean
adjacent-gap ratio <r> (Atas et al. 2013) certifies chaotic vs integrable configs.
Implemented in P1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from .pauli import X, Z
from .tensor import embed, embed_pair


@dataclass
class Reservoir:
    """A frozen disordered Ising reservoir Hamiltonian and its exact dynamics."""

    N: int
    H: np.ndarray
    J: np.ndarray            # (N, N) upper-triangular coupling matrix (symmetric fill 0)
    h: float
    seed: int
    topology: str            # "fc" (fully connected) or "nn" (nearest neighbour)
    _eig: tuple = field(default=None, repr=False, compare=False)

    # --- spectrum / evolution ------------------------------------------------
    def eig(self):
        """Cached eigendecomposition (eigenvalues ascending, unitary eigenvectors)."""
        if self._eig is None:
            w, V = np.linalg.eigh(self.H)
            self._eig = (w, V)
        return self._eig

    @property
    def eigenvalues(self):
        return self.eig()[0]

    def evolution(self, tau):
        """Exact U_tau = exp(-i H tau) via eigendecomposition (H = V diag(w) V^dag)."""
        w, V = self.eig()
        phases = np.exp(-1j * w * tau)
        return (V * phases) @ V.conj().T

    def evolution_expm(self, tau):
        """Reference U_tau via scipy.linalg.expm (matched to ``evolution`` in tests)."""
        return sla.expm(-1j * self.H * tau)

    # --- diagnostics ---------------------------------------------------------
    def r_value(self):
        """Mean adjacent-gap ratio <r> over the central 50% of the spectrum.

        Near-degenerate gaps (< 1e-12) are dropped. Atas anchors: Poisson 0.3863,
        GOE 0.5307 (used later to certify a config as chaotic).
        """
        w = np.sort(self.eigenvalues)
        lo, hi = len(w) // 4, len(w) - len(w) // 4
        w = w[lo:hi]
        gaps = np.diff(w)
        gaps = gaps[gaps > 1e-12]
        if len(gaps) < 2:
            return float("nan")
        r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
        return float(np.mean(r))


def fc_tfi(N, J=1.0, h=1.0, seed=7):
    """Fully-connected disordered TFI (Fujii-Nakajima). Couplings frozen by ``seed``."""
    rng = np.random.default_rng(seed)
    Jmat = np.zeros((N, N))
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        for j in range(i + 1, N):
            Jij = rng.uniform(-J / 2.0, J / 2.0)
            Jmat[i, j] = Jij
            H += Jij * embed_pair(X, i, X, j, N)
    for i in range(N):
        H += h * embed(Z, i, N)
    return Reservoir(N=N, H=H, J=Jmat, h=h, seed=seed, topology="fc")


def nn_tfi(N, J=1.0, h=1.0, seed=7, periodic=False):
    """Nearest-neighbour disordered TFI on a chain (Hamhoum). Frozen by ``seed``."""
    rng = np.random.default_rng(seed)
    Jmat = np.zeros((N, N))
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    bonds = [(i, i + 1) for i in range(N - 1)]
    if periodic and N > 2:
        bonds.append((N - 1, 0))
    for i, j in bonds:
        Jij = rng.uniform(-J / 2.0, J / 2.0)
        Jmat[min(i, j), max(i, j)] = Jij
        H += Jij * embed_pair(X, i, X, j, N)
    for i in range(N):
        H += h * embed(Z, i, N)
    return Reservoir(N=N, H=H, J=Jmat, h=h, seed=seed, topology="nn")
