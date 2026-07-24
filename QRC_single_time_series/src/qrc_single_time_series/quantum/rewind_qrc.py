"""rewind_qrc.py -- single-channel Hamhoum rewind QRC + Stinespring exact mode.

The rewind reservoir (Hamhoum et al., Algorithm 1) is STATELESS across prediction
steps: to predict the value after step t it re-initialises to |+>^{(x)N}, then
re-encodes the last ``t_w`` inputs (the rewind window) one at a time -- each input
is injected by resetting the injection qubit (site 0, G2) and re-preparing it as
Ry(2 arcsin sqrt(u))|0>, followed by one NN-TFI (or Opt-NN) first-order (kappa=1)
Trotter block. Local Z is read after every injection; the collected expectations
are the window's features, mapped by a ridge readout to the next value (next-step
target, Hamhoum Eq. 7).

Two exact evaluation modes that agree by construction:
  - ``window_features`` (density matrix): the (t_w - 1) injection resets make the
    reservoir mixed, so an N-qubit density matrix is evolved. Exact, N <= 10.
  - ``window_features_dilated`` (Stinespring): each reset is dilated to a fresh
    |0> ancilla + SWAP, giving a PURE statevector on N + (t_w - 1) qubits whose
    local-Z values equal the density-matrix ones exactly. This is the P3-deferred
    dilation; it unlocks N = 12-15 exactly (memory measured in the handoff).
Implemented in P6.
"""
from __future__ import annotations

import numpy as np

from .input_channels import inject
from .observables import build_observables
from .pauli import PLUS
from .schedule import build_block_schedule
from .exact_trotter_qrc import block_unitary
from .tensor import kron_all


class RewindReservoir:
    """Hamhoum single-channel rewind reservoir (exact: density matrix + dilation)."""

    def __init__(self, reservoir, t_w=10, tau=1.0, kappa=1, order=1,
                 observable_kind="z_local", inject_site=0, ordering=None):
        if inject_site != 0:
            raise NotImplementedError("dilation assumes injection at site 0 (G2)")
        self.res = reservoir
        self.N = reservoir.N
        self.t_w = int(t_w)
        self.tau = float(tau)
        self.inject_site = inject_site
        self.ops, self.obs_labels = build_observables(self.N, observable_kind)
        self.M = len(self.ops)
        self.schedule = build_block_schedule(reservoir, block_time=self.tau,
                                             kappa=kappa, order=order,
                                             ordering=ordering)
        self.U = block_unitary(self.schedule, self.N)      # NN-TFI Trotter block

    # -- initialisation -------------------------------------------------------
    def _plus_density(self):
        return kron_all([PLUS] * self.N)                   # |+><+|^{(x)N}

    @property
    def feature_dim(self):
        return self.M * self.t_w + 1

    # -- density-matrix exact mode -------------------------------------------
    def window_features(self, window):
        """Local-Z features for one rewind window (density matrix, exact)."""
        window = np.asarray(window, dtype=float)
        rho = self._plus_density()
        Ud = self.U.conj().T
        feats = np.empty((self.t_w, self.M))
        pad = self.t_w - len(window)
        for i in range(self.t_w):
            u = window[i - pad] if i >= pad else window[0]
            u = min(1.0, max(0.0, float(u)))          # encoder clips to [0,1]
            rho = inject(rho, u, self.N, site=0)
            rho = self.U @ rho @ Ud
            feats[i] = [np.trace(rho @ op).real for op in self.ops]
        return feats.reshape(-1)

    def features_series(self, inputs, include_bias=True):
        """Sliding-window feature matrix; row t uses inputs[t-t_w+1 .. t]."""
        u = np.asarray(inputs, dtype=float)
        rows = []
        for t in range(self.t_w - 1, len(u)):
            rows.append(self.window_features(u[t - self.t_w + 1: t + 1]))
        X = np.asarray(rows)
        if include_bias:
            X = np.hstack([X, np.ones((len(X), 1))])
        return X

    # -- Stinespring dilated statevector mode --------------------------------
    def n_dilated_qubits(self):
        return self.N + (self.t_w - 1)

    def window_features_dilated(self, window):
        """Same features via a PURE statevector on N + (t_w - 1) qubits (exact).

        Reset of the injection qubit before window step i (>0) is dilated: a fresh
        ancilla (one of the (t_w - 1) spares) is prepared in the injection state and
        SWAPped into site 0; the spent qubit is parked in the ancilla register and
        never touched again, so tracing it out is implicit. Local Z on sites 0..N-1
        therefore equals the density-matrix value exactly.
        """
        window = np.asarray(window, dtype=float)
        nq = self.n_dilated_qubits()
        pad = self.t_w - len(window)

        # statevector, qubit 0 = leftmost factor (G2). Reservoir sites 0..N-1,
        # ancillas N..nq-1. Init reservoir |+>^N (x) ancillas |0>.
        plus = np.array([1.0, 1.0]) / np.sqrt(2.0)
        zero = np.array([1.0, 0.0])
        state = plus
        for _ in range(self.N - 1):
            state = np.kron(state, plus)
        for _ in range(nq - self.N):
            state = np.kron(state, zero)

        U_res = self._embed_reservoir_block(nq)
        feats = np.empty((self.t_w, self.M))
        next_anc = self.N
        for i in range(self.t_w):
            u = window[i - pad] if i >= pad else window[0]
            u = min(1.0, max(0.0, float(u)))
            theta = 2.0 * np.arcsin(np.sqrt(u))
            if i == 0:
                state = self._apply_ry(state, 0, theta, nq)     # site 0 is |+>; encode
            else:
                # dilate reset: encode a fresh ancilla, SWAP it into site 0
                state = self._apply_ry(state, next_anc, theta, nq, from_zero=True)
                state = self._apply_swap(state, 0, next_anc, nq)
                next_anc += 1
            state = U_res @ state
            feats[i] = [self._z_expect(state, s, nq) for s in range(self.M)]
        return feats.reshape(-1)

    # -- dilation helpers (dense; small N in tests) --------------------------
    def _embed_reservoir_block(self, nq):
        U = self.U
        I_anc = np.eye(2 ** (nq - self.N))
        return np.kron(U, I_anc)

    def _single_qubit_gate(self, g, site, nq):
        left = np.eye(2 ** site)
        right = np.eye(2 ** (nq - 1 - site))
        return np.kron(np.kron(left, g), right)

    def _apply_ry(self, state, site, theta, nq, from_zero=False):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        Ry = np.array([[c, -s], [s, c]])
        if from_zero:
            g = Ry                              # ancilla already |0>
        else:
            # site currently |+>; reset-then-encode == project to |0> then Ry is
            # only valid on a fresh qubit. For site 0 at i==0 the reservoir qubit is
            # |+>; Hamhoum's first injection encodes over |+>, matching the DM path
            # which injects (reset+prepare) as well -> use reset+Ry via |0><0|.
            g = Ry
            state = self._reset_to_zero(state, site, nq)
        return self._single_qubit_gate(g, site, nq) @ state

    def _reset_to_zero(self, state, site, nq):
        # deterministic reset is non-unitary; only used for the i==0 injection on a
        # product |+> qubit where it is a valid projection+renorm.
        P0 = self._single_qubit_gate(np.array([[1.0, 0], [0, 0]]), site, nq)
        P1 = self._single_qubit_gate(np.array([[0, 0], [0, 1.0]]), site, nq)
        X = self._single_qubit_gate(np.array([[0, 1.0], [1.0, 0]]), site, nq)
        new = P0 @ state + X @ (P1 @ state)     # flip the |1> branch to |0>
        return new / np.linalg.norm(new)

    def _apply_swap(self, state, a, b, nq):
        d = 2 ** nq
        idx = np.arange(d)
        ba = (idx >> (nq - 1 - a)) & 1
        bb = (idx >> (nq - 1 - b)) & 1
        swap = idx ^ (((ba ^ bb) << (nq - 1 - a)) | ((ba ^ bb) << (nq - 1 - b)))
        return state[swap]

    def _z_expect(self, state, site, nq):
        d = 2 ** nq
        idx = np.arange(d)
        bit = (idx >> (nq - 1 - site)) & 1
        signs = 1.0 - 2.0 * bit
        return float(np.sum((np.abs(state) ** 2) * signs))


def copy_baseline_mse(inputs):
    """Trivial copy (persistence) baseline: predict u_t for u_{t+1}."""
    u = np.asarray(inputs, dtype=float)
    return float(np.mean((u[1:] - u[:-1]) ** 2))
