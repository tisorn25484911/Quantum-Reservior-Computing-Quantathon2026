"""qrc_core.py -- Reference implementation of a gate-model quantum reservoir.

Implements the Fujii--Nakajima quantum reservoir protocol in exact
density-matrix form:

    rho_{k+1} = U [ rho_in(u_k) (x) Tr_1(rho_k) ] U^dagger,

with temporal multiplexing (V virtual nodes per input interval), local
<Z_i> and two-point <Z_i Z_j> read-out features, a ridge-regression
read-out, an echo-state-network (ESN) baseline of matched feature count,
a binomial shot-noise emulator, and variance-normalised metrics.

Every component is validated against a known exact answer in
`run_validation_suite()`; run this module directly to execute the suite.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# Pauli operators and tensor-product helpers
# ----------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_all(ops):
    """Tensor product of a list of 2x2 operators (qubit 0 = leftmost factor)."""
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def local_op(op, site, n):
    """Embed a single-qubit operator `op` at `site` in an n-qubit register."""
    return kron_all([op if q == site else I2 for q in range(n)])


def two_site_op(op_a, a, op_b, b, n):
    ops = [I2] * n
    ops[a], ops[b] = op_a, op_b
    return kron_all(ops)


# ----------------------------------------------------------------------
# Reservoir Hamiltonian (disordered fully connected transverse-field Ising)
# ----------------------------------------------------------------------
def ising_hamiltonian(n, J=1.0, h=1.0, rng=None):
    """H = sum_{i<j} J_ij X_i X_j + (h/2) sum_i Z_i,  J_ij ~ U[-J/2, J/2].

    The fully connected random-coupling transverse-field Ising model used in
    the founding proposal (Fujii & Nakajima, PRApplied 8, 024030, 2017).
    """
    rng = np.random.default_rng(rng)
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            H += rng.uniform(-J / 2, J / 2) * two_site_op(X, i, X, j, n)
        H += 0.5 * h * local_op(Z, i, n)
    return H


def xxz_hx_hamiltonian(n, J=1.0, h=1.0, delta=1.0, periodic=False, rng=None):
    """H = sum_<i,j> J (X_i X_j + Y_i Y_j + delta Z_i Z_j) + h sum_i X_i.

    Nearest-neighbour XXZ chain in a transverse field, mirroring
    `Quantathon_stack/Hamiltonian_QRC/Hamiltonians.py::XXZ_hx`.

    Parameters
    ----------
    n : int
        Number of qubits (chain sites).
    J : float
        In-plane (XX + YY) exchange; the ZZ coupling is ``J * delta``.
    h : float
        Uniform transverse field along X (the ``hx`` knob).
    delta : float
        ZZ anisotropy. ``delta == 1`` is the isotropic Heisenberg point,
        ``delta == 0`` the XX model.
    periodic : bool
        Add the bond closing the ring (needs ``n > 2``).
    rng : ignored
        Accepted for a signature compatible with `ising_hamiltonian`; this
        Hamiltonian is deterministic (no disorder).
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    bonds = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        bonds.append((n - 1, 0))
    for i, j in bonds:
        H += J * two_site_op(X, i, X, j, n)
        H += J * two_site_op(Y, i, Y, j, n)
        H += J * delta * two_site_op(Z, i, Z, j, n)
    for i in range(n):
        H += h * local_op(X, i, n)
    return H


# Registry of selectable reservoir Hamiltonians. Each builder has the signature
# (n, J, h, rng=...) so `QuantumReservoir` can construct any of them uniformly.
HAMILTONIANS = {
    "ising": ising_hamiltonian,
    "xxz_hx": xxz_hx_hamiltonian,
}


def propagator(H, dt):
    """U = exp(-i H dt) via Hermitian eigendecomposition."""
    evals, evecs = np.linalg.eigh(H)
    return (evecs * np.exp(-1j * evals * dt)) @ evecs.conj().T


# ----------------------------------------------------------------------
# State injection (partial trace over the input qubit + re-preparation)
# ----------------------------------------------------------------------
def input_state(u):
    """rho_in(u) = |psi_u><psi_u|, |psi_u> = sqrt(1-u)|0> + sqrt(u)|1>, u in [0,1]."""
    a, b = np.sqrt(1.0 - u), np.sqrt(u)
    psi = np.array([a, b], dtype=complex)
    return np.outer(psi, psi.conj())


def trace_out_first(rho, n):
    """Partial trace over qubit 0 of an n-qubit density matrix."""
    d_rest = 2 ** (n - 1)
    r = rho.reshape(2, d_rest, 2, d_rest)
    return np.einsum("ajak->jk", r)


def inject(rho, u, n):
    """rho -> rho_in(u) (x) Tr_0(rho).

    The input qubit is site 0 (the leftmost tensor factor), so re-insertion
    lands at the correct slot with no index permutation required.
    """
    return np.kron(input_state(u), trace_out_first(rho, n))


# ----------------------------------------------------------------------
# The quantum reservoir
# ----------------------------------------------------------------------
class QuantumReservoir:
    """Exact density-matrix simulation of the Fujii--Nakajima protocol."""

    def __init__(self, n_qubits=5, J=1.0, h=1.0, dt=2.0, virtual_nodes=4,
                 use_zz=True, seed=7, hamiltonian="ising"):
        self.n = n_qubits
        self.V = virtual_nodes
        self.use_zz = use_zz
        self.hamiltonian = hamiltonian
        try:
            build_H = HAMILTONIANS[hamiltonian]
        except KeyError:
            raise ValueError(
                f"unknown hamiltonian {hamiltonian!r}; "
                f"choose one of {sorted(HAMILTONIANS)}"
            )
        # Every builder takes the same (n, J, h, rng) contract; `h` is the
        # transverse field (`hx`) for the XXZ choice.
        self.H = build_H(n_qubits, J=J, h=h, rng=seed)
        self.U_sub = propagator(self.H, dt / virtual_nodes)
        # Observables: local Z at each site (read at each virtual node) and,
        # optionally, all two-point ZZ correlators (read at the last node).
        self.Z_ops = [local_op(Z, q, self.n) for q in range(self.n)]
        self.ZZ_ops = ([two_site_op(Z, a, Z, b, self.n)
                        for a in range(self.n) for b in range(a + 1, self.n)]
                       if use_zz else [])
        self.n_features = self.n * self.V + len(self.ZZ_ops)

    def initial_state(self):
        dim = 2 ** self.n
        return np.eye(dim, dtype=complex) / dim  # maximally mixed start

    def run(self, inputs):
        """Drive the reservoir with `inputs` (values in [0, 1]).

        Returns the feature matrix of shape (T, n_features): exact
        expectation values <Z_i> at each of the V virtual nodes, plus
        <Z_i Z_j> at the final node of each interval.
        """
        rho = self.initial_state()
        feats = np.empty((len(inputs), self.n_features))
        for k, u in enumerate(inputs):
            rho = inject(rho, float(u), self.n)
            row = []
            for _ in range(self.V):
                rho = self.U_sub @ rho @ self.U_sub.conj().T
                row.extend(np.real(np.trace(op @ rho)) for op in self.Z_ops)
            row.extend(np.real(np.trace(op @ rho)) for op in self.ZZ_ops)
            feats[k] = row
        return feats


# ----------------------------------------------------------------------
# Classical baseline: leaky echo-state network of matched feature count
# ----------------------------------------------------------------------
class ESN:
    def __init__(self, n_nodes, spectral_radius=0.9, input_scale=1.0,
                 leak=0.3, seed=11):
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(n_nodes, n_nodes))
        W *= spectral_radius / max(abs(np.linalg.eigvals(W)))
        self.W = W
        self.W_in = rng.uniform(-input_scale, input_scale, size=(n_nodes, 1))
        self.leak = leak
        self.n_features = n_nodes

    def run(self, inputs):
        x = np.zeros(self.W.shape[0])
        feats = np.empty((len(inputs), self.n_features))
        for k, u in enumerate(inputs):
            pre = self.W @ x + self.W_in[:, 0] * float(u)
            x = (1 - self.leak) * x + self.leak * np.tanh(pre)
            feats[k] = x
        return feats


# ----------------------------------------------------------------------
# Read-out, shot noise, metrics
# ----------------------------------------------------------------------
def ridge_fit(features, targets, lam=1e-6):
    """Ridge regression with intercept: w = (X^T X + lam I)^{-1} X^T y."""
    Xd = np.hstack([features, np.ones((len(features), 1))])
    A = Xd.T @ Xd + lam * np.eye(Xd.shape[1])
    return np.linalg.solve(A, Xd.T @ targets)


def ridge_predict(features, w):
    Xd = np.hstack([features, np.ones((len(features), 1))])
    return Xd @ w


def add_shot_noise(features, shots, rng=None):
    """Emulate S-shot estimation of z = <Z> in [-1, 1]:
    p = (1+z)/2, sigma(z) = 2 sqrt(p(1-p)/S); z -> clip(z + xi*sigma, -1, 1)."""
    rng = np.random.default_rng(rng)
    z = np.clip(features, -1.0, 1.0)
    p = (1.0 + z) / 2.0
    sigma = 2.0 * np.sqrt(np.clip(p * (1 - p), 0.0, None) / shots)
    return np.clip(z + rng.standard_normal(z.shape) * sigma, -1.0, 1.0)


def nmse(y_true, y_pred):
    """Variance-normalised MSE. A value >= 1 is worse than the mean predictor."""
    return float(np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def train_eval(features, targets, washout=100, train_frac=0.7, lam=1e-6):
    """Washout -> chronological train/test split -> ridge -> test NMSE."""
    Xf, y = features[washout:], targets[washout:]
    n_tr = int(train_frac * len(y))
    w = ridge_fit(Xf[:n_tr], y[:n_tr], lam=lam)
    pred = ridge_predict(Xf[n_tr:], w)
    return nmse(y[n_tr:], pred), pred, y[n_tr:]


# ----------------------------------------------------------------------
# Benchmarks and diagnostics
# ----------------------------------------------------------------------
def narma10(T, seed=0):
    """NARMA-10: u_k ~ U[0, 0.5];
    y_{k+1} = 0.3 y_k + 0.05 y_k sum_{i=0}^{9} y_{k-i} + 1.5 u_{k-9} u_k + 0.1."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=T)
    y = np.zeros(T)
    for k in range(9, T - 1):
        y[k + 1] = (0.3 * y[k] + 0.05 * y[k] * np.sum(y[k - 9:k + 1])
                    + 1.5 * u[k - 9] * u[k] + 0.1)
    return u, y


def memory_function(run_fn, n_features, T=2200, d_max=25, washout=200,
                    lam=1e-6, seed=3):
    """Linear memory function MF_d = corr^2(y_hat_d, u_{t-d}) on i.i.d. input;
    memory capacity MC = sum_d MF_d (Jaeger 2001)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=T)
    feats = run_fn(u)
    mf = np.zeros(d_max + 1)
    for d in range(1, d_max + 1):
        target = np.roll(u, d)
        Xf, y = feats[washout:], target[washout:]
        n_tr = int(0.7 * len(y))
        w = ridge_fit(Xf[:n_tr], y[:n_tr], lam=lam)
        pred = ridge_predict(Xf[n_tr:], w)
        c = np.corrcoef(pred, y[n_tr:])[0, 1]
        mf[d] = max(c, 0.0) ** 2
    return mf, float(mf.sum())


# ----------------------------------------------------------------------
# Validation suite (anchors with known exact answers)
# ----------------------------------------------------------------------
def run_validation_suite():
    n = 3
    # 1. Injection anchor: rebuild a known product state exactly.
    rho = kron_all([input_state(0.0), input_state(1.0), input_state(0.5)])
    rho2 = inject(rho, 0.5, n)
    zs = [np.real(np.trace(local_op(Z, q, n) @ rho2)) for q in range(n)]
    assert np.allclose(zs, [0.0, -1.0, 0.0], atol=1e-12), zs
    assert abs(np.trace(rho2) - 1.0) < 1e-12
    # 2. Propagator unitarity and trace preservation through a full step.
    H = ising_hamiltonian(4, rng=0)
    U = propagator(H, 1.7)
    assert np.allclose(U @ U.conj().T, np.eye(16), atol=1e-10)
    res = QuantumReservoir(n_qubits=4, seed=0)
    r = res.initial_state()
    for u in [0.2, 0.9, 0.4]:
        r = inject(r, u, 4)
        r = res.U_sub @ r @ res.U_sub.conj().T
    assert abs(np.trace(r) - 1.0) < 1e-10
    assert np.all(np.linalg.eigvalsh(r) > -1e-10)          # positivity
    # 2b. XXZ+hx choice: Hermitian Hamiltonian, and a reservoir built on it
    #     stays a valid (trace-1, positive) density matrix through a full step.
    Hx = xxz_hx_hamiltonian(4, J=0.8, h=0.5, delta=1.0)
    assert np.allclose(Hx, Hx.conj().T, atol=1e-12)
    res_x = QuantumReservoir(n_qubits=4, J=0.8, h=0.5, seed=0,
                             hamiltonian="xxz_hx")
    rx = res_x.initial_state()
    for u in [0.2, 0.9, 0.4]:
        rx = inject(rx, u, 4)
        rx = res_x.U_sub @ rx @ res_x.U_sub.conj().T
    assert abs(np.trace(rx) - 1.0) < 1e-10
    assert np.all(np.linalg.eigvalsh(rx) > -1e-10)
    # 3. Ridge anchor: exact recovery of a known affine map.
    rng = np.random.default_rng(1)
    Xf = rng.normal(size=(300, 6))
    w_true, b_true = rng.normal(size=6), 0.7
    y = Xf @ w_true + b_true
    w = ridge_fit(Xf, y, lam=1e-12)
    assert np.allclose(w[:-1], w_true, atol=1e-6) and abs(w[-1] - b_true) < 1e-6
    # 4. Metric anchors: perfect predictor -> 0; mean predictor -> 1.
    yt = rng.normal(size=500)
    assert nmse(yt, yt) < 1e-24
    assert abs(nmse(yt, np.full_like(yt, yt.mean())) - 1.0) < 1e-12
    # 5. Shot-noise emulator: error scales as S^{-1/2}.
    z = np.full((4000,), 0.3)
    e1 = np.std(add_shot_noise(z, 100, rng=0) - z)
    e2 = np.std(add_shot_noise(z, 400, rng=0) - z)
    assert 1.6 < e1 / e2 < 2.4, (e1, e2)
    print("All validation anchors passed.")


if __name__ == "__main__":
    run_validation_suite()
