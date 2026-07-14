"""qrc_core.py -- Exact NumPy reference for the windowed restart QRC.

Simulates, by exact statevector evolution, the same circuit family the
Qiskit path (code/qrc_qiskit.py) will run:

    |0...0>  --[ for each u_k in window:  U_enc(u_k) then W ]-->  |psi>
    features = <Z_i>, <X_i>, <Z_i Z_j>   (2n + n(n-1)/2 = 20 for n = 5)

U_enc(u) = tensor_i RY(gamma * g_i * u)   (g_i fixed random gains)
W        = CZ ring, then fixed random RY/RZ on each qubit (frozen entangler)

Qubit 0 is the LEFTMOST tensor factor throughout. The Qiskit port must
reproduce these features noiselessly at the same seed (validation gate).

Also hosts: ridge read-out (direct lambda or GCV via SVD), NMSE, and the
finite-shot sampling model used by the shot-noise anchor.
"""

import numpy as np

# ---------------------------------------------------------------- config
N_QUBITS = 5
GAMMA = np.pi / 2      # encoding gain; sweep {pi/4, pi/2, pi} in experiments
SEED = 7


def print_config():
    print(f"qrc_core config: n={N_QUBITS} gamma={GAMMA:.4f} seed={SEED}")


# ---------------------------------------------------------------- gates
I2 = np.eye(2, dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def op_at(gate, i, n):
    """Embed a 1-qubit gate at site i (site 0 = leftmost tensor factor)."""
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, gate if k == i else I2)
    return out


def cp_at(i, j, theta, n):
    """Controlled-phase(theta) between sites i and j as a 2^n diagonal.
    theta = pi is CZ."""
    d = np.ones(2 ** n, dtype=complex)
    for b in range(2 ** n):
        if (b >> (n - 1 - i)) & 1 and (b >> (n - 1 - j)) & 1:
            d[b] = np.exp(1j * theta)
    return np.diag(d)


def cz_at(i, j, n):
    """CZ between sites i and j as a 2^n diagonal."""
    return cp_at(i, j, np.pi, n)


# ---------------------------------------------------------------- reservoir
class WindowedReservoir:
    """Frozen random windowed feature map; exact statevector."""

    def __init__(self, n=N_QUBITS, gamma=GAMMA, seed=SEED, ent_scale=1.0):
        """ent_scale s is the digital JDt analogue (handbook 40.1): every
        entangler angle is scaled by s -- CP(s*pi) ring (s=1 -> CZ exactly)
        then RY(s*a_i), RZ(s*b_i). s -> 0 is the near-identity
        under-scrambled regime; s > 1 over-scrambles. Sweep log-spaced.

        rng draw ORDER is unchanged vs the s=1 original, so any seed
        reproduces the same gains/angles at every s."""
        self.n, self.gamma, self.ent_scale = n, gamma, ent_scale
        rng = np.random.default_rng(seed)
        self.gains = rng.uniform(0.5, 1.5, size=n)          # g_i
        # frozen entangler W: CP(s*pi) ring then random RY(s*a), RZ(s*b)
        W = np.eye(2 ** n, dtype=complex)
        for i in range(n):
            W = cp_at(i, (i + 1) % n, ent_scale * np.pi, n) @ W
        self.w_angles = rng.uniform(0, 2 * np.pi, size=(n, 2))
        for i in range(n):
            W = op_at(ry(ent_scale * self.w_angles[i, 0]), i, n) @ W
            W = op_at(rz(ent_scale * self.w_angles[i, 1]), i, n) @ W
        self.W = W
        # observables
        self.obs = ([op_at(Z, i, n) for i in range(n)]
                    + [op_at(X, i, n) for i in range(n)]
                    + [op_at(Z, i, n) @ op_at(Z, j, n)
                       for i in range(n) for j in range(i + 1, n)])

    def n_features(self):
        return 2 * self.n + self.n * (self.n - 1) // 2

    def _u_enc(self, u):
        out = np.array([[1.0 + 0j]])
        for i in range(self.n):
            out = np.kron(out, ry(self.gamma * self.gains[i] * u))
        return out

    def state(self, window):
        """Run one restart circuit on a window (oldest input first)."""
        psi = np.zeros(2 ** self.n, dtype=complex)
        psi[0] = 1.0
        for u in window:
            psi = self.W @ (self._u_enc(u) @ psi)
        return psi

    def features(self, window):
        psi = self.state(window)
        return np.array([np.real(psi.conj() @ (O @ psi)) for O in self.obs])

    def feature_matrix(self, series, L):
        """Feature row per step k >= L-1 using series[k-L+1 .. k]."""
        T = len(series)
        Xf = np.zeros((T - L + 1, self.n_features()))
        for k in range(L - 1, T):
            Xf[k - L + 1] = self.features(series[k - L + 1:k + 1])
        return Xf

    def sampled_z_features(self, window, shots, rng):
        """Finite-shot estimate of the <Z_i> block (shot-noise anchor).

        Samples bitstrings from |psi|^2; Z_i outcome is +1/-1 per bit.
        """
        psi = self.state(window)
        p = np.abs(psi) ** 2
        counts = rng.multinomial(shots, p / p.sum())
        n = self.n
        est = np.zeros(n)
        for b, c in enumerate(counts):
            if c == 0:
                continue
            for i in range(n):
                bit = (b >> (n - 1 - i)) & 1
                est[i] += c * (1.0 - 2.0 * bit)
        return est / shots

    def sampled_features(self, window, shots, rng, probs=None):
        """Finite-shot estimate of ALL 20 features, statistically identical
        to the noiseless Qiskit sampled path: one S-shot multinomial in the
        Z basis (gives Z_i and Z_iZ_j) and one in the X basis (H rotation,
        gives X_i). S = shots per basis. Pass probs=(p_z, p_x) to reuse
        precomputed distributions (shots-curve fast path)."""
        n = self.n
        if probs is None:
            psi = self.state(window)
            p_z = np.abs(psi) ** 2
            H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            Hn = np.array([[1.0 + 0j]])
            for _ in range(n):
                Hn = np.kron(Hn, H1)
            p_x = np.abs(Hn @ psi) ** 2
        else:
            p_z, p_x = probs
        bits = np.array([[(b >> (n - 1 - i)) & 1 for i in range(n)]
                         for b in range(2 ** n)])
        signs = 1.0 - 2.0 * bits                      # (2^n, n) of +-1
        cz = rng.multinomial(shots, p_z / p_z.sum()) / shots
        cx = rng.multinomial(shots, p_x / p_x.sum()) / shots
        z = cz @ signs
        x = cx @ signs
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        zz = np.array([cz @ (signs[:, i] * signs[:, j]) for i, j in pairs])
        return np.concatenate([z, x, zz])

    def basis_probs(self, window):
        """(p_z, p_x) for sampled_features' fast path."""
        n = self.n
        psi = self.state(window)
        H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        Hn = np.array([[1.0 + 0j]])
        for _ in range(n):
            Hn = np.kron(Hn, H1)
        return np.abs(psi) ** 2, np.abs(Hn @ psi) ** 2


# ---------------------------------------------------------------- read-out
def ridge_fit(Xd, y, lam):
    """w = (X'X + lam I)^-1 X'y. Caller appends the bias column."""
    k = Xd.shape[1]
    return np.linalg.solve(Xd.T @ Xd + lam * np.eye(k), Xd.T @ y)


def ridge_gcv(Xd, y, lams):
    """Pick lambda by generalised cross-validation via SVD."""
    U, s, Vt = np.linalg.svd(Xd, full_matrices=False)
    Uty = U.T @ y
    T = len(y)
    best = (np.inf, None)
    for lam in lams:
        f = s ** 2 / (s ** 2 + lam)          # shrinkage factors
        resid = y - U @ (f * Uty)
        df = f.sum()
        score = (resid @ resid / T) / (1 - df / T) ** 2
        if score < best[0]:
            best = (score, lam)
    lam = best[1]
    w = Vt.T @ ((s / (s ** 2 + lam)) * Uty)
    return w, lam


def add_bias(Xf):
    return np.hstack([Xf, np.ones((len(Xf), 1))])


def nmse(y_true, y_pred, y_train_mean):
    """NMSE vs the train-mean predictor: mean predictor scores exactly 1."""
    denom = np.sum((y_true - y_train_mean) ** 2)
    return np.sum((y_true - y_pred) ** 2) / denom


if __name__ == "__main__":
    print_config()
    r = WindowedReservoir()
    print("n_features =", r.n_features())
    print("features(ramp window) =", np.round(r.features(np.linspace(0, 1, 24)), 4))
