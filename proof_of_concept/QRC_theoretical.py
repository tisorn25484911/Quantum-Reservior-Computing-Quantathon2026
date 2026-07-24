"""QRC_theoretical.py -- exact NumPy statevector reference for the QRC.

This is the *theoretical* (classically exact) engine of the proof of
concept. It simulates, by exact statevector evolution, the same gate-model
quantum reservoir the Qiskit path (``QRC_QPU_implementation.py``) runs on a
statevector / shot-based simulator. Because the two are the same circuit
family at the same seed, the Qiskit port must reproduce these features to
~1e-10 -- that agreement is the *validation gate* the whole pipeline rests on.

Two reservoirs live here:

* ``WindowedReservoir``  -- univariate, one scalar input per step. This is
  the class the Qiskit twin imports and validates against. Per window step,
  oldest input first::

        |0...0>  --[ for each u_k:  U_enc(u_k) then W ]-->  |psi>
        U_enc(u) = (x)_i RY(gamma * g_i * u)        (g_i frozen random gains)
        W        = CP(s*pi) ring, then frozen RY(s*a_i), RZ(s*b_i)   (s = ent_scale)
        features = <Z_i>, <X_i>, <Z_i Z_j>          (2n + n(n-1)/2 = 20 at n=5)

* ``MultiChannelReservoir`` -- the multivariate extension used for the
  *compound-event* deliverable. Several drivers (e.g. sea-surface
  temperature, the Southern Oscillation Index, the Pacific Decadal
  Oscillation) are encoded onto DIFFERENT rotation axes of EVERY qubit
  before the shared entangler mixes them::

        U_enc(u_vec) = (x)_i  RZ(g2_i u2) RX(g1_i u1) RY(g0_i u0)
                                    channel 2   channel 1   channel 0

  Because each qubit accumulates all channels *before* W entangles the
  register, every read-out feature (including <Z_i Z_j>) is a nonlinear
  joint function of all drivers' histories. That is the substrate for the
  inter-driver (and tail) dependence a *bank* of one-channel reservoirs
  cannot represent -- the mechanism the compound-event experiment tests.
  With one channel it reduces EXACTLY to ``WindowedReservoir`` (same seed).

Conventions
-----------
* Qubit 0 is the LEFTMOST tensor factor throughout (big-endian). The Qiskit
  port reverses to little-endian in exactly one place per pathway.
* Rotation-angle sign convention: RY(t) = exp(-i t Y / 2), etc. (Qiskit's).

Run this module directly to execute the validation suite.

Protocol lineage (audit 2026-07). The founding proposal (Fujii & Nakajima,
"Harnessing disordered-ensemble quantum dynamics for machine learning",
Phys. Rev. Applied 8, 024030, 2017) keeps ONE persistent reservoir state
across the whole series, injects each input by partial trace, and reads
ensemble expectation values online. The reservoirs here instead RESTART from
|0..0> and re-run only the last L inputs per output step -- this is the
"rewinding protocol" of Mujal, Martinez-Pena, Giorgi, Soriano & Zambrini,
npj Quantum Information 9, 16 (2023), the standard way to run QRC on
gate-based hardware where expectation values come from repeated projective
shots and no state survives measurement. The trade-off is quantified in the
notebook: the rewinding reservoir retains far less linear input memory than
the persistent-state one (memory-capacity plot), in exchange for shallow,
QPU-runnable circuits. The Fujii--Nakajima-faithful persistent-state
implementation lives in Quantathon_stack/Hamiltonian_QRC/qrc_core.py and is
validated against the paper's own NARMA benchmark in the notebook appendix.
As in both papers, ALL reservoir parameters (gains, entangler angles) are
drawn randomly once per seed and frozen; only the linear read-out is trained.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- config
N_QUBITS = 5
GAMMA = np.pi / 2      # encoding gain; sweep {pi/4, pi/2, pi} in experiments
SEED = 7


def print_config():
    print(f"QRC_theoretical config: n={N_QUBITS} gamma={GAMMA:.4f} seed={SEED}")


# ---------------------------------------------------------------- gates
I2 = np.eye(2, dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


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


def _build_entangler(n, w_angles, ent_scale):
    """Frozen entangler W: CP(s*pi) ring, then RY(s*a_i), RZ(s*b_i)."""
    W = np.eye(2 ** n, dtype=complex)
    for i in range(n):
        W = cp_at(i, (i + 1) % n, ent_scale * np.pi, n) @ W
    for i in range(n):
        W = op_at(ry(ent_scale * w_angles[i, 0]), i, n) @ W
        W = op_at(rz(ent_scale * w_angles[i, 1]), i, n) @ W
    return W


def _build_observables(n):
    """The 2n + n(n-1)/2 read-out operators: <Z_i>, <X_i>, <Z_i Z_j>."""
    return ([op_at(Z, i, n) for i in range(n)]
            + [op_at(X, i, n) for i in range(n)]
            + [op_at(Z, i, n) @ op_at(Z, j, n)
               for i in range(n) for j in range(i + 1, n)])


# ---------------------------------------------------------------- univariate reservoir
class WindowedReservoir:
    """Frozen random windowed feature map, one scalar input per step.

    This is the exact statevector twin the Qiskit backend must match.
    """

    def __init__(self, n=N_QUBITS, gamma=GAMMA, seed=SEED, ent_scale=1.0):
        """ent_scale s is the digital ``J*dt`` analogue (handbook 40.1): every
        entangler angle is scaled by s -- CP(s*pi) ring (s=1 -> CZ exactly)
        then RY(s*a_i), RZ(s*b_i). s -> 0 is the near-identity, under-scrambled
        regime; s > 1 over-scrambles. Sweep log-spaced.

        The rng draw ORDER (gains, then entangler angles) is fixed so any seed
        reproduces the same gains/angles at every ent_scale.
        """
        self.n, self.gamma, self.ent_scale = n, gamma, ent_scale
        rng = np.random.default_rng(seed)
        self.gains = rng.uniform(0.5, 1.5, size=n)             # g_i
        self.w_angles = rng.uniform(0, 2 * np.pi, size=(n, 2))  # a_i, b_i
        self.W = _build_entangler(n, self.w_angles, ent_scale)
        self.obs = _build_observables(n)

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

    def sampled_features(self, window, shots, rng, probs=None):
        """Finite-shot estimate of ALL features, statistically identical to
        the noiseless Qiskit sampled path: one S-shot multinomial in the Z
        basis (gives Z_i and Z_iZ_j) and one in the X basis (H rotation, gives
        X_i). S = shots per basis. Pass probs=(p_z, p_x) to reuse precomputed
        distributions."""
        n = self.n
        if probs is None:
            p_z, p_x = self.basis_probs(window)
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
        """(p_z, p_x) computational-basis probabilities for the two bases."""
        n = self.n
        psi = self.state(window)
        H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        Hn = np.array([[1.0 + 0j]])
        for _ in range(n):
            Hn = np.kron(Hn, H1)
        return np.abs(psi) ** 2, np.abs(Hn @ psi) ** 2


# ---------------------------------------------------------------- multivariate reservoir
# Channel c is encoded on rotation axis AXES[c]. Channel 0 uses RY, so a
# one-channel MultiChannelReservoir is identical to a WindowedReservoir at the
# same seed. RY, RX, RZ do not commute, so up to three drivers stay
# distinguishable on each qubit (RY(a) then RY(b) would merely add to RY(a+b)).
AXES = [ry, rx, rz]


class MultiChannelReservoir:
    """Windowed restart reservoir with several input channels per step.

    ``series`` and ``window`` are 2-D: shape (T, C) and (L, C), one column per
    driver. All channels share the frozen entangler W and the 20 observables;
    only the encoding differs (one rotation axis per channel).
    """

    def __init__(self, n=N_QUBITS, gamma=GAMMA, seed=SEED, ent_scale=1.0,
                 n_channels=2):
        if n_channels > len(AXES):
            raise ValueError(f"n_channels={n_channels} exceeds the {len(AXES)} "
                             f"available rotation axes {['RY', 'RX', 'RZ']}")
        self.n, self.gamma, self.ent_scale = n, gamma, ent_scale
        self.C = n_channels
        rng = np.random.default_rng(seed)
        # gains[c] are channel c's per-qubit frozen gains. Drawing (C, n) with
        # C=1 draws the SAME n numbers as WindowedReservoir's size-n draw, so
        # the one-channel reduction is seed-exact.
        self.gains = rng.uniform(0.5, 1.5, size=(n_channels, n))
        self.w_angles = rng.uniform(0, 2 * np.pi, size=(n, 2))
        self.W = _build_entangler(n, self.w_angles, ent_scale)
        self.obs = _build_observables(n)

    def n_features(self):
        return 2 * self.n + self.n * (self.n - 1) // 2

    def _u_enc(self, u_vec):
        """(x)_i [ RZ(g2_i u2) RX(g1_i u1) RY(g0_i u0) ]. RY is applied first
        (innermost) so channel 0 creates the superposition the later axes act
        on, and the C=1 case is exactly the univariate RY encoding."""
        out = np.array([[1.0 + 0j]])
        for i in range(self.n):
            gate_i = I2
            for c in range(self.C):          # c=0 innermost (applied first)
                gate_i = AXES[c](self.gamma * self.gains[c, i]
                                 * u_vec[c]) @ gate_i
            out = np.kron(out, gate_i)
        return out

    def state(self, window):
        """window: shape (L, C), oldest input first."""
        window = np.atleast_2d(window)
        psi = np.zeros(2 ** self.n, dtype=complex)
        psi[0] = 1.0
        for u_vec in window:
            psi = self.W @ (self._u_enc(u_vec) @ psi)
        return psi

    def features(self, window):
        psi = self.state(window)
        return np.array([np.real(psi.conj() @ (O @ psi)) for O in self.obs])

    def feature_matrix(self, series, L):
        """series: shape (T, C). Row per step k >= L-1 from series[k-L+1 .. k]."""
        series = np.atleast_2d(series)
        if series.shape[1] != self.C and series.shape[0] == self.C:
            series = series.T
        T = len(series)
        Xf = np.zeros((T - L + 1, self.n_features()))
        for k in range(L - 1, T):
            Xf[k - L + 1] = self.features(series[k - L + 1:k + 1])
        return Xf


# ---------------------------------------------------------------- read-out
def add_bias(Xf):
    return np.hstack([Xf, np.ones((len(Xf), 1))])


def ridge_fit(Xd, y, lam):
    """w = (X'X + lam I)^-1 X'y. Caller appends the bias column."""
    k = Xd.shape[1]
    return np.linalg.solve(Xd.T @ Xd + lam * np.eye(k), Xd.T @ y)


def ridge_gcv(Xd, y, lams):
    """Pick lambda by generalised cross-validation via SVD (leak-free: uses
    only the rows passed in, which the caller restricts to the train span)."""
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


def nmse(y_true, y_pred, y_train_mean):
    """NMSE vs the train-mean predictor: the mean predictor scores exactly 1.

    Convention note: Fujii & Nakajima Eq. (A1) normalise by sum(ybar^2), which
    reads lower whenever the target mean is non-zero. Ours is the stricter
    convention; the two scales are not directly comparable."""
    denom = np.sum((y_true - y_train_mean) ** 2)
    return np.sum((y_true - y_pred) ** 2) / denom


# ---------------------------------------------------------------- theoretical diagnostics
# Quantities computed DIRECTLY from the reservoir's matrices (entangler W,
# observables, and the states they produce) -- no fitted detector, no task
# data. They characterise the reservoir itself, so they can be swept over
# (gamma, ent_scale) to see the operating-point landscape from first principles.

def entangler_scrambling(W):
    """Circular variance of the frozen entangler's eigenphases, in [0, 1].

    W is unitary, so its eigenvalues sit on the unit circle at phases {phi_k}.
    We return 1 - |mean(e^{i phi_k})|: 0 means the phases are aligned (a
    near-identity, under-scrambling entangler) and values toward 1 mean the
    phases are spread around the circle (strong scrambling). Pure linear
    algebra on the matrix -- no input data enters."""
    ev = np.linalg.eigvals(W)
    return float(1.0 - np.abs(np.mean(ev)))


def _memory_of_target(Xd, target, n_train, lam):
    """Squared out-of-sample correlation of the optimal linear reconstruction
    of `target` from features `Xd` (bias already appended). The building block
    of memory capacity; the linear read-out is the closed-form projection, so
    this is a property of the reservoir, not of a trained model."""
    w = ridge_fit(Xd[:n_train], target[:n_train], lam)
    pred, yt = Xd[n_train:] @ w, target[n_train:]
    if np.std(pred) < 1e-12 or np.std(yt) < 1e-12:
        return 0.0
    c = np.corrcoef(pred, yt)[0, 1]
    return 0.0 if np.isnan(c) else max(c, 0.0) ** 2


def linear_memory_capacity(reservoir, L, T=1500, d_max=None, washout=100,
                           seed=0, lam=1e-6, train_frac=0.7):
    """Jaeger (2001) short-term linear memory capacity of a windowed reservoir,
    computed from its own feature statistics on i.i.d. input:

        MC = sum_{channels c} sum_{delays d} corr^2( best linear estimate of
                                                     u_c[k-d] from features_k,
                                                     u_c[k-d] )

    MC quantifies how much of the recent multi-channel input history the 20-D
    feature map linearly retains -- a matrix-level expressivity/memory measure
    independent of any downstream task. Works for WindowedReservoir (1 channel)
    and MultiChannelReservoir (C channels); returns (MC, per_delay_curve) where
    the curve is averaged over channels."""
    C = getattr(reservoir, "C", 1)
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=T if C == 1 else (T, C))
    Xf = reservoir.feature_matrix(u, L)               # rows k = L-1 .. T-1
    ks = np.arange(L - 1, T)
    if d_max is None:
        d_max = L - 1
    d_max = min(d_max, L - 1)

    keep = np.arange(washout, len(Xf))
    Xk, kk = Xf[keep], ks[keep]
    n_train = int(train_frac * len(keep))
    mu, sd = Xk[:n_train].mean(0), Xk[:n_train].std(0)
    sd[sd == 0] = 1.0
    Xd = add_bias((Xk - mu) / sd)

    per_delay = np.zeros(d_max + 1)
    mc_total = 0.0
    for d in range(1, d_max + 1):
        chans = [u[kk - d]] if C == 1 else [u[kk - d, c] for c in range(C)]
        for tgt in chans:
            r2 = _memory_of_target(Xd, tgt, n_train, lam)
            per_delay[d] += r2 / C
            mc_total += r2
    return mc_total, per_delay


# ---------------------------------------------------------------- validation suite
def run_validation_suite():
    """Anchors with known exact answers -- run on `python QRC_theoretical.py`."""
    # 1. Rotation gates are unitary and satisfy RY(a)RY(b) = RY(a+b).
    for g in (rx, ry, rz):
        U = g(0.7)
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-12), g.__name__
    assert np.allclose(ry(0.3) @ ry(0.4), ry(0.7), atol=1e-12)

    # 2. Reservoir state is normalised; feature values lie in [-1, 1].
    res = WindowedReservoir(seed=0)
    w = np.linspace(0, 1, 24)
    psi = res.state(w)
    assert abs(np.vdot(psi, psi) - 1.0) < 1e-10
    f = res.features(w)
    assert f.shape == (res.n_features(),) == (20,)
    assert np.all(np.abs(f) <= 1.0 + 1e-9)

    # 3. One-channel MultiChannelReservoir == WindowedReservoir (seed-exact).
    mc1 = MultiChannelReservoir(seed=0, n_channels=1)
    assert np.allclose(mc1.gains[0], res.gains, atol=1e-12)
    fm = mc1.features(w.reshape(-1, 1))
    assert np.allclose(fm, f, atol=1e-12), np.max(np.abs(fm - f))

    # 4. Two channels genuinely interact: swapping the two drivers changes the
    #    features (a bank of independent reservoirs would be permutation-blind
    #    only per channel; here the shared entangler couples them).
    mc2 = MultiChannelReservoir(seed=0, n_channels=2)
    rng = np.random.default_rng(3)
    win = rng.uniform(0, 1, size=(24, 2))
    fa = mc2.features(win)
    fb = mc2.features(win[:, ::-1])          # swap channel 0 <-> 1
    assert not np.allclose(fa, fb, atol=1e-6), "channels do not interact"

    # 5. Ridge anchor: exact recovery of a known affine map at tiny lambda.
    Xf = rng.normal(size=(300, 6))
    w_true, b_true = rng.normal(size=6), 0.7
    y = Xf @ w_true + b_true
    wgcv, _ = ridge_gcv(add_bias(Xf), y, np.array([1e-10]))
    assert np.allclose(wgcv[:-1], w_true, atol=1e-5) and abs(wgcv[-1] - b_true) < 1e-5

    # 6. Metric anchors: perfect predictor -> 0; train-mean predictor -> 1.
    yt = rng.normal(size=500)
    assert nmse(yt, yt, yt.mean()) < 1e-24
    assert abs(nmse(yt, np.full_like(yt, yt.mean()), yt.mean()) - 1.0) < 1e-12

    print("All validation anchors passed.")


if __name__ == "__main__":
    print_config()
    r = WindowedReservoir()
    print("n_features =", r.n_features())
    print("features(ramp) =", np.round(r.features(np.linspace(0, 1, 24)), 4))
    run_validation_suite()
