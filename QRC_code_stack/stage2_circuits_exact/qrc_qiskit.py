"""qrc_qiskit.py -- Qiskit port of the windowed restart reservoir.

Runs the SAME circuit family as the exact NumPy reference (qrc_core.py),
reusing the gains g_i, encoding gain gamma, and frozen-entangler angles
FROM a WindowedReservoir instance, so identical seeds give identical
circuits. Per window step (circuit time order, matching qrc_core.state):

    RY(gamma * g_i * u_k) on each qubit          (encoding)
    CZ ring (0,1)(1,2)(2,3)(3,4)(4,0)            (entangler part 1)
    RY(a_i) then RZ(b_i) on each qubit           (entangler part 2)

Three evaluation modes:
    exact    Statevector, no shots, no noise -- the ceiling AND the
             validation gate against the NumPy reference
    sampled  finite shots on noiseless AerSimulator
    noisy    finite shots + depolarizing/readout noise, transpiled to
             the ECR/RZ/SX/X basis on a ring coupling map

Endianness (the classic Qiskit trap):
    reference: qubit 0 = LEFTMOST tensor factor (bit n-1-i of the index)
    qiskit:    qubit 0 = least-significant bit  (bit i of the index)
Each pathway reverses exactly ONCE, in exactly one place:
    statevector path -> _state_to_ref()
    counts path      -> counts_to_features()
Nothing else may reorder qubits.

Measurement: two circuits per window. Z-basis counts give the 5 <Z_i>
and 10 <Z_i Z_j> features; an H layer before measurement gives the 5
<X_i>. "Shots-per-feature" S = shots per basis circuit; report S next
to every sampled number (CLAUDE.md).

IBM path (DORMANT -- implemented, never exercised in this project):
    get_backend("ibm") authenticates qiskit-ibm-runtime with the
    IBM_QUANTUM_TOKEN env var and returns the least-busy real device;
    run_counts() dispatches it through SamplerV2. All reported results
    are simulation until this path is deliberately turned on.
"""

import os

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from qrc_core import SEED, WindowedReservoir

BASIS_GATES = ["ecr", "rz", "sx", "x"]
NOISE_P1 = 3e-4        # depolarizing, 1q gates (sx, x; rz is virtual)
NOISE_P2 = 7e-3        # depolarizing, 2q gates (ecr)
NOISE_RO = 1.5e-2      # symmetric readout flip probability


def print_config():
    print(f"qrc_qiskit config: basis={BASIS_GATES} "
          f"noise p1={NOISE_P1} p2={NOISE_P2} ro={NOISE_RO} seed={SEED}")


# -------------------------------------------------------------- endianness
def _state_to_ref(vec, n):
    """Qiskit little-endian statevector -> reference big-endian order.

    THE single reversal point for the statevector path.
    """
    out = np.empty_like(vec)
    for j in range(len(vec)):
        rj = int(format(j, f"0{n}b")[::-1], 2)
        out[rj] = vec[j]
    return out


def counts_to_features(counts_z, counts_x, n):
    """Counts (Z-basis, X-basis) -> the 20-feature vector, reference order:
    [<Z_i>]*n, [<X_i>]*n, [<Z_i Z_j>] for i<j lexicographic.

    THE single reversal point for the counts path: Qiskit count keys are
    LITTLE-ENDIAN (qubit 0 = rightmost char); key[::-1] puts qubit i at
    char i, matching the reference convention.
    """
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    z = np.zeros(n)
    zz = np.zeros(len(pairs))
    tot = 0
    for key, c in counts_z.items():
        bits = key.replace(" ", "")[::-1]              # <-- the reversal
        s = np.array([1.0 - 2.0 * int(b) for b in bits])   # '0'->+1 '1'->-1
        z += c * s
        zz += c * np.array([s[i] * s[j] for i, j in pairs])
        tot += c
    z, zz = z / tot, zz / tot

    x = np.zeros(n)
    tot = 0
    for key, c in counts_x.items():
        bits = key.replace(" ", "")[::-1]              # <-- the reversal
        x += c * np.array([1.0 - 2.0 * int(b) for b in bits])
        tot += c
    x /= tot
    return np.concatenate([z, x, zz])


# -------------------------------------------------------------- backends
def make_noise_model(p1=NOISE_P1, p2=NOISE_P2, p_ro=NOISE_RO):
    """Transparent parameterised noise: depolarizing on sx/x (p1) and
    ecr (p2), symmetric readout flips (p_ro). rz is virtual -> no error."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    nm = NoiseModel(basis_gates=BASIS_GATES)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["ecr"])
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


def get_backend(kind="aer", noise=True, seed=SEED,
                p1=NOISE_P1, p2=NOISE_P2, p_ro=NOISE_RO, min_qubits=5):
    """Backend factory. kind='aer' (primary) | 'ibm' (DORMANT hook)."""
    if kind == "aer":
        from qiskit_aer import AerSimulator
        nm = make_noise_model(p1, p2, p_ro) if noise else None
        # density_matrix at n=5: noise channels applied exactly, sampling
        # cost ~independent of shots (vs per-shot trajectories)
        be = AerSimulator(noise_model=nm, seed_simulator=seed,
                          method="density_matrix" if noise else "automatic")
        print(f"backend: AerSimulator noise={'on' if noise else 'off'}"
              + (f" (p1={p1} p2={p2} ro={p_ro})" if noise else "")
              + f" seed={seed}")
        return be
    if kind == "ibm":
        # Dormant real-hardware hook. Requires IBM_QUANTUM_TOKEN.
        from qiskit_ibm_runtime import QiskitRuntimeService
        token = os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            raise RuntimeError("IBM path: set the IBM_QUANTUM_TOKEN env var")
        service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                       token=token)
        be = service.least_busy(operational=True, simulator=False,
                                min_num_qubits=min_qubits)
        print(f"backend: IBM device {be.name} (REAL HARDWARE)")
        return be
    raise ValueError(f"unknown backend kind {kind!r}")


def run_counts(backend, circuits, shots, seed=SEED):
    """Counts for a list of bound circuits. Dispatches AerSimulator via
    backend.run; anything else (IBM device) via SamplerV2 (dormant)."""
    if backend.__class__.__name__ == "AerSimulator":
        res = backend.run(circuits, shots=shots, seed_simulator=seed).result()
        out = res.get_counts()
        return [out] if isinstance(out, dict) else out
    from qiskit_ibm_runtime import SamplerV2          # dormant path
    job = SamplerV2(mode=backend).run(circuits, shots=shots)
    return [r.data.meas.get_counts() for r in job.result()]


# -------------------------------------------------------------- reservoir
class QiskitReservoir:
    """Circuit twin of a WindowedReservoir (same seeds => same physics)."""

    def __init__(self, res=None, seed=SEED):
        self.res = res if res is not None else WindowedReservoir(seed=seed)
        self.n = self.res.n
        self._templates = {}        # L -> (bare, z_meas, x_meas, params)

    # ---- circuit construction
    def _build_templates(self, L):
        if L in self._templates:
            return self._templates[L]
        n, res = self.n, self.res
        u = ParameterVector("u", L)
        s = res.ent_scale                            # digital JDt analogue
        bare = QuantumCircuit(n)
        for k in range(L):
            for i in range(n):                       # encoding layer
                bare.ry(res.gamma * res.gains[i] * u[k], i)
            for i in range(n):                       # CP(s*pi) ring
                bare.cp(s * np.pi, i, (i + 1) % n)   # s=1 -> CZ
            for i in range(n):                       # frozen RY then RZ
                bare.ry(s * res.w_angles[i, 0], i)
                bare.rz(s * res.w_angles[i, 1], i)
        z_meas = bare.copy()
        z_meas.measure_all()
        x_meas = bare.copy()
        for i in range(n):                           # X basis: H then Z-meas
            x_meas.h(i)
        x_meas.measure_all()
        self._templates[L] = (bare, z_meas, x_meas, u)
        return self._templates[L]

    def _bind(self, template, params, window):
        return template.assign_parameters(
            {params[k]: float(window[k]) for k in range(len(window))})

    # ---- exact path (ceiling + validation gate)
    def exact_features(self, window):
        """Statevector features; must equal res.features(window)."""
        bare, _, _, u = self._build_templates(len(window))
        psi = _state_to_ref(Statevector(self._bind(bare, u, window)).data,
                            self.n)
        return np.array([np.real(psi.conj() @ (O @ psi))
                         for O in self.res.obs])

    def exact_feature_matrix(self, series, L):
        """Same shape/indexing as WindowedReservoir.feature_matrix."""
        T = len(series)
        Xf = np.zeros((T - L + 1, self.res.n_features()))
        for k in range(L - 1, T):
            Xf[k - L + 1] = self.exact_features(series[k - L + 1:k + 1])
        return Xf

    # ---- sampled / noisy path
    def transpiled_templates(self, L, backend=None):
        """Transpile the two measured templates once (params stay symbolic).
        Hardware-aware: ECR/RZ/SX/X basis on a ring coupling map for aer;
        the device's own target when an IBM backend is passed."""
        _, z_meas, x_meas, u = self._build_templates(L)
        if backend is not None and backend.__class__.__name__ != "AerSimulator":
            tz = transpile(z_meas, backend=backend, optimization_level=1,
                           seed_transpiler=SEED)
            tx = transpile(x_meas, backend=backend, optimization_level=1,
                           seed_transpiler=SEED)
        else:
            ring = [[i, (i + 1) % self.n] for i in range(self.n)]
            ring += [[j, i] for i, j in ring]
            tz, tx = (transpile(c, basis_gates=BASIS_GATES, coupling_map=ring,
                                optimization_level=1, seed_transpiler=SEED)
                      for c in (z_meas, x_meas))
        return tz, tx, u

    def sampled_feature_matrix(self, series, L, shots, backend,
                               batch=64, verbose=True):
        """Finite-shot features for every window. S = shots PER BASIS
        circuit (2 circuits per window). Returns (Xf, meta)."""
        tz, tx, u = self.transpiled_templates(L, backend)
        T = len(series)
        ks = range(L - 1, T)
        Xf = np.zeros((T - L + 1, self.res.n_features()))
        todo = [(k, series[k - L + 1:k + 1]) for k in ks]
        if verbose:
            print(f"sampling {len(todo)} windows, S={shots} shots/basis, "
                  f"2 circuits/window")
        for b0 in range(0, len(todo), batch):
            chunk = todo[b0:b0 + batch]
            circs = []
            for _, w in chunk:
                circs.append(self._bind(tz, u, w))
                circs.append(self._bind(tx, u, w))
            counts = run_counts(backend, circs, shots, seed=SEED + b0)
            for j, (k, _) in enumerate(chunk):
                Xf[k - L + 1] = counts_to_features(
                    counts[2 * j], counts[2 * j + 1], self.n)
        meta = {"shots_per_basis": shots, "circuits_per_window": 2}
        return Xf, meta


if __name__ == "__main__":
    print_config()
    qr = QiskitReservoir()
    w = np.linspace(0, 1, 24)

    ref = qr.res.features(w)
    exa = qr.exact_features(w)
    print(f"validation gate: max|qiskit_exact - numpy_ref| = "
          f"{np.max(np.abs(exa - ref)):.2e}")

    be0 = get_backend("aer", noise=False)
    s0, _ = qr.sampled_feature_matrix(w, 24, shots=4096, backend=be0,
                                      verbose=False)
    print(f"noiseless sampled (S=4096): max|sampled - exact| = "
          f"{np.max(np.abs(s0[0] - exa)):.3f}")

    be1 = get_backend("aer", noise=True)
    s1, _ = qr.sampled_feature_matrix(w, 24, shots=4096, backend=be1,
                                      verbose=False)
    print(f"noisy sampled     (S=4096): max|sampled - exact| = "
          f"{np.max(np.abs(s1[0] - exa)):.3f}")
