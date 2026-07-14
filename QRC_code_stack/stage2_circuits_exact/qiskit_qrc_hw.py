"""qiskit_qrc_hw.py -- A hardware-aware quantum reservoir computer in Qiskit.

Two parts:

  PART A  Windowed (restart-protocol) reservoir executed the way an IBM job
          would run: transpiled to a linear coupling map and a native basis
          (ecr, id, rz, sx, x), under a depolarising + readout noise model,
          with features estimated from a finite number of shots.
          Task: one-step-ahead prediction of the solar clear-sky index
          (synthetic surrogate from datasets.py).

  PART B  A mid-circuit measurement + reset demonstration: the dynamic-
          circuit primitive that realises the Fujii--Nakajima partial-trace
          injection on hardware and gives memory unbounded by T1/T2.

Swap `AerSimulator` for a `QiskitRuntimeService` backend (and keep the same
`transpile` call against `backend.target`) to run PART A on an IBM device.
Run:  python3 qiskit_qrc_hw.py            (about a minute on one core)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from datasets import solar_surrogate

# ----------------------------------------------------------------- config
N, WINDOW, T, SHOTS, SEED = 5, 8, 400, 2048, 7
rng = np.random.default_rng(SEED)

BASIS = ["ecr", "id", "rz", "sx", "x"]                # IBM Heron native set
COUPLING = CouplingMap([(i, i + 1) for i in range(N - 1)])  # linear chain


def noise_model() -> NoiseModel:
    nm = NoiseModel(basis_gates=BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(4e-4, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(6e-3, 2), ["ecr"])
    ro = ReadoutError([[0.99, 0.01], [0.02, 0.98]])
    for q in range(N):
        nm.add_readout_error(ro, [q])
    return nm


BACKEND = AerSimulator(noise_model=noise_model(), seed_simulator=SEED)

# --------------------------------------------- PART A: windowed reservoir
fixed_angles = rng.uniform(0, 2 * np.pi, size=(2, N))
input_gains = rng.uniform(0.4, 1.0, size=(WINDOW, N))


def reservoir_circuit(u_win: np.ndarray) -> QuantumCircuit:
    """Fresh circuit re-uploading the last WINDOW inputs (restart protocol)."""
    qc = QuantumCircuit(N, N)
    for j in range(WINDOW):
        for q in range(N):
            qc.ry(float(np.pi * input_gains[j, q] * u_win[j]), q)
        for layer in range(2):                       # frozen entangling block
            for q in range(N):
                qc.ry(float(fixed_angles[layer, q]), q)
            for a in range(layer % 2, N - 1, 2):
                qc.cz(a, a + 1)
    qc.measure(range(N), range(N))
    return qc


def counts_to_features(counts: dict) -> np.ndarray:
    """<Z_i> and <Z_i Z_j> from a counts dictionary (little-endian keys)."""
    shots = sum(counts.values())
    z = np.zeros(N)
    zz = np.zeros((N, N))
    for key, c in counts.items():
        bits = np.array([1 - 2 * int(b) for b in key[::-1]])   # 0->+1, 1->-1
        z += c * bits
        zz += c * np.outer(bits, bits)
    z /= shots
    zz /= shots
    pairs = [zz[a, b] for a in range(N) for b in range(a + 1, N)]
    return np.concatenate([z, pairs])                # 5 + 10 = 15 features


# Data: daytime clear-sky index, one-step-ahead target
sol = solar_surrogate()
day = sol[(sol.elevation_deg > 5) & sol.ghi.notna()]
kt = np.clip(day.kt.to_numpy()[:T + 1], 0, 1.05)
u_series, target = kt[:-1] / 1.05, kt[1:]

# Build, transpile, and run ALL step circuits as one batched job -- exactly
# how a hardware submission is structured (one job, T circuits, S shots).
circuits = []
for k in range(T):
    win = np.zeros(WINDOW)
    lo = max(0, k - WINDOW + 1)
    win[WINDOW - (k - lo + 1):] = u_series[lo:k + 1]
    circuits.append(reservoir_circuit(win))
tcircs = transpile(circuits, backend=BACKEND, coupling_map=COUPLING,
                   basis_gates=BASIS, optimization_level=1, seed_transpiler=SEED)
print(f"transpiled depth (first circuit): {tcircs[0].depth()}, "
      f"2q gates: {tcircs[0].count_ops().get('ecr', 0)}")
job = BACKEND.run(tcircs, shots=SHOTS)
Xq = np.array([counts_to_features(c) for c in job.result().get_counts()])

# Hybrid augmentation: append cheap classical lags to the quantum features.
# In principle the read-out can fall back on the lags alone; in practice,
# with ~250 training points, fifteen shot-noisy quantum columns inject
# enough variance that even a tuned ridge cannot fully ignore them, and the
# hybrid lands slightly BELOW the classical-only baseline. Reproducing this
# effect -- and being suspicious of any benchmark that hides it -- is one of
# the more valuable exercises in this file.
lags = np.stack([np.roll(u_series, i) for i in range(3)], axis=1)
X = np.hstack([Xq, lags])

# Ridge read-out with the penalty chosen by generalised cross-validation
# (GCV) on the training span only -- deterministic and split-free, which
# matters here because a short chronological validation tail can land in an
# unrepresentative cloud regime and mis-select the penalty.
washout, split = 20, int(0.7 * T)


def fit_eval(Xf):
    Xd = np.hstack([Xf, np.ones((T, 1))])
    A, ytr = Xd[washout:split], target[washout:split]
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Uty, n = U.T @ ytr, A.shape[0]
    best = (np.inf, None)
    for lam in 10.0 ** np.arange(-6, 2):
        f = s ** 2 / (s ** 2 + lam)              # shrinkage factors
        resid = ytr - U @ (f * Uty)
        gcv = n * (resid @ resid) / (n - f.sum()) ** 2
        if gcv < best[0]:
            best = (gcv, lam)
    lam = best[1]
    w = Vt.T @ ((s / (s ** 2 + lam)) * Uty)
    p = Xd[split:] @ w
    return np.mean((target[split:] - p) ** 2) / np.var(target[split:])


nmse = fit_eval(X)
nmse_qonly = fit_eval(Xq)
nmse_conly = fit_eval(lags)
pers = np.mean((target[split:] - u_series[split:] * 1.05) ** 2) / np.var(target[split:])
print(f"PART A | noisy, {SHOTS} shots, native basis")
print(f"  quantum + classical lags : NMSE = {nmse:.4f}")
print(f"  quantum features only    : NMSE = {nmse_qonly:.4f}")
print(f"  classical lags only      : NMSE = {nmse_conly:.4f}")
print(f"  persistence              : NMSE = {pers:.4f}")

# ------------------------- PART B: mid-circuit measurement + reset (leaky)
# One LONG circuit processes STEPS inputs sequentially. Each step: encode
# u_k on the I/O qubit, entangle, measure the I/O qubit into its own
# classical register, then RESET it. The unmeasured qubits carry memory
# forward; the reset realises the partial-trace injection on hardware and
# decouples usable memory from T1/T2 (the NISQRC mechanism).
STEPS = 12
qr, io = QuantumRegister(N, "q"), 0
regs = [ClassicalRegister(1, f"s{k}") for k in range(STEPS)]
qc = QuantumCircuit(qr, *regs)
u_demo = u_series[:STEPS]
for k in range(STEPS):
    if k:
        qc.reset(io)
    qc.ry(float(np.pi * u_demo[k]), io)
    for a in range(N - 1):
        qc.cz(a, a + 1)
    for q in range(N):
        qc.ry(float(fixed_angles[0, q]), q)
    qc.measure(io, regs[k])
tqc = transpile(qc, backend=BACKEND, coupling_map=COUPLING,
                basis_gates=BASIS, optimization_level=1, seed_transpiler=SEED)
res = BACKEND.run(tqc, shots=4096).result().get_counts()
traj = np.zeros(STEPS)
for key, c in res.items():                 # key = "sK ... s1 s0" space-joined
    bits = key.split()[::-1]               # -> chronological order
    for k in range(STEPS):
        traj[k] += c * (1 - 2 * int(bits[k]))
traj /= 4096
print("PART B | per-step <Z_io> trajectory from mid-circuit measurement:")
print("  ", np.array2string(traj, precision=3))
