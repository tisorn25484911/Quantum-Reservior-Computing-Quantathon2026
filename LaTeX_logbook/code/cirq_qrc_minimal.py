"""cirq_qrc_minimal.py -- A minimal quantum reservoir computer in Cirq.

Protocol: the *restart (rewind) protocol*. For every time step k we build a
fresh circuit that re-uploads the most recent L inputs u_{k-L+1}, ..., u_k,
interleaved with a fixed random entangling block W. Because the reservoir
has fading memory, a finite window L approximates the full input history,
and no mid-circuit measurement is needed -- every feature is read from a
terminal measurement, which suits any gate-model backend.

Two design choices matter and are worth reading closely: the encoding
scale (pi/2 here; a pi-scale drives the features into a high-frequency,
poorly-generalising regime) and the observable set (local <Z_i> alone is
nearly rank-deficient; adding <X_i> and <Z_i Z_j> triples the usable
feature directions).

Task: one-step-ahead prediction of the NARMA-2 benchmark sequence.
Run:   python3 cirq_qrc_minimal.py          (finishes in a few seconds)
"""
import cirq
import numpy as np
import sympy

# ----------------------------------------------------------------- config
N_QUBITS = 5        # reservoir size
WINDOW = 6          # input window L re-uploaded per circuit (fading memory)
T = 400             # length of the driving sequence
SHOTS = 0           # 0 -> exact expectation values; >0 -> sampled estimates
SCALE = np.pi / 2   # encoding scale (see module docstring)
NOISE_P = 0.0       # per-moment depolarising probability (try 0.002)
SEED = 7

rng = np.random.default_rng(SEED)
qubits = cirq.LineQubit.range(N_QUBITS)

# ------------------------------------------------- fixed entangling block W
# One layer = random single-qubit Y-rotations (drawn once, then frozen)
# followed by a brickwork of CZ gates. W is the "reservoir dynamics": it is
# never trained, exactly as W_res is frozen in a classical echo-state network.
fixed_angles = rng.uniform(0, 2 * np.pi, size=(2, N_QUBITS))


def entangling_block() -> cirq.Circuit:
    ops = []
    for layer in range(2):
        ops.append(cirq.Moment(
            cirq.ry(fixed_angles[layer, i]).on(q) for i, q in enumerate(qubits)))
        pairs = [(i, i + 1) for i in range(layer % 2, N_QUBITS - 1, 2)]
        ops.append(cirq.Moment(cirq.CZ(qubits[a], qubits[b]) for a, b in pairs))
    return cirq.Circuit(ops)


W_BLOCK = entangling_block()

# ------------------------------------------------ parameterised input layer
# The window of inputs enters through symbolic RY angles; sympy symbols keep
# one template circuit whose parameters are bound per time step via a
# ParamResolver -- much faster than rebuilding gates from floats.
theta = [sympy.Symbol(f"th{j}") for j in range(WINDOW)]


def template() -> cirq.Circuit:
    c = cirq.Circuit()
    for j in range(WINDOW):
        # Each past input is written on every qubit (global re-uploading),
        # scaled per qubit by a fixed random gain to break symmetry.
        gains = input_gains[j]
        c.append(cirq.Moment(
            cirq.ry(gains[i] * theta[j]).on(q) for i, q in enumerate(qubits)))
        c += W_BLOCK
    if SHOTS:
        c.append(cirq.measure(*qubits, key="z"))
    return c


input_gains = rng.uniform(0.4, 1.0, size=(WINDOW, N_QUBITS))
TEMPLATE = template()

# ----------------------------------------------------------------- backend
if NOISE_P > 0:
    sim = cirq.DensityMatrixSimulator(
        noise=cirq.ConstantQubitNoiseModel(cirq.depolarize(NOISE_P)))
else:
    sim = cirq.Simulator(seed=SEED)
observables = ([cirq.Z(q) for q in qubits] + [cirq.X(q) for q in qubits]
               + [cirq.Z(qubits[a]) * cirq.Z(qubits[b])
                  for a in range(N_QUBITS) for b in range(a + 1, N_QUBITS)])


def features_for_window(u_win: np.ndarray) -> np.ndarray:
    """Map the last L inputs (in [0,1]) to the 20 features <Z_i>,<X_i>,<Z_iZ_j>."""
    resolver = cirq.ParamResolver({theta[j]: SCALE * u_win[j]
                                   for j in range(WINDOW)})
    if SHOTS:
        result = sim.run(TEMPLATE, resolver, repetitions=SHOTS)
        bits = 1.0 - 2.0 * result.measurements["z"]      # 0 -> +1, 1 -> -1
        z = bits.mean(axis=0)
        zz = [np.mean(bits[:, a] * bits[:, b])
              for a in range(N_QUBITS) for b in range(a + 1, N_QUBITS)]
        return np.concatenate([z, np.zeros(N_QUBITS), zz])  # (X needs basis change)
    vals = sim.simulate_expectation_values(TEMPLATE, observables, resolver)
    return np.real(np.asarray(vals))


# --------------------------------------------------------------- benchmark
def narma2(T, seed=0):
    g = np.random.default_rng(seed)
    u = g.uniform(0, 0.5, size=T)
    y = np.zeros(T)
    for k in range(1, T - 1):
        y[k + 1] = 0.4 * y[k] + 0.4 * y[k] * y[k - 1] + 0.6 * u[k] ** 3 + 0.1
    return u, y


u, y = narma2(T, seed=1)
u_scaled = u / 0.5                                  # -> [0, 1]

X = np.zeros((T, len(observables)))
for k in range(T):
    win = np.zeros(WINDOW)
    lo = max(0, k - WINDOW + 1)
    win[WINDOW - (k - lo + 1):] = u_scaled[lo:k + 1]
    X[k] = features_for_window(win)

# Ridge read-out (the only trained component)
washout, split = 20, int(0.7 * T)
Xd = np.hstack([X, np.ones((T, 1))])
A = Xd[washout:split]
w = np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ y[washout:split])
pred = Xd[split:] @ w
nmse = np.mean((y[split:] - pred) ** 2) / np.var(y[split:])
print(f"Cirq restart-protocol QRC | qubits={N_QUBITS} window={WINDOW} "
      f"shots={'exact' if SHOTS == 0 else SHOTS} | test NMSE = {nmse:.4f}")
