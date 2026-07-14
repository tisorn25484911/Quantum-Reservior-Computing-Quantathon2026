"""pennylane_qrc_hybrid.py -- A QRC-inspired hybrid model in PennyLane.

Design: a windowed quantum feature map in the reservoir spirit (the
entangling body is frozen; only a classical ridge read-out is trained),
evaluated for the whole time series in ONE batched QNode call via
PennyLane's parameter broadcasting. A second stage then shows the one
thing differentiability buys a reservoir practitioner cheaply: tuning a
single global encoding gain gamma by alternating (i) a classical ridge
refit and (ii) a gradient step on gamma with the read-out held fixed.
This avoids differentiating through the linear solve and is the standard
practical recipe when a model sits on the reservoir/variational boundary.

Task: 3-month-ahead prediction of the REAL monthly ENSO sea-surface-
temperature anomaly (statsmodels 'elnino' dataset, 1950-2010).
Run:  python3 pennylane_qrc_hybrid.py        (about a minute)
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from datasets import enso_real

# ----------------------------------------------------------------- config
N, WINDOW, SEED, HORIZON = 5, 8, 7, 3
rng = np.random.default_rng(SEED)
dev = qml.device("default.qubit", wires=N)          # exact, broadcast-capable

fixed_weights = rng.uniform(0, 2 * np.pi, size=(2, N, 3))   # frozen reservoir
input_gains = rng.uniform(0.4, 1.0, size=(WINDOW, N))


@qml.qnode(dev, interface="autograd")
def reservoir(u_win, gamma):
    """u_win: (batch, WINDOW) windows; gamma: global encoding gain.

    Returns 2N + N(N-1)/2 = 20 features per window: <Z_i>, <X_i>, <Z_i Z_j>.
    Broadcasting handles the batch axis; one call evaluates every window.
    """
    for j in range(WINDOW):
        for q in range(N):
            qml.RY(gamma * np.pi * input_gains[j, q] * u_win[..., j], wires=q)
        qml.StronglyEntanglingLayers(fixed_weights, wires=range(N))
    return ([qml.expval(qml.PauliZ(q)) for q in range(N)]
            + [qml.expval(qml.PauliX(q)) for q in range(N)]
            + [qml.expval(qml.PauliZ(a) @ qml.PauliZ(b))
               for a in range(N) for b in range(a + 1, N)])


def windows(u):
    Xw = np.zeros((len(u), WINDOW))
    for k in range(len(u)):
        lo = max(0, k - WINDOW + 1)
        Xw[k, WINDOW - (k - lo + 1):] = u[lo:k + 1]
    return Xw


def ridge(F, y, lam=1e-2):
    Fd = np.hstack([F, np.ones((len(F), 1))])
    return np.linalg.solve(Fd.T @ Fd + lam * np.eye(Fd.shape[1]), Fd.T @ y)


def nmse(y, p):
    return float(np.mean((y - p) ** 2) / np.var(y))


# ---------------------------------------------------------------- pipeline
en = enso_real()
an = en.anomaly_c.to_numpy()
u = (an - an.min()) / (an.max() - an.min())          # -> [0, 1]
y = np.roll(an, -HORIZON)
u, y = u[:-HORIZON], y[:-HORIZON]
Xw = windows(u)
washout, split = 24, int(0.7 * len(y))


def evaluate(gamma):
    F = np.array(reservoir(Xw, float(gamma))).T          # (T, 20), one call
    w = ridge(F[washout:split], y[washout:split])
    Fd = np.hstack([F, np.ones((len(F), 1))])
    return nmse(y[split:], Fd[split:] @ w)


print(f"fixed reservoir, gamma = 1.00    | ENSO {HORIZON}-mo NMSE = "
      f"{evaluate(1.0):.4f}")
print(f"persistence baseline             | ENSO {HORIZON}-mo NMSE = "
      f"{nmse(y[split:], an[:-HORIZON][split:]):.4f}")

# ------------------- alternating tuning of the encoding gain (14 iterations)
# Step (i): refit the ridge read-out classically at the current gamma.
# Step (ii): one gradient step on gamma with that read-out FROZEN, so the
# gradient flows only through the quantum circuit (parameter-shift capable).
gamma = pnp.array(1.0, requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.15)
tr = slice(washout, split)
for it in range(14):
    F_tr = np.array(reservoir(Xw[tr], float(gamma))).T
    w_fix = pnp.array(ridge(F_tr, y[tr]), requires_grad=False)

    def loss(g):
        F = qml.math.stack(reservoir(Xw[tr], g)).T
        Fd = qml.math.concatenate([F, pnp.ones((F.shape[0], 1))], axis=1)
        return pnp.mean((Fd @ w_fix - y[tr]) ** 2)

    gamma, c = opt.step_and_cost(loss, gamma)
    print(f"  iter {it}: train MSE = {float(c):.4f}  ->  "
          f"gamma = {float(gamma):.3f}")

print(f"tuned encoding, gamma = {float(gamma):.2f}    | ENSO {HORIZON}-mo "
      f"NMSE = {evaluate(gamma):.4f}")
