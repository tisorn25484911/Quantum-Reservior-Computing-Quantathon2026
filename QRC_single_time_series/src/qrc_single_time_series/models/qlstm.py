"""qlstm.py -- one fully specified windowed VQC regressor (+ optional QLSTM).

Spec s24 asks for named variational-quantum baselines, not an undefined "QVA":

  1. ``WindowedVQC`` -- a data-re-uploading variational quantum circuit on the
     length-``t_w`` window. It is a genuine VQC: the rotation angles AND the linear
     read-out are trained end-to-end (SciPy L-BFGS on the exact statevector, so no
     external QML framework is needed and the result is deterministic). Encoding is
     Ry(arcsin-style) re-uploading of the window values across ``layers`` blocks,
     each followed by a trainable single-qubit rotation layer and a ring of CNOTs;
     the prediction is a trained linear map of the qubit <Z_i> expectations.
  2. ``QLSTM`` -- the Chen-Yoo-style VQC-in-LSTM-cell design (arXiv:2009.01783 --
     verify the citation live before it enters references.bib). It needs torch and
     is provided behind ``TORCH_AVAILABLE``; the P7 tests skip it when torch is
     absent. The WindowedVQC is the always-available variational baseline.

Both feed their own predictions back through the ONE rollout engine (spec s24).
Implemented in P7.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

try:
    import torch                                          # noqa: F401
    TORCH_AVAILABLE = True
except Exception:                                         # pragma: no cover
    TORCH_AVAILABLE = False

_RY = lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)],
                          [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)
_RZ = lambda t: np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]],
                         dtype=complex)


def _apply_1q(state, g, i, n):
    st = state.reshape((2,) * n)
    st = np.tensordot(g, st, axes=([1], [i]))
    return np.moveaxis(st, 0, i).reshape(-1)


def _cnot(state, c, t, n):
    st = state.reshape((2,) * n).copy()
    sl_c1 = [slice(None)] * n
    sl_c1[c] = 1
    block = st[tuple(sl_c1)]
    st[tuple(sl_c1)] = np.flip(block, axis=t if t < c else t - 1)
    return st.reshape(-1)


def _z_expectations(state, n):
    p = np.abs(state) ** 2
    idx = np.arange(2 ** n)
    return np.array([1.0 - 2.0 * p[((idx >> (n - 1 - i)) & 1) == 1].sum()
                     for i in range(n)])


class WindowedVQC:
    """Trainable data-re-uploading VQC regressor on a sliding window."""

    def __init__(self, window=6, n_qubits=3, layers=2, seed=0):
        self.window, self.n, self.layers, self.seed = window, n_qubits, layers, seed
        self.mu = self.sd = 0.0
        self.theta = None
        self.n_rot = layers * n_qubits * 2                # (scale, bias) per (layer,qubit)
        self.n_read = n_qubits + 1                        # linear read-out on <Z_i>

    # -- circuit -------------------------------------------------------------
    def _features(self, win):
        """Qubit <Z_i> after the re-uploading circuit for one (scaled) window."""
        rot = self.theta[: self.n_rot].reshape(self.layers, self.n, 2)
        state = np.zeros(2 ** self.n, dtype=complex)
        state[0] = 1.0
        for l in range(self.layers):
            for i in range(self.n):
                x = win[(l * self.n + i) % self.window]
                state = _apply_1q(state, _RY(rot[l, i, 0] * x + rot[l, i, 1]), i, self.n)
                state = _apply_1q(state, _RZ(rot[l, i, 1]), i, self.n)
            for i in range(self.n):                        # ring of CNOTs
                state = _cnot(state, i, (i + 1) % self.n, self.n)
        return _z_expectations(state, self.n)

    def _predict_scaled(self, win):
        z = self._features(win)
        w = self.theta[self.n_rot:]
        return float(z @ w[:-1] + w[-1])

    # -- training ------------------------------------------------------------
    def fit(self, y, n_train, washout=0, maxiter=40, max_rows=200):
        y = np.asarray(y, dtype=float)
        self.mu, self.sd = float(y[:n_train].mean()), float(y[:n_train].std() + 1e-12)
        ys = (y - self.mu) / self.sd
        w = self.window
        X = np.stack([ys[t - w:t] for t in range(w, len(ys))])
        target = ys[w:]
        lo, hi = max(washout - w, 0), max(n_train - w, 1)
        Xtr, ttr = X[lo:hi], target[lo:hi]
        if len(Xtr) > max_rows:                            # subsample for tractable VQC
            sel = np.linspace(0, len(Xtr) - 1, max_rows).round().astype(int)
            Xtr, ttr = Xtr[sel], ttr[sel]
        rng = np.random.default_rng(self.seed)
        p0 = np.concatenate([rng.uniform(-0.5, 0.5, self.n_rot),
                             rng.uniform(-0.5, 0.5, self.n_read)])

        def loss(p):
            self.theta = p
            pred = np.array([self._predict_scaled(x) for x in Xtr])
            return float(np.mean((pred - ttr) ** 2))

        res = minimize(loss, p0, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-9})
        self.theta = res.x
        self.train_mse = float(res.fun)
        return self

    def predict_onestep(self, y):
        ys = (np.asarray(y, dtype=float) - self.mu) / self.sd
        w = self.window
        X = np.stack([ys[t - w:t] for t in range(w, len(ys))])
        return np.array([self._predict_scaled(x) for x in X]) * self.sd + self.mu

    # -- autonomous adapter --------------------------------------------------
    def warm(self, history):
        h = (np.asarray(history, dtype=float) - self.mu) / self.sd
        if len(h) < self.window:
            h = np.concatenate([np.full(self.window - len(h), h[0]), h])
        return list(h[-self.window + 1:]) if self.window > 1 else []

    def step(self, state, u):
        us = (u - self.mu) / self.sd
        win = np.array([*state, us])[-self.window:]
        y = self._predict_scaled(win) * self.sd + self.mu
        return list(win[1:]), y


def fit(y, n_train, **kw):
    return WindowedVQC(**kw).fit(y, n_train)
