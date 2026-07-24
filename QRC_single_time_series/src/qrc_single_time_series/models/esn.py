"""esn.py -- leaky echo-state network: effective-rank size-matched + best-of-class (G7).

Classical reservoir control. A random recurrent reservoir with fixed spectral
radius / leak / input scaling, read out by the shared ridge readout. The honesty
battery (G7) requires TWO configurations per experiment:

  (i)  size-matched at equal EFFECTIVE feature count to the QRC -- the reservoir
       size N_r is grown until the participation-ratio effective rank of its
       collected states matches the QRC's (NOT the nominal qubit/node count); this
       is the only comparable ESN column.
  (ii) best-of-class -- spectral radius / leak / input scale tuned on DEV only
       (never on test; Hamhoum-style best-on-test selection is deliberately not
       replicated).

State update (leaky-integrator ESN, Jaeger):
    x_{t} = (1-a) x_{t-1} + a tanh(W_in u_t + W x_{t-1} + b)
Teacher-forced training pairs the post-update state z_t = [x_t; u_t; 1] with the
next value y_{t+1} (G3). ``warm``/``step`` drive the SAME state autonomously
(spec s16.1 -- the reservoir state is retained, never reset). Implemented in P7.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import readout as R
from ..evaluation.metrics import effective_rank


@dataclass
class ESN:
    n_reservoir: int
    spectral_radius: float
    leak: float
    input_scale: float
    seed: int
    W: np.ndarray = field(repr=False, default=None)
    Win: np.ndarray = field(repr=False, default=None)
    bias: np.ndarray = field(repr=False, default=None)
    readout: object = None
    eff_rank: float = None

    # -- reservoir construction ----------------------------------------------
    def __post_init__(self):
        if self.W is None:
            rng = np.random.default_rng(self.seed)
            Nr = self.n_reservoir
            W = rng.uniform(-1.0, 1.0, (Nr, Nr))
            W[rng.uniform(0, 1, (Nr, Nr)) > 0.1] = 0.0     # 10% connectivity
            radius = np.max(np.abs(np.linalg.eigvals(W)))
            self.W = W * (self.spectral_radius / radius) if radius > 0 else W
            self.Win = rng.uniform(-1.0, 1.0, Nr) * self.input_scale
            self.bias = rng.uniform(-1.0, 1.0, Nr) * 0.1

    def _update(self, x, u):
        pre = self.Win * u + self.W @ x + self.bias
        return (1.0 - self.leak) * x + self.leak * np.tanh(pre)

    def collect(self, u_seq, washout=0):
        """Post-update states z_t=[x_t;u_t;1] over the driven sequence."""
        u_seq = np.asarray(u_seq, dtype=float)
        x = np.zeros(self.n_reservoir)
        rows = []
        for u in u_seq:
            x = self._update(x, u)
            rows.append(np.concatenate([x, [u, 1.0]]))
        return np.array(rows)[washout:]

    # -- fit / predict -------------------------------------------------------
    def fit(self, y, n_train, lam="gcv", washout=50):
        y = np.asarray(y, dtype=float)
        Z = self.collect(y[:-1])                          # z_t for t=0..len-2
        target = y[1:]                                    # y_{t+1}
        Ztr, ytr = Z[washout:n_train], target[washout:n_train]
        self.readout = R.fit(Ztr, ytr, lam=lam)
        self.eff_rank = effective_rank(Ztr[:, : self.n_reservoir])
        return self

    def predict_onestep(self, y):
        Z = self.collect(np.asarray(y, dtype=float)[:-1])
        return self.readout.predict(Z)

    # -- autonomous adapter --------------------------------------------------
    def warm(self, history):
        h = np.asarray(history, dtype=float)
        x = np.zeros(self.n_reservoir)
        for u in h[:-1]:                                  # last value arrives as first u
            x = self._update(x, u)
        return x

    def step(self, x, u):
        x = self._update(x, u)
        z = np.concatenate([x, [u, 1.0]])
        yhat = float(self.readout.predict(z[None, :])[0])
        return x, yhat


def best_of_class(y, n_train, dev_slice, seed=0,
                  radii=(0.7, 0.9, 1.1), leaks=(0.3, 0.6, 1.0),
                  scales=(0.2, 0.5, 1.0), n_reservoir=200, lam="gcv"):
    """Grid over (spectral_radius, leak, input_scale) selected on DEV NMSE only."""
    from ..evaluation.metrics import nmse_variance
    y = np.asarray(y, dtype=float)
    best, best_nmse = None, np.inf
    for r in radii:
        for a in leaks:
            for sc in scales:
                esn = ESN(n_reservoir, r, a, sc, seed).fit(y, n_train, lam=lam)
                pred = esn.predict_onestep(y)             # aligned to y[1:]
                dev = dev_slice
                nm = nmse_variance(y[1:][dev], pred[dev])
                if nm < best_nmse:
                    best_nmse, best = nm, esn
    best.dev_nmse = float(best_nmse)
    return best


def size_matched(y, n_train, target_eff_rank, seed=0,
                 spectral_radius=0.9, leak=0.6, input_scale=0.5,
                 sizes=(2, 3, 5, 8, 12, 20, 40, 80), lam="gcv"):
    """Smallest reservoir whose state effective rank first reaches ``target``.

    G7: the comparable ESN matches the QRC's EFFECTIVE feature count (participation
    ratio), not its qubit count. Returns the ESN plus the matching table.
    """
    y = np.asarray(y, dtype=float)
    table = []
    chosen = None
    for Nr in sizes:
        esn = ESN(Nr, spectral_radius, leak, input_scale, seed).fit(y, n_train, lam=lam)
        table.append({"n_reservoir": Nr, "eff_rank": esn.eff_rank})
        if chosen is None and esn.eff_rank >= target_eff_rank:
            chosen = esn
    if chosen is None:                                    # never reached: take largest
        chosen = esn
    chosen.match_table = table
    chosen.target_eff_rank = float(target_eff_rank)
    return chosen
