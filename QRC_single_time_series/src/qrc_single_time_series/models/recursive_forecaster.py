"""recursive_forecaster.py -- adapters exposing warm()/step() for the rollout engine.

Physics-free adapters that satisfy the ``autonomous.rollout`` model protocol, so
the whole closed-loop layer is testable before any quantum object exists (phase
file mandate):

  - ``AR1``: toy AR(1) y_t = c + phi y_{t-1}; the regression fixture for the
    horizon/feedback tests.
  - ``EncodedRecursive``: wraps a (feature_fn, readout, encode, decode) chain --
    predictions are decoded to physical units, fed back through ``encode`` -- the
    shape the stateful-FN and rewind reservoirs will plug into (spec s16.1/2).
  - ``RewindBuffer``: fixed-length window carrying a provenance mask (True =
    a fed-back prediction); shifts as the rollout advances until the window is all
    predictions (spec s16.2 window provenance). Implemented in P4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AR1:
    """Toy AR(1): warm -> last value; step(u) -> c + phi u (deterministic)."""

    phi: float
    c: float = 0.0

    def warm(self, history):
        return float(history[-1])

    def step(self, state, u):
        yhat = self.c + self.phi * u
        return yhat, yhat


@dataclass
class RewindBuffer:
    """Sliding window of the last ``length`` inputs with a predicted/real mask."""

    length: int
    values: np.ndarray = None
    predicted: np.ndarray = None

    def warm(self, history):
        h = np.asarray(history, dtype=float)
        if len(h) < self.length:
            h = np.concatenate([np.full(self.length - len(h), h[0]), h])
        self.values = h[-self.length:].astype(float).copy()
        self.predicted = np.zeros(self.length, dtype=bool)   # all real after warmup
        return self.values.copy()

    def push(self, value, predicted):
        """Shift in a new value at the right edge; drop the oldest."""
        self.values = np.roll(self.values, -1)
        self.predicted = np.roll(self.predicted, -1)
        self.values[-1] = float(value)
        self.predicted[-1] = bool(predicted)
        return self.values.copy()

    @property
    def all_predicted(self):
        return bool(self.predicted.all())

    @property
    def provenance_fraction(self):
        return float(self.predicted.mean())


@dataclass
class EncodedRecursive:
    """Adapter: encode(u)->s, feature_fn(s,state)->(state,feat), readout(feat)->y_phys.

    Feeds the decoded physical prediction back through ``encode`` each step. Kept
    minimal; the reservoir feature_fn/readout are supplied by later phases.
    """

    encode: object
    decode: object
    feature_fn: object
    readout: object
    _last_feat: np.ndarray = field(default=None, repr=False)

    def warm(self, history):
        state = None
        for u in np.asarray(history, dtype=float):
            state, feat = self.feature_fn(self.encode(u), state)
        self._last_feat = feat
        return state

    def step(self, state, u):
        s = self.encode(u)
        state, feat = self.feature_fn(s, state)
        yhat = float(self.readout(feat))
        return state, yhat


@dataclass
class StatefulFN:
    """Stateful FN reservoir autonomous adapter (spec s16.1: rho retained across steps).

    Drives an ``ExactQRC`` incrementally and a trained ``Readout``. ``warm`` replays
    the real history into the persistent density matrix; ``step`` injects the fed-back
    value, reads the virtual-node features and maps them through the readout. rho is
    NEVER reinitialised during the autonomous rollout.
    """

    qrc: object
    readout: object

    def warm(self, history):
        rho = self.qrc.initial_state()
        for u in np.asarray(history, dtype=float)[:-1]:
            _, rho = self.qrc.step(rho, u)
        return rho

    def step(self, rho, u):
        row, rho = self.qrc.step(rho, u)
        feat = np.concatenate([row, [1.0]])          # append bias column
        yhat = float(self.readout.predict(feat[None, :])[0])
        return rho, yhat
