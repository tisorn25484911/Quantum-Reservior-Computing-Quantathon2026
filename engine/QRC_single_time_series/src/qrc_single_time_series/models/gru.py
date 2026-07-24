"""gru.py -- small GRU forecaster (torch, optional ``neural`` extra).

Thin wrapper over the shared sliding-window RNN regressor in ``lstm.py`` with the
GRU cell. Same dev-selected early-stopping discipline and the same ``warm``/``step``
autonomous adapter (spec s24). ``AVAILABLE`` mirrors torch availability so the P7
tests skip cleanly when the optional extra is absent. Implemented in P7.
"""
from __future__ import annotations

from .lstm import AVAILABLE, RNNForecaster


def fit(y, n_train, **kw):
    return RNNForecaster(cell="gru", **kw).fit(y, n_train)
