"""lstm.py -- small LSTM forecaster (torch, optional ``neural`` extra).

A deliberately small single-layer LSTM regressor on a sliding window, trained with
early stopping on a DEV tail. On a few hundred monthly points these overfit and are
expected to LOSE to the reservoirs (consistent with Hamhoum Table 3 and the RC
literature) -- that is a finding, reported with the same dev-selection discipline,
not a bug. torch is an optional extra; ``AVAILABLE`` is False when it is absent and
the P7 tests skip accordingly. The ``warm``/``step`` adapter drives the network
autonomously through the ONE rollout engine (spec s24). Implemented in P7.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
    AVAILABLE = True
except Exception:                                         # pragma: no cover
    AVAILABLE = False


def _windows(y, w):
    """Sliding windows X[t]=y[t-w:t] and targets y[t] (t>=w)."""
    y = np.asarray(y, dtype=float)
    X = np.stack([y[t - w:t] for t in range(w, len(y))])
    return X, y[w:]


if AVAILABLE:
    class _Net(nn.Module):
        def __init__(self, cell, hidden):
            super().__init__()
            rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
            self.rnn = rnn(1, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)


class RNNForecaster:
    """Shared LSTM/GRU sliding-window regressor (``cell`` selects the cell type)."""

    def __init__(self, cell="lstm", window=12, hidden=16, seed=0,
                 lr=0.01, max_epochs=300, patience=25):
        if not AVAILABLE:                                 # pragma: no cover
            raise RuntimeError("lstm/gru baselines require the optional 'torch' extra")
        self.cell, self.window, self.hidden, self.seed = cell, window, hidden, seed
        self.lr, self.max_epochs, self.patience = lr, max_epochs, patience
        self.mu = self.sd = 0.0
        self.net = None

    def fit(self, y, n_train, dev_frac=0.2):
        torch.manual_seed(self.seed)
        y = np.asarray(y, dtype=float)
        self.mu, self.sd = float(y[:n_train].mean()), float(y[:n_train].std() + 1e-12)
        ys = (y - self.mu) / self.sd
        X, t = _windows(ys, self.window)
        m = n_train - self.window
        n_dev = max(int(dev_frac * m), 1)
        Xtr, ttr = X[:m - n_dev], t[:m - n_dev]
        Xdev, tdev = X[m - n_dev:m], t[m - n_dev:m]
        to = lambda a: torch.tensor(a, dtype=torch.float32)
        Xtr_t, ttr_t = to(Xtr)[..., None], to(ttr)
        Xdev_t, tdev_t = to(Xdev)[..., None], to(tdev)
        self.net = _Net(self.cell, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.MSELoss()
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net.train(); opt.zero_grad()
            lossf(self.net(Xtr_t), ttr_t).backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                dv = float(lossf(self.net(Xdev_t), tdev_t))
            if dv < best - 1e-6:
                best, best_state, bad = dv, {k: v.clone() for k, v in
                                             self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.dev_nmse = best / (tdev.var() + 1e-12)
        return self

    def _predict_scaled(self, win):
        with torch.no_grad():
            x = torch.tensor(win, dtype=torch.float32)[None, :, None]
            return float(self.net(x))

    def predict_onestep(self, y):
        ys = (np.asarray(y, dtype=float) - self.mu) / self.sd
        X, _ = _windows(ys, self.window)
        with torch.no_grad():
            out = self.net(torch.tensor(X, dtype=torch.float32)[..., None]).numpy()
        return out * self.sd + self.mu

    # -- autonomous adapter --------------------------------------------------
    def warm(self, history):
        h = (np.asarray(history, dtype=float) - self.mu) / self.sd
        if len(h) < self.window:
            h = np.concatenate([np.full(self.window - len(h), h[0]), h])
        return list(h[-self.window + 1:]) if self.window > 1 else []

    def step(self, state, u):
        us = (u - self.mu) / self.sd
        win = np.array([*state, us])[-self.window:]
        yscaled = self._predict_scaled(win)
        return list(win[1:]), yscaled * self.sd + self.mu


def fit(y, n_train, **kw):
    return RNNForecaster(cell="lstm", **kw).fit(y, n_train)
