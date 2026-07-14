"""baselines.py -- Full baseline battery + shared eval harness.

Battery (playbook: battery is part of the model):
    mean         train-mean predictor, NMSE = 1.0 by construction
    persistence  y_hat(k+H) = y(k)
    linear_lags  ridge (GCV) on the same L-lag window the QRC sees
    esn          size-matched leaky echo state network, 20 units = 20 features
    haar         WindowedReservoir with the frozen entangler replaced by a
                 Haar-random unitary -- same observables, same read-out;
                 separates "tuned dynamics" from "being quantum"

Alignment convention shared by every model (and the Qiskit path):
    sample k uses inputs y[k-L+1 .. k] and target y[k+H].
Chronological split only; ridge lambda by GCV inside the train span.
"""

import numpy as np

from qrc_core import WindowedReservoir, add_bias, nmse, ridge_gcv

HORIZON = 3      # months ahead
L = 24           # input window, months (>= 18 per handbook 40.3)
LAMS = np.logspace(-8, 4, 25)
EVENT_THRESHOLD = 0.5   # deg C |anomaly| defining a warm/cold event


def print_config():
    print(f"baselines config: H={HORIZON} L={L} event_thr={EVENT_THRESHOLD}")


# ------------------------------------------------------------ harness
def make_targets(y):
    """Sample indices k = L-1 .. T-H-1; returns (ks, y_target)."""
    ks = np.arange(L - 1, len(y) - HORIZON)
    return ks, y[ks + HORIZON]


def chrono_split(n_samples, train_frac=0.8):
    cut = int(n_samples * train_frac)
    return np.arange(cut), np.arange(cut, n_samples)


def fit_predict_ridge(Xf, yt, tr, te):
    """Standardise by train stats, ridge via GCV on train, predict test."""
    mu, sd = Xf[tr].mean(axis=0), Xf[tr].std(axis=0)
    sd[sd == 0] = 1.0
    Xd = add_bias((Xf - mu) / sd)
    w, lam = ridge_gcv(Xd[tr], yt[tr], LAMS)
    return Xd[te] @ w, lam


def score(y_true, y_pred, y_train):
    """NMSE overall + event-conditional (|anomaly| > threshold)."""
    m = y_train.mean()
    ev = np.abs(y_true) > EVENT_THRESHOLD
    out = {"nmse": float(nmse(y_true, y_pred, m)), "n": int(len(y_true)),
           "n_event": int(ev.sum())}
    out["nmse_event"] = float(nmse(y_true[ev], y_pred[ev], m)) if ev.any() else None
    return out


# ------------------------------------------------------------ battery
def run_mean(y, ks, yt, tr, te):
    return np.full(len(te), yt[tr].mean())


def run_persistence(y, ks, yt, tr, te):
    return y[ks[te]]


def lag_matrix(y, ks):
    """The L-lag window ending at k, one row per sample."""
    return np.stack([y[k - L + 1:k + 1] for k in ks])


def run_linear_lags(y, ks, yt, tr, te):
    pred, _ = fit_predict_ridge(lag_matrix(y, ks), yt, tr, te)
    return pred


class ESN:
    """Leaky echo state network, N units = N read-out features."""

    def __init__(self, n_units=20, rho=0.9, leak=0.3, in_scale=1.0, seed=7):
        rng = np.random.default_rng(seed)
        Wr = rng.normal(size=(n_units, n_units))
        Wr *= rho / np.max(np.abs(np.linalg.eigvals(Wr)))
        self.Wr = Wr
        self.Win = rng.uniform(-in_scale, in_scale, size=n_units)
        self.b = rng.uniform(-0.1, 0.1, size=n_units)
        self.leak = leak

    def states(self, y, x0=None):
        """Recurrent state trajectory over the full series."""
        x = np.zeros(len(self.Win)) if x0 is None else x0
        out = np.zeros((len(y), len(x)))
        for t, u in enumerate(y):
            x = (1 - self.leak) * x + self.leak * np.tanh(
                self.Wr @ x + self.Win * u + self.b)
            out[t] = x
        return out


def run_esn(y, ks, yt, tr, te, seed=7):
    Xf = ESN(seed=seed).states(y)[ks]     # state at time k, washout = L-1
    pred, _ = fit_predict_ridge(Xf, yt, tr, te)
    return pred


def haar_reservoir(seed=7):
    """WindowedReservoir with W replaced by a Haar-random unitary."""
    r = WindowedReservoir(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    A = rng.normal(size=(2 ** r.n, 2 ** r.n)) + 1j * rng.normal(size=(2 ** r.n, 2 ** r.n))
    Q, R = np.linalg.qr(A)
    r.W = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))
    return r


def run_haar(y, ks, yt, tr, te, scaler, seed=7):
    Xf = haar_reservoir(seed=seed).feature_matrix(scaler(y), L)[ks - (L - 1)]
    pred, _ = fit_predict_ridge(Xf, yt, tr, te)
    return pred


def run_battery(y, scaler, seed=7):
    """All baselines on anomaly series y; scaler maps y -> [0,1] for
    quantum encodings (fit on train span by the caller)."""
    ks, yt = make_targets(y)
    tr, te = chrono_split(len(ks))
    runners = {
        "mean": lambda: run_mean(y, ks, yt, tr, te),
        "persistence": lambda: run_persistence(y, ks, yt, tr, te),
        "linear_lags": lambda: run_linear_lags(y, ks, yt, tr, te),
        "esn": lambda: run_esn(y, ks, yt, tr, te, seed=seed),
        "haar": lambda: run_haar(y, ks, yt, tr, te, scaler, seed=seed),
    }
    results = {}
    for name, fn in runners.items():
        results[name] = score(yt[te], fn(), yt[tr])
    return results, (ks, yt, tr, te)


if __name__ == "__main__":
    print_config()
    rng = np.random.default_rng(0)
    ar = np.zeros(500)
    for t in range(1, 500):
        ar[t] = 0.9 * ar[t - 1] + rng.normal(scale=0.3)
    scaler = lambda y: np.clip((y - ar[:400].min()) / np.ptp(ar[:400]), 0, 1)
    results, _ = run_battery(ar, scaler)
    for name, s in results.items():
        print(f"{name:12s} NMSE={s['nmse']:.3f} event={s['nmse_event']} "
              f"(n={s['n']}, n_event={s['n_event']})")
