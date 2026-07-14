"""rfqrc_baselines.py -- expanded baseline battery for stage 6.

Standing five (Part IX R5) plus the Part X additions:

    mean          NMSE = 1.0 line by construction
    persistence   y_hat(k+h) = y(k)
    linear_lags   ridge on the delay-embedded window
    esn           leaky echo state network, feature-count matched
    haar          RFQRC with entangler="haar_control" (built via config)
    nvar          next-generation reservoir computing (Gauthier et al.,
                  Nat. Commun. 12, 5564 (2021)): delay embedding +
                  quadratic monomials + ridge. RF-QRC's sharpest
                  classical analogue -- absent from the published
                  comparisons; if it wins, that is the headline.
    qelm          RF-QRC ablation with leak_eps = 1 (no memory) -- built
                  via config, not here.

Feature-count matching rule (Part X Phase 3): ESN nodes = NVAR features
= quantum feature dimension, so all read-outs train equally many weights.

Fairness rule: any nonlinear head offered to the quantum model must be
offered to every baseline in the same table -- one evaluation harness
(`fit_eval`) applies any model's features to the same ridge/GCV read-out.
"""

from __future__ import annotations

import numpy as np

SEED = 7


# ------------------------------------------------------------ read-out
def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge solution; anchor: recovers a known linear map to 1e-10."""
    Xd = np.hstack([X, np.ones((len(X), 1))])
    A = Xd.T @ Xd + lam * np.eye(Xd.shape[1])
    return np.linalg.solve(A, Xd.T @ y)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def ridge_gcv(X: np.ndarray, y: np.ndarray,
              lams=np.logspace(-9, 2, 23)) -> tuple[np.ndarray, float]:
    """Generalised cross-validation via SVD (thin chronological tails
    mis-select lambda on regime-switching series -- playbook gotcha)."""
    Xd = np.hstack([X, np.ones((len(X), 1))])
    U, s, Vt = np.linalg.svd(Xd, full_matrices=False)
    Uty = U.T @ y
    n = len(y)
    best = (np.inf, None)
    for lam in np.atleast_1d(lams):
        f = s ** 2 / (s ** 2 + lam)
        resid = y - U @ (f * Uty)
        dof = n - np.sum(f)
        g = np.sum(resid ** 2) / max(dof, 1e-9) ** 2 * n
        if g < best[0]:
            best = (g, lam)
    lam = float(best[1])
    w = Vt.T @ ((s / (s ** 2 + lam)) * Uty)
    return w, lam


def fit_eval(F_train, y_train, F_test, y_test, lam=None):
    """One harness for every model (fairness rule). lam=None -> GCV."""
    if lam is None:
        w, lam = ridge_gcv(F_train, y_train)
        Xd = np.hstack([F_test, np.ones((len(F_test), 1))])
        pred = Xd @ w
    else:
        w = ridge_fit(F_train, y_train, lam)
        pred = ridge_predict(F_test, w)
    resid = y_test - pred
    return float(np.mean(resid ** 2) / np.var(y_test)), pred, lam


# ------------------------------------------------------------ baselines
def delay_embed(x: np.ndarray, n_lags: int) -> np.ndarray:
    """Rows: [x_k, x_{k-1}, ..., x_{k-n_lags+1}] (zero-padded start).
    Multichannel x of shape (T, C) embeds each channel."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] == 1:
        x = x.T
    T, C = x.shape
    out = np.zeros((T, n_lags * C))
    for d in range(n_lags):
        out[d:, d * C:(d + 1) * C] = x[:T - d]
    return out


def nvar_features(x: np.ndarray, n_lags: int,
                  max_features: int | None = None) -> np.ndarray:
    """NVAR feature vector (Gauthier et al. 2021): constant + linear
    delay embedding + all unique quadratic monomials of it. Truncated to
    max_features columns (constant+linear first, then quadratics in
    lexicographic order) to honour the feature-count matching rule."""
    lin = delay_embed(x, n_lags)
    T, d = lin.shape
    iu = np.triu_indices(d)
    quad = lin[:, iu[0]] * lin[:, iu[1]]
    F = np.hstack([np.ones((T, 1)), lin, quad])
    if max_features is not None:
        F = F[:, :max_features]
    return F


class ESN:
    """Leaky echo state network, feature-count matched (nodes = quantum
    feature dimension). Standard dense reservoir, spectral radius rho."""

    def __init__(self, n_nodes: int, n_inputs: int = 1, rho: float = 0.9,
                 leak: float = 0.3, input_scale: float = 0.5,
                 seed: int = SEED):
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(n_nodes, n_nodes))
        eig = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = rho * W / eig
        self.Win = input_scale * rng.uniform(-1, 1, (n_nodes, n_inputs))
        self.leak = leak

    def run(self, inputs: np.ndarray, washout: int = 20) -> np.ndarray:
        x = np.atleast_2d(np.asarray(inputs, dtype=float))
        if x.shape[0] == 1:
            x = x.T
        T = x.shape[0]
        states = np.zeros((T, self.W.shape[0]))
        s = np.zeros(self.W.shape[0])
        for k in range(T):
            pre = self.W @ s + self.Win @ x[k]
            s = (1 - self.leak) * s + self.leak * np.tanh(pre)
            states[k] = s
        return states[washout:]


def persistence_pred(y: np.ndarray, horizon: int) -> np.ndarray:
    """y_hat(k + horizon) = y(k), aligned to the target index."""
    y = np.asarray(y)
    return y[:-horizon] if horizon > 0 else y.copy()


# ------------------------------------------------------------- anchors
def run_anchors(verbose: bool = True) -> bool:
    ok = True
    rng = np.random.default_rng(0)

    # ridge recovers a known linear map to 1e-10
    X = rng.normal(size=(300, 12))
    w_true = rng.normal(size=12)
    y = X @ w_true
    w = ridge_fit(X, y, lam=1e-12)
    err = float(np.max(np.abs(w[:-1] - w_true)))
    good = err < 1e-10
    ok &= good
    if verbose:
        print(f"  [{'PASS' if good else 'FAIL'}] ridge linear-map recovery: "
              f"{err:.1e}")

    # NVAR exactly recovers a known quadratic map to 1e-10
    T = 400
    x = rng.uniform(-1, 1, T)
    y = 0.7 * x - 0.4 * np.r_[0.0, x[:-1]] + 1.3 * x * np.r_[0.0, x[:-1]]
    F = nvar_features(x, n_lags=2)
    w = ridge_fit(F[2:], y[2:], lam=1e-12)
    pred = ridge_predict(F[2:], w)
    err = float(np.max(np.abs(pred - y[2:])))
    good = err < 1e-10
    ok &= good
    if verbose:
        print(f"  [{'PASS' if good else 'FAIL'}] NVAR quadratic-map "
              f"recovery: {err:.1e}")

    # NMSE of the mean predictor is exactly 1.000
    y = rng.normal(size=500)
    n = float(np.mean((y - y.mean()) ** 2) / np.var(y))
    good = abs(n - 1.0) < 1e-12
    ok &= good
    if verbose:
        print(f"  [{'PASS' if good else 'FAIL'}] NMSE(mean) = {n:.12f}")
    return ok


if __name__ == "__main__":
    import sys
    print("rfqrc_baselines anchors:")
    sys.exit(0 if run_anchors() else 1)
