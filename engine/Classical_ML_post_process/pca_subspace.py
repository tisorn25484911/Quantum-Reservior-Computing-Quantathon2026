"""Principal-component *subspace* anomaly detection for multivariate
time-series.

Where ``egads_pc_cluster`` clusters *whole series* by their Table-1 features
(EGADS Section 3.3) to triage structurally-odd series, this module is the
complementary, per-timestamp multivariate detector: it models the joint
correlation structure of the channels with PCA and flags timestamps whose
multichannel state departs from that structure. It is the standard "PC
detection" for correlated multivariate signals and is what actually resolves
*co-exceedance / compound* anomalies (all channels excited together), which the
level-invariant feature detector cannot see.

Method and references (used strictly, not invented):

    * Lakhina, Crovella & Diot, "Diagnosing Network-Wide Traffic Anomalies",
      SIGCOMM 2004 -- the normal/anomalous subspace decomposition: split the PCA
      space into a low-dimensional *normal subspace* (leading PCs) and a
      *residual subspace* (the rest); an anomaly projects energy into the
      residual subspace.
    * Jackson & Mudholkar, "Control Procedures for Residuals Associated with
      Principal Component Analysis", Technometrics 21(3), 1979 -- the squared
      prediction error (SPE / Q-statistic) and its parametric upper control
      limit Q_alpha.
    * Hotelling's T^2 -- Mahalanobis distance inside the normal subspace, the
      classic multivariate SPC statistic for excursions *along* the principal
      modes (a compound co-exceedance is a large T^2 along the dominant mode).

Two complementary scores are produced per timestamp:

    T2  (Hotelling)  large  -> unusually large excursion along the normal modes
                              (e.g. every driver in its alarming tail together)
    SPE (Q-statistic) large -> the channels violate their normal correlation
                              (a relationship that usually holds has broken)

Both are fit on the training span only (leak-free) and thresholded with either
their parametric control limits or the EGADS Section 4.1 rules (K-sigma /
empirical density quantile).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def lag_embed(X, emb=1, lag=1):
    """Stack `emb` lagged copies of a (T, C) series -> (T', C*emb).

    emb=1 returns the instantaneous multivariate state (the default). emb>1
    augments each timestamp with its recent history so the PCA also captures
    short-run temporal correlation, closer in spirit to Lakhina's use of
    time-windows. Row t of the output holds [x_t, x_{t-lag}, ...]; the first
    (emb-1)*lag timestamps have no full history and are dropped, so the returned
    `offset` tells callers how to align labels.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if emb <= 1:
        return X, 0
    T = X.shape[0]
    blocks = [X[(emb - 1 - i) * lag: T - i * lag] for i in range(emb)]
    return np.hstack(blocks), (emb - 1) * lag


@dataclass
class PCASubspaceDetector:
    """PCA normal/residual subspace detector with SPE (Q) and Hotelling T^2.

    Parameters
    ----------
    var_normal : float
        Fraction of variance defining the *normal* subspace dimension k; the
        remaining PCs form the residual subspace. (E.g. 0.9 -> keep enough PCs
        for 90% of the training variance.)
    k : int or None
        Explicit normal-subspace dimension; overrides ``var_normal`` when set.
    emb, lag : int
        Optional lag-embedding of the multivariate state (see ``lag_embed``).
    alpha : float
        Tail probability for the parametric control limits (0.01 -> 99% limit).
    k_sigma : float
        K for the EGADS Section-4.1 K-sigma rule when that thresholding is used.
    """

    var_normal: float = 0.9
    k: int | None = None
    emb: int = 1
    lag: int = 1
    alpha: float = 0.01
    k_sigma: float = 3.0

    # fitted state
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)
    components_: np.ndarray = field(default=None, repr=False)   # (m, m) V
    eigvals_: np.ndarray = field(default=None, repr=False)      # variances
    k_: int = field(default=None, repr=False)
    offset_: int = field(default=0, repr=False)
    spe_limit_: float = field(default=None, repr=False)
    t2_limit_: float = field(default=None, repr=False)
    train_spe_: np.ndarray = field(default=None, repr=False)
    train_t2_: np.ndarray = field(default=None, repr=False)

    # ----------------------------------------------------------------- fitting
    def fit(self, X, n_train=None):
        """Fit PCA + control limits on the training span.

        X : (T, C) multivariate series. n_train : rows [0, n_train) used to
        estimate the normal model; defaults to all rows.
        """
        Xe, self.offset_ = lag_embed(X, self.emb, self.lag)
        n_train = Xe.shape[0] if n_train is None else n_train - self.offset_
        n_train = max(2, min(n_train, Xe.shape[0]))
        tr = Xe[:n_train]

        self.mean_ = tr.mean(0)
        self.std_ = tr.std(0)
        self.std_[self.std_ < 1e-12] = 1.0
        Ztr = (tr - self.mean_) / self.std_

        # PCA via SVD of the centred training block
        Zc = Ztr - Ztr.mean(0)
        _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        self.components_ = Vt.T                       # columns are PCs
        self.eigvals_ = (S ** 2) / (n_train - 1)      # variance per PC

        # normal-subspace dimension
        if self.k is not None:
            self.k_ = int(self.k)
        else:
            ratio = np.cumsum(self.eigvals_) / self.eigvals_.sum()
            self.k_ = int(np.searchsorted(ratio, self.var_normal) + 1)
        self.k_ = max(1, min(self.k_, self.components_.shape[1] - 0))

        # parametric limits from the training statistics
        self._set_limits(Ztr, n_train)
        s = self.score(X)                             # cache train scores
        self.train_spe_, self.train_t2_ = s["spe"], s["t2"]
        return self

    # --------------------------------------------------------------- scoring
    def _project(self, Z):
        proj = Z @ self.components_                    # scores on all PCs
        return proj

    def score(self, X):
        """Return per-timestamp {'spe', 't2'} aligned to X's timeline.

        Rows before the embedding offset (no full history) get score 0.
        """
        Xe, off = lag_embed(X, self.emb, self.lag)
        Z = (Xe - self.mean_) / self.std_
        proj = self._project(Z)
        k = self.k_
        # Hotelling T^2 on the normal subspace (Mahalanobis along leading PCs)
        lam = self.eigvals_[:k]
        t2 = ((proj[:, :k] ** 2) / lam).sum(1)
        # SPE / Q-statistic = energy left in the residual subspace
        spe = (proj[:, k:] ** 2).sum(1)
        # re-align to the original timeline
        T = np.asarray(X).shape[0]
        full_spe = np.zeros(T)
        full_t2 = np.zeros(T)
        full_spe[off:off + spe.size] = spe
        full_t2[off:off + t2.size] = t2
        return {"spe": full_spe, "t2": full_t2}

    # ----------------------------------------------- parametric control limits
    def _set_limits(self, Ztr, n_train):
        proj = self._project(Ztr)
        k = self.k_
        # --- Jackson-Mudholkar Q upper control limit (residual eigenvalues) ---
        resid_eig = self.eigvals_[k:]
        if resid_eig.size == 0:
            self.spe_limit_ = np.inf
        else:
            t1 = resid_eig.sum()
            t2s = (resid_eig ** 2).sum()
            t3 = (resid_eig ** 3).sum()
            h0 = 1.0 - (2.0 * t1 * t3) / (3.0 * t2s ** 2) if t2s > 0 else 1.0
            c = stats.norm.ppf(1.0 - self.alpha)
            if t1 > 0 and t2s > 0:
                term = (c * np.sqrt(2.0 * t2s * h0 ** 2) / t1
                        + 1.0 + t2s * h0 * (h0 - 1.0) / (t1 ** 2))
                self.spe_limit_ = float(t1 * term ** (1.0 / h0))
            else:
                self.spe_limit_ = np.inf
        # --- Hotelling T^2 limit via the F-distribution (SPC standard) ---
        n = n_train
        if n > k + 1:
            f = stats.f.ppf(1.0 - self.alpha, k, n - k)
            self.t2_limit_ = float(k * (n - 1) / (n - k) * f)
        else:
            t2_train = ((proj[:, :k] ** 2) / self.eigvals_[:k]).sum(1)
            self.t2_limit_ = float(np.quantile(t2_train, 1.0 - self.alpha))

    # --------------------------------------------------------------- flagging
    def predict(self, X, which="t2", rule="limit"):
        """Boolean anomaly flags for X.

        which : "t2" (excursion along modes / co-exceedance) or
                "spe" (correlation-structure break), or "either" (logical OR).
        rule  : "limit"  -> parametric control limit (Jackson-Mudholkar / F),
                "ksigma" -> EGADS Section-4.1 K-sigma on the train score dist,
                "density"-> upper (1-alpha) empirical quantile of train scores.
        """
        s = self.score(X)
        if which == "either":
            return (self._flag(s["t2"], "t2", rule)
                    | self._flag(s["spe"], "spe", rule))
        return self._flag(s[which], which, rule)

    def _flag(self, score, which, rule):
        if rule == "limit":
            thr = self.t2_limit_ if which == "t2" else self.spe_limit_
        elif rule == "ksigma":
            base = self.train_t2_ if which == "t2" else self.train_spe_
            thr = base.mean() + self.k_sigma * base.std()
        elif rule == "density":
            base = self.train_t2_ if which == "t2" else self.train_spe_
            thr = np.quantile(base, 1.0 - self.alpha)
        else:
            raise ValueError(rule)
        return score > thr

    def explained_variance_ratio(self):
        return self.eigvals_ / self.eigvals_.sum()
