"""memory_capacity.py -- linear memory capacity of a reservoir feature stream.

Reservoir-agnostic: everything here takes a *feature matrix* ``X`` of shape
``(T, F)`` -- one row of read-out features per input step -- together with the
scalar driving stream ``u`` that produced it. Nothing in this module knows or
cares whether those features came from a quantum reservoir, an echo-state
network, or a delay line, so it plugs into whatever driver the Quantathon stack
grows next.

The quantity computed is the linear memory function of Jaeger (2001),

    MF_d = corr^2( u^(k-d),  w_d . x^(k) ),

the squared correlation between the input delayed by ``d`` and the best *linear*
read-out of the reservoir state at step ``k``; the memory capacity is
``MC = sum_d MF_d``. Two disciplines separate this implementation from the
textbook one-liner, and both matter:

* **Held-out evaluation.** ``MF_d`` is fitted on a training window and scored on
  a later, disjoint window. In-sample squared correlation is an upward-biased
  estimate that inflates MC toward the feature count whenever ``F`` is not small
  compared with ``T`` -- the usual way a reservoir is accidentally reported as
  having more memory than it has. Pass ``train_frac=1.0`` for the in-sample
  convention when reproducing a paper that uses it.
* **A surrogate noise floor.** Finite samples give every delay a small positive
  ``MF_d``, so a naive sum accumulates a long tail of noise that grows with
  ``max_delay``. The floor is estimated by scoring shuffled targets -- which
  carry no recoverable memory by construction -- and taking a high quantile.
  ``MF_d`` below the floor is set to zero before summing. The quantile is 0.99
  rather than the more usual 0.95 because the floor is applied at every delay:
  a 0.95 floor lets through one spurious delay in twenty by construction, and
  MC sums them all.

The theoretical bound is ``MC <= rank(X)``, at most the number of features. A
reported MC near that bound on a scalar drive is a red flag, not a triumph:
check ``effective_rank`` before believing it.

Run the module directly to execute the validation anchors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MemoryCapacityResult",
    "memory_function",
    "linear_memory_capacity",
    "effective_rank",
    "delay_line_features",
    "leaky_integrator_features",
    "esn_features",
    "shuffled_features",
]


# ----------------------------------------------------------------------
# Ridge read-out
# ----------------------------------------------------------------------
def _ridge_weights(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge solution for the design matrix ``X`` with an appended intercept.

    The intercept column is *not* penalised: centring is not part of what the
    regulariser is meant to shrink, and penalising it biases MF_d downward for
    features with a large offset (``<Z_i>`` typically has one).
    """
    Xd = np.hstack([X, np.ones((len(X), 1))])
    penalty = np.eye(Xd.shape[1])
    penalty[-1, -1] = 0.0
    A = Xd.T @ Xd + lam * penalty
    return np.linalg.solve(A, Xd.T @ y)


def _ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def _squared_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Squared Pearson correlation, guarded against a constant prediction.

    A read-out that collapses to a constant has zero recoverable memory, which
    is the right answer; ``np.corrcoef`` would return NaN there.
    """
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    if sd_true < 1e-12 or sd_pred < 1e-12:
        return 0.0
    c = float(np.corrcoef(y_true, y_pred)[0, 1])
    return c * c


# ----------------------------------------------------------------------
# Memory function
# ----------------------------------------------------------------------
def _fit_score(X: np.ndarray, y: np.ndarray, n_train: int,
               lam: float) -> float:
    """Fit on the first ``n_train`` rows, score squared correlation on the rest."""
    w = _ridge_weights(X[:n_train], y[:n_train], lam)
    if n_train >= len(y):                      # in-sample convention
        return _squared_corr(y, _ridge_predict(X, w))
    return _squared_corr(y[n_train:], _ridge_predict(X[n_train:], w))


def memory_function(X: np.ndarray, u: np.ndarray, max_delay: int = 25,
                    washout: int | None = None, train_frac: float = 0.7,
                    lam: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Linear memory function ``MF_d`` for ``d = 1 .. max_delay``.

    Parameters
    ----------
    X : ndarray, shape (T, F)
        Reservoir features, one row per input step.
    u : ndarray, shape (T,)
        The scalar driving stream that produced ``X``. Memory capacity is only
        meaningful for an i.i.d. drive: temporal correlation in ``u`` lets a
        memoryless read-out "predict" past inputs from the present one, which
        inflates MF at every delay. :func:`linear_memory_capacity` warns when
        the drive looks autocorrelated.
    max_delay : int
        Largest delay probed. MC is a sum over delays, so this must be pushed
        out until ``MF_d`` has decayed into the noise floor -- otherwise MC is
        truncated. :func:`linear_memory_capacity` reports whether it has.
    washout : int, optional
        Leading steps discarded so the reservoir has forgotten its initial
        state. Defaults to ``max(2 * max_delay, 100)``, and is clamped to at
        least ``max_delay`` so that every delayed target is well defined
        without wrap-around.
    train_frac : float
        Fraction of the post-washout window used for fitting; the remainder is
        the held-out scoring window. ``1.0`` gives the in-sample convention.
    lam : float
        Ridge penalty.

    Returns
    -------
    delays : ndarray of int, shape (max_delay,)
        The delays probed, ``1 .. max_delay``.
    mf : ndarray of float, shape (max_delay,)
        Raw (un-thresholded) ``MF_d``, each in ``[0, 1]``.
    """
    X = np.asarray(X, dtype=float)
    u = np.asarray(u, dtype=float).ravel()
    d_max = int(max_delay)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (T, F); got shape {X.shape}.")
    if len(X) != len(u):
        raise ValueError(
            f"X has {len(X)} rows but u has {len(u)} steps; they must match.")
    if d_max < 1:
        raise ValueError("max_delay must be at least 1.")

    wash = max(2 * d_max, 100) if washout is None else int(washout)
    wash = max(wash, d_max)                    # delayed targets must exist
    n_eval = len(u) - wash
    if n_eval < 4 * X.shape[1] + 10:
        raise ValueError(
            f"Only {n_eval} steps survive the washout of {wash} for "
            f"{X.shape[1]} features. Ridge would interpolate and MF_d would be "
            "meaningless; use a longer drive or fewer features.")

    Xe = X[wash:]
    n_train = len(Xe) if train_frac >= 1.0 else int(train_frac * len(Xe))
    if n_train < X.shape[1] + 2:
        raise ValueError(
            f"Training window of {n_train} rows is too short to fit "
            f"{X.shape[1]} features.")

    delays = np.arange(1, d_max + 1)
    mf = np.empty(d_max)
    for i, d in enumerate(delays):
        # y[k] = u[k - d] over the evaluation window -- an explicit shift, not
        # np.roll, which would wrap the tail of u round to the front.
        y = u[wash - d: len(u) - d]
        mf[i] = _fit_score(Xe, y, n_train, lam)
    return delays, mf


# ----------------------------------------------------------------------
# Surrogate noise floor
# ----------------------------------------------------------------------
def _noise_floor(X: np.ndarray, u: np.ndarray, max_delay: int, washout: int,
                 train_frac: float, lam: float, n_surrogate: int,
                 quantile: float, rng: np.random.Generator) -> float:
    """High quantile of ``MF`` scored against targets with the memory destroyed.

    The surrogate target is a random permutation of the delayed input: same
    marginal distribution, same length, no temporal relationship to ``X``. Any
    ``MF`` it earns is finite-sample overfitting, so its upper quantile is the
    level a genuine ``MF_d`` must clear.
    """
    Xe = X[washout:]
    n_train = len(Xe) if train_frac >= 1.0 else int(train_frac * len(Xe))
    scores = np.empty(n_surrogate)
    for s in range(n_surrogate):
        d = int(rng.integers(1, max_delay + 1))
        y = u[washout - d: len(u) - d].copy()
        rng.shuffle(y)
        scores[s] = _fit_score(Xe, y, n_train, lam)
    return float(np.quantile(scores, quantile))


@dataclass
class MemoryCapacityResult:
    """Outcome of a linear memory-capacity measurement.

    Attributes
    ----------
    mc : float
        Memory capacity: the sum of ``mf_thresholded``. The headline number.
    mc_raw : float
        Sum of the un-thresholded ``mf``, for reference. The gap between this
        and ``mc`` is how much of the naive figure was noise.
    delays, mf, mf_thresholded : ndarray
        The delay grid, the raw memory function, and the memory function with
        sub-floor entries zeroed.
    noise_floor : float
        Surrogate-derived level below which ``MF_d`` is not distinguishable
        from overfitting.
    n_features : int
        Feature count.
    rank : int
        Numerical rank of the (centred) feature matrix. This is the hard bound:
        ``mc <= rank``, always. It is at most ``n_features`` and drops below it
        when features are exactly redundant.
    effective_rank : float
        Participation ratio of the feature covariance -- how many feature
        directions carry appreciable variance. A *soft* concentration
        diagnostic, not a bound: strongly correlated features can have an
        effective rank near 1 while still spanning a full-rank space, so ``mc``
        may legitimately exceed it. Compare ``mc`` against ``rank``; read
        ``effective_rank`` as a warning about exponential concentration.
    saturated : bool
        True when ``MF`` at the largest delay is still above the noise floor,
        i.e. the sum was truncated and ``mc`` is an underestimate.
    warnings : list of str
        Conditions that make ``mc`` hard to interpret; empty when clean.
    """

    mc: float
    mc_raw: float
    delays: np.ndarray
    mf: np.ndarray
    mf_thresholded: np.ndarray
    noise_floor: float
    n_features: int
    rank: int
    effective_rank: float
    saturated: bool
    warnings: list[str]

    def summary(self) -> str:
        """One-block human-readable report, including any caveats."""
        lines = [
            f"MC = {self.mc:.3f}   (raw sum {self.mc_raw:.3f}, "
            f"floor {self.noise_floor:.4f})",
            f"features = {self.n_features}, rank = {self.rank}, "
            f"effective rank = {self.effective_rank:.2f}",
            f"MF_1 = {self.mf[0]:.3f}, MF_{self.delays[-1]} = "
            f"{self.mf[-1]:.4f}",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def linear_memory_capacity(X: np.ndarray, u: np.ndarray, max_delay: int = 25,
                           washout: int | None = None, train_frac: float = 0.7,
                           lam: float = 1e-6, n_surrogate: int = 200,
                           quantile: float = 0.99,
                           seed: int | None = 0) -> MemoryCapacityResult:
    """Linear memory capacity of ``X`` under drive ``u``, with a noise floor.

    Thin orchestration over :func:`memory_function`: it adds the surrogate
    floor, the effective-rank diagnostic, and the checks that catch the three
    ways this measurement is usually misread -- an autocorrelated drive, a
    delay grid that truncates the sum, and an MC that has run up against the
    feature count.

    Parameters are as in :func:`memory_function`, plus ``n_surrogate`` /
    ``quantile`` controlling the floor estimate and ``seed`` for its
    reproducibility.
    """
    X = np.asarray(X, dtype=float)
    u = np.asarray(u, dtype=float).ravel()
    d_max = int(max_delay)

    delays, mf = memory_function(X, u, max_delay=d_max, washout=washout,
                                 train_frac=train_frac, lam=lam)

    wash = max(2 * d_max, 100) if washout is None else int(washout)
    wash = max(wash, d_max)
    rng = np.random.default_rng(seed)
    floor = _noise_floor(X, u, d_max, wash, train_frac, lam,
                         n_surrogate, quantile, rng)

    mf_thr = np.where(mf > floor, mf, 0.0)
    mc = float(mf_thr.sum())
    Xe = X[wash:]
    eff_rank = effective_rank(Xe)
    rank = int(np.linalg.matrix_rank(Xe - Xe.mean(axis=0, keepdims=True)))

    notes: list[str] = []
    # An autocorrelated drive leaks present-into-past and inflates every MF_d.
    if len(u) > 2:
        rho1 = abs(float(np.corrcoef(u[:-1], u[1:])[0, 1]))
        if rho1 > 0.1:
            notes.append(
                f"drive is autocorrelated (lag-1 |rho| = {rho1:.2f}); MC is "
                "only well defined for an i.i.d. drive and is inflated here")
    saturated = bool(mf[-1] > floor)
    if saturated:
        notes.append(
            f"MF is still above the noise floor at d = {d_max}; the sum is "
            "truncated, so MC is a lower bound -- raise max_delay")
    # MC <= rank(X) is the hard bound; sitting near it means the feature space,
    # not the dynamics, is what limits the measurement.
    if rank > 0 and mc > 0.9 * rank:
        notes.append(
            f"MC ({mc:.2f}) is close to its bound rank(X) = {rank}; the "
            "read-out is memory-saturated, so MC reflects the size of the "
            "feature space rather than the reservoir's dynamics")
    # Separately: features that carry variance in only a few directions are the
    # signature of exponential concentration, whatever MC came out as.
    if X.shape[1] >= 4 and eff_rank < 0.25 * X.shape[1]:
        notes.append(
            f"features are concentrated (effective rank {eff_rank:.2f} of "
            f"{X.shape[1]}); most read-out directions carry little "
            "independent variance")

    return MemoryCapacityResult(
        mc=mc, mc_raw=float(mf.sum()), delays=delays, mf=mf,
        mf_thresholded=mf_thr, noise_floor=floor, n_features=X.shape[1],
        rank=rank, effective_rank=eff_rank, saturated=saturated,
        warnings=notes)


# ----------------------------------------------------------------------
# Feature-space diagnostic
# ----------------------------------------------------------------------
def effective_rank(X: np.ndarray) -> float:
    """Participation ratio of the feature covariance spectrum.

    ``PR = (sum_i lambda_i)^2 / sum_i lambda_i^2`` for the eigenvalues of
    ``cov(X)``: the number of feature directions that actually carry variance.
    Equals ``F`` for ``F`` equally-weighted independent features and collapses
    toward 1 when the features are redundant -- which is the signature of the
    exponential-concentration failure mode, where a nominally large quantum
    feature vector carries almost no independent information.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (T, F); got shape {X.shape}.")
    Xc = X - X.mean(axis=0, keepdims=True)
    evals = np.linalg.eigvalsh(np.cov(Xc, rowvar=False, ddof=1)
                              if X.shape[1] > 1
                              else np.atleast_2d(np.var(Xc, ddof=1)))
    evals = np.clip(evals, 0.0, None)
    total = evals.sum()
    if total < 1e-15:
        return 0.0
    return float(total ** 2 / np.sum(evals ** 2))


# ----------------------------------------------------------------------
# Reference feature streams: analytic anchors and controls
# ----------------------------------------------------------------------
def delay_line_features(u: np.ndarray, length: int) -> np.ndarray:
    """Perfect tapped delay line: column ``j`` is ``u`` delayed by ``j``.

    The analytic anchor for the whole module. Its memory function is exactly 1
    for ``d < length`` and 0 beyond, so ``MC = length`` -- a delay line stores
    its taps perfectly and nothing else.
    """
    u = np.asarray(u, dtype=float).ravel()
    L = int(length)
    X = np.zeros((len(u), L))
    for j in range(L):
        X[j:, j] = u[:len(u) - j]
    return X


def leaky_integrator_features(u: np.ndarray, leaks=(0.1, 0.3, 0.6, 0.9)
                              ) -> np.ndarray:
    """Bank of linear leaky integrators, ``x <- (1 - a) x + a u``.

    Fading memory with no nonlinearity: each column has an exponential memory
    kernel of time constant ``1 / a``. Gives a smooth, monotonically decaying
    memory function -- the shape a healthy reservoir should resemble.
    """
    u = np.asarray(u, dtype=float).ravel()
    leaks = np.asarray(leaks, dtype=float)
    X = np.empty((len(u), len(leaks)))
    x = np.zeros(len(leaks))
    for k, uk in enumerate(u):
        x = (1.0 - leaks) * x + leaks * uk
        X[k] = x
    return X


def esn_features(u: np.ndarray, n_nodes: int = 20, spectral_radius: float = 0.9,
                 input_scale: float = 1.0, leak: float = 0.3,
                 seed: int | None = 11) -> np.ndarray:
    """Leaky echo-state network features -- the size-matched classical baseline.

    Any memory-capacity claim for a quantum reservoir has to be read against an
    ESN whose ``n_nodes`` equals the quantum feature count, so that both
    read-outs train the same number of weights. Quantum reservoirs commonly
    lose this comparison on linear memory; that is a result to report, not to
    hide.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_nodes, n_nodes))
    W *= spectral_radius / max(abs(np.linalg.eigvals(W)))
    W_in = rng.uniform(-input_scale, input_scale, size=n_nodes)

    u = np.asarray(u, dtype=float).ravel()
    X = np.empty((len(u), n_nodes))
    x = np.zeros(n_nodes)
    for k, uk in enumerate(u):
        x = (1 - leak) * x + leak * np.tanh(W @ x + W_in * float(uk))
        X[k] = x
    return X


def shuffled_features(X: np.ndarray, seed: int | None = 0) -> np.ndarray:
    """Row-permuted copy of ``X``: same marginals, temporal structure destroyed.

    The null control. Whatever MC this scores is the measurement's own floor;
    a reservoir whose MC is not clearly above it has demonstrated no memory.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    return X[rng.permutation(len(X))]


# ----------------------------------------------------------------------
# Validation anchors
# ----------------------------------------------------------------------
def run_validation_suite() -> None:
    """Check the estimator against cases whose memory capacity is known."""
    rng = np.random.default_rng(0)
    T = 4000
    u = rng.uniform(0.0, 1.0, size=T)

    # 1. Delay line of length L: MF_d = 1 for d < L, 0 beyond, so MC = L - 1
    #    over delays 1..max_delay (d = 0 is not part of the sum).
    L = 8
    res = linear_memory_capacity(delay_line_features(u, L), u, max_delay=20)
    assert abs(res.mc - (L - 1)) < 0.05, res.mc
    assert np.all(res.mf[:L - 1] > 0.99), res.mf[:L - 1]
    assert np.all(res.mf_thresholded[L - 1:] == 0.0), res.mf_thresholded[L - 1:]
    assert not res.saturated

    # 2. Memoryless features (i.i.d. noise, no dependence on u): MC ~ 0 once
    #    the surrogate floor has removed the finite-sample tail.
    noise = rng.normal(size=(T, 8))
    res_noise = linear_memory_capacity(noise, u, max_delay=20)
    assert res_noise.mc < 0.1, res_noise.mc
    assert res_noise.mc_raw > res_noise.mc      # the floor did remove something

    # 3. Instantaneous features (u itself, no delay): d = 0 only, so MC ~ 0
    #    while the *fit* is perfect -- separates memory from expressivity.
    res_inst = linear_memory_capacity(u.reshape(-1, 1), u, max_delay=20)
    assert res_inst.mc < 0.1, res_inst.mc

    # 4. Leaky integrators: finite, monotonically decaying memory function.
    res_leak = linear_memory_capacity(leaky_integrator_features(u), u,
                                      max_delay=30)
    assert 0.5 < res_leak.mc < 4.0, res_leak.mc
    assert res_leak.mf[0] > res_leak.mf[10] > res_leak.mf[-1], res_leak.mf

    # 5. The bound MC <= rank(X) holds for an ESN, and the shuffled control of
    #    the same features scores essentially nothing.
    esn = esn_features(u, n_nodes=20, seed=11)
    res_esn = linear_memory_capacity(esn, u, max_delay=30)
    assert res_esn.mc <= res_esn.rank + 1e-9, (res_esn.mc, res_esn.rank)
    assert res_esn.rank <= res_esn.n_features
    res_ctrl = linear_memory_capacity(shuffled_features(esn), u, max_delay=30)
    assert res_ctrl.mc < 0.2 * res_esn.mc, (res_ctrl.mc, res_esn.mc)

    # 6. Effective rank: L independent delay-line taps -> PR ~ L; L copies of
    #    one tap -> PR ~ 1.
    assert abs(effective_rank(delay_line_features(u, 6)[50:]) - 6) < 0.4
    repeated = np.repeat(u.reshape(-1, 1), 6, axis=1)
    assert abs(effective_rank(repeated) - 1.0) < 0.05

    # 7. Held-out scoring is the stricter convention: with many features and a
    #    short drive, the in-sample estimate inflates MC.
    short_u = rng.uniform(0.0, 1.0, size=700)
    wide = rng.normal(size=(700, 40))
    mc_out = linear_memory_capacity(wide, short_u, max_delay=10).mc_raw
    mc_in = linear_memory_capacity(wide, short_u, max_delay=10,
                                   train_frac=1.0).mc_raw
    assert mc_in > mc_out, (mc_in, mc_out)

    # 8. Guard rails fire rather than returning a meaningless number:
    #    length mismatch, too few samples per feature, a degenerate delay grid,
    #    and a 1-D feature array.
    guards = {
        "length mismatch": lambda: memory_function(np.zeros((100, 2)), u),
        "under-determined": lambda: memory_function(np.zeros((T, 1200)), u),
        "max_delay < 1": lambda: memory_function(np.zeros((T, 2)), u,
                                                 max_delay=0),
        "1-D features": lambda: memory_function(u, u),
    }
    for name, bad in guards.items():
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError(f"expected a ValueError for: {name}")

    print("All memory-capacity validation anchors passed.")


def main() -> int:
    run_validation_suite()

    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, size=4000)

    print("\n== linear memory capacity of reference feature streams ==")
    streams = {
        "delay line (L=8)": delay_line_features(u, 8),
        "leaky integrators": leaky_integrator_features(u),
        "ESN (20 nodes)": esn_features(u, n_nodes=20),
        "ESN, shuffled control": shuffled_features(esn_features(u, n_nodes=20)),
        "i.i.d. noise": rng.normal(size=(len(u), 20)),
    }
    for name, X in streams.items():
        res = linear_memory_capacity(X, u, max_delay=30)
        print(f"\n-- {name}")
        print(res.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
