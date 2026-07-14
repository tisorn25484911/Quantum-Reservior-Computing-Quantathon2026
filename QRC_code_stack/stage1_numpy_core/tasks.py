"""Benchmark tasks and input streams for quantum reservoir computing.

All tasks follow the conventions of the review's anchor paper (Kobayashi &
Motome, PRL 136, 040602, 2026) and the classical RC literature:

* The *driving* input ``u`` is a scalar stream in ``[0, 1]`` (a uniform random
  stream for NARMA/STM/parity; a synthetic diurnal signal for the solar task).
* A *task* maps that input stream to a scalar target stream ``y`` of the same
  length, where ``y[k]`` is the value the trained read-out should predict at
  step ``k`` from the reservoir features at step ``k``.

These are deliberately small, self-contained, and dependency-free (NumPy only).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "uniform_input",
    "narma",
    "short_term_memory",
    "parity_check",
    "synthetic_solar",
    "mackey_glass",
]


def uniform_input(n_steps: int, low: float = 0.0, high: float = 0.2,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Uniform i.i.d. driving stream.

    The default range ``[0, 0.2]`` matches the anchor paper's NARMA driving
    range (chosen to keep the recursive NARMA target from diverging). The stream
    is *not* rescaled here; encodings rescale to ``[0, 1]`` at injection time.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    low, high : float
        Range of the uniform distribution.
    rng : numpy.random.Generator, optional
        Random generator; a fresh default generator is used if omitted.
    """
    rng = np.random.default_rng() if rng is None else rng
    return rng.uniform(low, high, size=int(n_steps))


def narma(u: np.ndarray, order: int = 2,
          coeffs: tuple[float, float, float, float] = (0.3, 0.05, 1.5, 0.1)
          ) -> np.ndarray:
    r"""Nonlinear auto-regressive moving-average (NARMA) target.

    Implements the recursion used in the anchor paper,

    .. math::
        \bar y^{(k+1)} = \alpha\,\bar y^{(k)}
        + \beta\,\bar y^{(k)}\sum_{j=0}^{n-1}\bar y^{(k-j)}
        + \gamma\,u^{(k-n+1)}u^{(k)} + \delta,

    with ``(alpha, beta, gamma, delta)`` defaulting to ``(0.3, 0.05, 1.5, 0.1)``.
    NARMA needs *both* memory and nonlinearity, which is why it is the workhorse
    benchmark for the memory--nonlinearity trade-off.

    Returns an array of the same length as ``u`` (the first ``order`` entries are
    warmup and should fall inside the washout window).
    """
    alpha, beta, gamma, delta = coeffs
    n = int(order)
    y = np.zeros_like(u, dtype=float)
    for k in range(len(u) - 1):
        if k < n - 1:
            # not enough history yet; hold at the running value
            y[k + 1] = y[k]
            continue
        window = np.sum(y[k - n + 1: k + 1])
        y[k + 1] = (alpha * y[k]
                    + beta * y[k] * window
                    + gamma * u[k - n + 1] * u[k]
                    + delta)
    return y


def short_term_memory(u: np.ndarray, delay: int = 1) -> np.ndarray:
    """Short-term memory (STM) target: reproduce the input delayed by ``delay``.

    A pure *linear-memory* probe. ``y[k] = u[k - delay]`` (zero for ``k < delay``).
    """
    d = int(delay)
    y = np.zeros_like(u, dtype=float)
    if d < len(u):
        y[d:] = u[:len(u) - d]
    return y


def parity_check(u: np.ndarray, window: int = 3, threshold: float | None = None
                 ) -> np.ndarray:
    """Parity-check target: parity of the binarised recent inputs.

    A pure *nonlinearity* probe. The continuous input is binarised about
    ``threshold`` (its median by default), then the target is the XOR (parity)
    of the last ``window`` bits. Returns values in ``{0, 1}``.
    """
    thr = np.median(u) if threshold is None else threshold
    bits = (u > thr).astype(int)
    w = int(window)
    y = np.zeros_like(u, dtype=float)
    for k in range(len(u)):
        lo = max(0, k - w + 1)
        y[k] = np.sum(bits[lo: k + 1]) % 2
    return y


def synthetic_solar(n_steps: int, period: int = 24, noise: float = 0.05,
                    seed: int | None = 0) -> np.ndarray:
    """Synthetic 'solar generation' series in ``[0, 1]``.

    A stand-in for a clean-energy / occupational-health diurnal signal: a
    rectified diurnal sinusoid with a slow weather-like envelope and additive
    noise, clipped to ``[0, 1]``. Deterministic given ``seed`` so figures are
    reproducible.

    Parameters
    ----------
    n_steps : int
        Length of the series.
    period : int
        Steps per 'day' (diurnal period).
    noise : float
        Standard deviation of additive Gaussian noise.
    seed : int, optional
        Seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_steps))
    diurnal = np.clip(np.sin(2 * np.pi * t / period), 0.0, None)  # daylight only
    # slow multiplicative weather envelope (cloud cover drifting over ~weeks)
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * t / (period * 7) + 0.5)
    signal = diurnal * envelope
    signal = signal + rng.normal(0.0, noise, size=signal.shape)
    return np.clip(signal, 0.0, 1.0)


def mackey_glass(n_steps: int, tau: int = 17, dt: float = 1.0, beta: float = 0.2,
                 gamma: float = 0.1, n_exp: int = 10, seed: int | None = 0,
                 discard: int = 250) -> np.ndarray:
    """Mackey--Glass chaotic series, normalised to ``[0, 1]``.

    Generated by Euler integration of the delay differential equation

        x'(t) = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t),

    which is chaotic for ``tau = 17``. Used for the closed-loop / long-horizon
    forecasting demonstration. The first ``discard`` samples are dropped as
    transient.
    """
    rng = np.random.default_rng(seed)
    history_len = int(tau / dt)
    total = int(n_steps) + discard + history_len
    x = np.empty(total, dtype=float)
    x[:history_len] = 1.2 + 0.2 * (rng.random(history_len) - 0.5)
    for k in range(history_len, total):
        x_tau = x[k - history_len]
        x[k] = x[k - 1] + dt * (beta * x_tau / (1.0 + x_tau ** n_exp)
                                - gamma * x[k - 1])
    x = x[history_len + discard:]
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x


# --------------------------------------------------------------------------
# Hidden-Markov-model (HMM) processes
#
# These give the QHMM reservoir a fair home turf: sequences generated by a
# finite-memory stochastic process, where the natural benchmarks are
# next-symbol prediction and hidden-state filtering. Both require integrating
# information over the observation history -- exactly what a reservoir's fading
# memory provides.
# --------------------------------------------------------------------------

def hmm_sequence(T: np.ndarray, E: np.ndarray, n_steps: int,
                 seed: int | None = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample an HMM: hidden chain ``T``, emissions ``E``.

    Parameters
    ----------
    T : ndarray, shape (S, S)
        Hidden-state transition matrix, ``T[i, j] = P(s' = j | s = i)``.
    E : ndarray, shape (S, A)
        Emission matrix, ``E[i, a] = P(obs = a | s = i)``.
    n_steps : int
        Length of the sampled sequence.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    obs : ndarray of int, shape (n_steps,)
        Observed symbol sequence (values in ``0..A-1``).
    hidden : ndarray of int, shape (n_steps,)
        Hidden-state sequence (values in ``0..S-1``), for diagnostics only --
        a model never sees it.
    """
    rng = np.random.default_rng(seed)
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=float)
    S = T.shape[0]
    # start from the stationary distribution of T
    evals, evecs = np.linalg.eig(T.T)
    k = int(np.argmin(np.abs(evals - 1.0)))
    pi = np.real(evecs[:, k]); pi = np.abs(pi) / np.abs(pi).sum()
    s = rng.choice(S, p=pi)
    obs = np.empty(int(n_steps), dtype=int)
    hid = np.empty(int(n_steps), dtype=int)
    for k_ in range(int(n_steps)):
        hid[k_] = s
        obs[k_] = rng.choice(E.shape[1], p=E[s])
        s = rng.choice(S, p=T[s])
    return obs, hid


def heralding_coin(n_steps: int, p_head: float = 0.8, p_stay: float = 0.9,
                   seed: int | None = 0) -> tuple[np.ndarray, np.ndarray]:
    """The 'heralding coin': a 2-state HMM with persistent bias.

    A coin whose bias flips between head-heavy (``p_head``) and tail-heavy
    (``1 - p_head``) according to a sticky 2-state hidden chain
    (stay-probability ``p_stay``). Predicting the next flip well requires
    inferring the current hidden bias from recent observations -- a clean,
    minimal memory task. Returns ``(obs, hidden)`` as in :func:`hmm_sequence`.
    """
    T = np.array([[p_stay, 1 - p_stay],
                  [1 - p_stay, p_stay]])
    E = np.array([[p_head, 1 - p_head],
                  [1 - p_head, p_head]])
    return hmm_sequence(T, E, n_steps, seed=seed)


def hmm_filter_targets(T: np.ndarray, E: np.ndarray, obs: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Exact Bayesian filter targets for an HMM observation sequence.

    Runs the forward (filtering) recursion and returns

    * ``p_next``: shape ``(n,)`` -- the optimal one-step-ahead probability of
      observing symbol ``1`` given the history up to and including step ``k``
      (for binary alphabets; for larger alphabets the probability of symbol 1).
    * ``p_state``: shape ``(n,)`` -- the filtered probability of hidden state 1
      given the history (binary hidden chains; first excited state otherwise).

    These are the *information-theoretic optimum* any causal predictor can
    achieve, so a reservoir's NMSE against ``p_next`` measures how much of the
    available structure it extracts -- an honest yardstick with a known ceiling.
    """
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=float)
    n = len(obs)
    S = T.shape[0]
    # stationary start
    evals, evecs = np.linalg.eig(T.T)
    k = int(np.argmin(np.abs(evals - 1.0)))
    b = np.real(evecs[:, k]); b = np.abs(b) / np.abs(b).sum()
    p_next = np.empty(n)
    p_state = np.empty(n)
    for k_ in range(n):
        # condition on the new observation
        b = b * E[:, obs[k_]]
        b = b / b.sum()
        p_state[k_] = b[1] if S >= 2 else b[0]
        # propagate and predict the next emission
        b = T.T @ b
        p_next[k_] = float(b @ E[:, 1])
    return p_next, p_state
