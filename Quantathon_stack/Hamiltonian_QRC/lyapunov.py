"""lyapunov.py -- largest Lyapunov exponent and Lyapunov time.

Two estimators, and the difference between them is the whole point:

  `lyapunov_ensemble`   uses R independent realizations of the same system,
                        started from perturbed initial conditions. Divergence
                        is measured directly between genuinely independent
                        trajectories -- no embedding, no surrogate neighbours.

  `lyapunov_rosenstein` uses a single trajectory, substituting spatial
                        nearest neighbours inside one delay embedding for
                        those independent perturbations.

Measured against the published lambda_1 of the chaotic tier (see
`Data/README.md`), the ensemble estimator lands within -20%..+63%; Rosenstein
on the same systems runs +75% to +1210%. **A single-trajectory lambda_1 from a
real record is an upper bound with roughly factor-of-2 uncertainty, not a
measurement.** `estimate()` therefore prefers the ensemble whenever an
ensemble exists, and labels which one it used.

The reportable quantity is the Lyapunov time

    T_lambda = 1 / lambda_1,

the interval over which an initial error grows by e. It is the natural ceiling
on forecast horizon: predicting H samples ahead costs H*dt/T_lambda e-foldings
of whatever error you started with.

    from lyapunov import estimate
    est = estimate(series)
    print(est.lyap, est.lyap_time, est.method)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
# Delay embedding (only needed by the single-trajectory estimator)
# ----------------------------------------------------------------------
def mutual_information(x, max_lag, bins=None):
    """Time-delayed mutual information I(tau) in nats, via a 2-D histogram.

    Preferred over the autocorrelation for choosing tau: the ACF only sees
    linear dependence, and a chaotic series can be linearly decorrelated while
    remaining strongly deterministically coupled.
    """
    x = np.asarray(x, float)
    n = len(x)
    bins = bins or max(8, int(np.sqrt(n / 5)))
    out = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        a, b = x[:n - lag], x[lag:]
        h, _, _ = np.histogram2d(a, b, bins=bins)
        p = h / h.sum()
        pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        nz = p > 0
        out[lag] = np.sum(p[nz] * np.log(p[nz] / (pa @ pb)[nz]))
    return out


def _first_minimum(v, fallback):
    for i in range(1, len(v) - 1):
        if v[i] < v[i - 1] and v[i] <= v[i + 1]:
            return i
    return fallback


def estimate_tau(x, max_lag=None):
    """Delay tau, from the first minimum of the mutual information."""
    n = len(x)
    max_lag = max_lag or min(n // 4, 200)
    mi = mutual_information(x, max_lag)
    return max(1, _first_minimum(mi, max(1, max_lag // 4)))


def embed(x, m, tau):
    """Time-delay embedding: (N - (m-1)tau, m) rows [x_i, x_{i+tau}, ...]."""
    x = np.asarray(x, float)
    n = len(x) - (m - 1) * tau
    if n <= 0:
        raise ValueError(f"series too short for m={m}, tau={tau}")
    return np.stack([x[i * tau:i * tau + n] for i in range(m)], axis=1)


def false_nearest_neighbours(x, tau, m_max=10, rtol=15.0, atol=2.0):
    """Fraction of false neighbours at each embedding dimension 1..m_max.

    A neighbour is false if unfolding one more dimension separates it far more
    than the attractor size explains -- i.e. it was only adjacent because the
    embedding was projecting the attractor onto too few dimensions.
    """
    sigma = np.std(x)
    frac = []
    for m in range(1, m_max + 1):
        try:
            y, y2 = embed(x, m, tau), embed(x, m + 1, tau)
        except ValueError:
            break
        n = len(y2)
        y = y[:n]
        d = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = np.argmin(d, axis=1)
        dm = d[np.arange(n), nn]
        extra = np.abs(y2[:, -1] - y2[nn, -1])
        ok = dm > 0
        false = ((extra[ok] / dm[ok] > rtol) |
                 (np.sqrt(dm[ok] ** 2 + extra[ok] ** 2) / sigma > atol))
        frac.append(float(np.mean(false)))
    return np.array(frac)


def estimate_dim(x, tau, m_max=10, thresh=0.01):
    """Smallest embedding dimension whose false-neighbour fraction < thresh."""
    frac = false_nearest_neighbours(x, tau, m_max)
    below = np.flatnonzero(frac < thresh)
    return int(below[0] + 1) if below.size else int(np.argmin(frac) + 1)


# ----------------------------------------------------------------------
# Divergence curves
# ----------------------------------------------------------------------
def lyapunov_ensemble(X, dt=1.0, horizon=None, pairs=None, seed=0):
    """Divergence curve from *multiple realizations* of one system.

    `X` is (R, N): R trajectories of the same system started from initial
    conditions separated by a small known perturbation, all on the attractor.
    Then

        y(k) = ln sqrt( < (x_i(k) - x_j(k))^2 > )   over pairs (i, j),

    and lambda_1 is the slope of the initial linear stretch.

    Note the log-of-RMS rather than mean-of-logs. A *scalar* observable of two
    diverging trajectories oscillates through zero even while the underlying
    state separation grows monotonically; taking logs first turns each of
    those crossings into a large negative excursion that drags the curve and
    fakes a slope. Squaring first is blind to the crossings.

    Returns (t, curve).
    """
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("need at least two realizations, shape (R, N)")
    r, n = X.shape
    # Default to the whole record. fit_growth_window locates the scaling
    # region by fraction of total rise, which is only meaningful once the
    # curve has actually saturated -- truncating early makes a partial rise
    # look complete and biases the window down into the transient.
    horizon = n - 1 if horizon is None else min(horizon, n - 1)

    ij = [(i, j) for i in range(r) for j in range(i + 1, r)]
    if pairs is not None and len(ij) > pairs:
        rng = np.random.default_rng(seed)
        ij = [ij[k] for k in rng.choice(len(ij), pairs, replace=False)]

    d = np.array([X[i, :horizon + 1] - X[j, :horizon + 1] for i, j in ij])
    curve = 0.5 * np.log(np.mean(d ** 2, axis=0))
    return np.arange(horizon + 1) * dt, curve


def lyapunov_rosenstein(x, m=None, tau=None, dt=1.0, theiler=None,
                        horizon=None, max_n=5000):
    """Mean log divergence of nearest-neighbour pairs in a delay embedding.

    y[k] = < ln ||z_{i+k} - z_{nn(i)+k}|| > over reference points i; lambda_1
    is the slope of its initial linear stretch. `m` and `tau` are estimated
    from the data when not given.

    `theiler` rejects neighbours closer in *time* than the embedding window --
    without it, temporally adjacent points masquerade as independent
    neighbours and the curve flattens toward zero.

    `max_n` truncates the record before embedding. The neighbour search is a
    full O(N^2) distance matrix, so the hourly and half-hourly records (2-3e4
    samples) would need tens of GB. Truncation rather than decimation, because
    decimating changes dt and therefore the exponent's units; taking a prefix
    keeps the sampling rate and just shortens the record.

    Read the module docstring before quoting the result: this estimator
    inflates badly whenever a periodic component lets neighbours phase-match,
    which every seasonal climate record has.

    Returns (t, curve, (m, tau)).
    """
    x = np.asarray(x, float)
    if max_n and len(x) > max_n:
        x = x[:max_n]
    tau = estimate_tau(x) if tau is None else tau
    m = estimate_dim(x, tau) if m is None else m
    y = embed(x, m, tau)
    n = len(y)
    theiler = (m - 1) * tau if theiler is None else theiler
    horizon = min(n // 10, 100) if horizon is None else horizon

    d = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
    idx = np.arange(n)
    d[np.abs(idx[:, None] - idx[None, :]) <= theiler] = np.inf
    nn = np.argmin(d, axis=1)
    valid = np.isfinite(d[idx, nn])

    curve = np.full(horizon + 1, np.nan)
    for k in range(horizon + 1):
        i = idx[valid]
        i = i[(i + k < n) & (nn[i] + k < n)]
        if i.size == 0:
            break
        sep = np.linalg.norm(y[i + k] - y[nn[i] + k], axis=-1)
        sep = sep[sep > 0]
        if sep.size:
            curve[k] = np.mean(np.log(sep))
    return np.arange(horizon + 1) * dt, curve, (int(m), int(tau))


# ----------------------------------------------------------------------
# Fitting the scaling region
# ----------------------------------------------------------------------
def fit_growth_window(t, curve, lo_frac=0.10, hi_frac=0.60):
    """Slope of a divergence curve over its *fractional-rise* window.

    Index-based windowing assumes every system leaves its transient and
    saturates on the same schedule, which is false: against ground truth a
    fixed index window put Hadley +92% off while Lorenz-63 landed within 10%.

    Locating the window by how far the curve has *risen* adapts to each
    system. The curve climbs from ln(ic_eps) to ln(attractor diameter); the
    middle of that rise is the exponential stretch, the bottom is contaminated
    by re-orientation onto the unstable manifold, the top by saturation.

    This placement is the dominant remaining error source in the whole
    estimate -- more than the divergence curve itself. Systems whose transient
    and exponential phases overlap are still misjudged.

    Returns (slope, (lo, hi)) with lo/hi as indices into `curve`.
    """
    curve = np.asarray(curve, float)
    ok = np.isfinite(curve)
    if ok.sum() < 4:
        return np.nan, (0, 0)
    c = curve[ok]
    base, rise = c[0], c.max() - c[0]
    if rise <= 0:
        return np.nan, (0, 0)
    idx = np.flatnonzero(ok)
    above_lo = curve >= base + lo_frac * rise
    above_hi = curve >= base + hi_frac * rise
    lo = int(np.argmax(above_lo)) if above_lo.any() else int(idx[0])
    hi = int(np.argmax(above_hi)) if above_hi.any() else int(idx[-1])
    if hi - lo < 3:
        hi = min(len(curve) - 1, lo + 3)
    sl = slice(lo, hi + 1)
    tt, cc = t[sl], curve[sl]
    good = np.isfinite(cc)
    if good.sum() < 3:
        return np.nan, (lo, hi)
    return float(np.polyfit(tt[good], cc[good], 1)[0]), (lo, hi)


def fit_slope(t, curve, lo=None, hi=None):
    """Least-squares slope over an index window [lo, hi].

    Default: skip the first 1/20 of the curve (transient) and stop where the
    curve first reaches 95% of its range (saturation). Kept as the fallback
    and for comparison; `fit_growth_window` is the better default.
    """
    curve = np.asarray(curve, float)
    ok = np.isfinite(curve)
    k = np.flatnonzero(ok)
    if k.size < 4:
        return np.nan, (0, 0)
    lo = max(1, len(curve) // 20) if lo is None else lo
    if hi is None:
        span = curve[ok]
        sat = np.flatnonzero(span >= span.min() + 0.95 * (span.max() - span.min()))
        hi = int(k[sat[0]]) if sat.size else int(k[-1])
    lo, hi = int(lo), int(max(hi, lo + 3))
    sl = slice(lo, min(hi + 1, len(curve)))
    tt, cc = t[sl], curve[sl]
    good = np.isfinite(cc)
    if good.sum() < 3:
        return np.nan, (lo, hi)
    return float(np.polyfit(tt[good], cc[good], 1)[0]), (lo, hi)


def lyapunov_time(lyap):
    """T_lambda = 1/lambda_1, in the time unit lambda_1 was measured in.

    Infinite for a non-positive exponent: no exponential error growth means no
    predictability horizon set by the dynamics.
    """
    if lyap is None or not np.isfinite(lyap) or lyap <= 0:
        return np.inf
    return 1.0 / float(lyap)


# ----------------------------------------------------------------------
# Top-level estimate
# ----------------------------------------------------------------------
@dataclass
class LyapunovEstimate:
    """Result of an estimate, with the provenance needed to judge it."""

    lyap: float
    lyap_time: float
    method: str                       # "ensemble" | "rosenstein"
    t: np.ndarray = field(repr=False, default=None)
    curve: np.ndarray = field(repr=False, default=None)
    window: tuple = (0, 0)
    time_unit: str = ""
    truth: float | None = None
    embedding: tuple | None = None    # (m, tau), rosenstein only
    n_realizations: int = 0

    @property
    def error(self):
        """Relative error against published truth, if known."""
        if self.truth is None or not np.isfinite(self.lyap) or self.truth == 0:
            return None
        return self.lyap / self.truth - 1.0

    @property
    def reliable(self) -> bool:
        """Whether the number came from independent realizations.

        False means single-trajectory Rosenstein -- read as an upper bound.
        """
        return self.method == "ensemble"

    def summary(self) -> str:
        u = self.time_unit or "sample"
        s = (f"lambda_1 = {self.lyap:.4f} /{u}   "
             f"T_lambda = {self.lyap_time:.3g} {u}   [{self.method}")
        if self.embedding:
            s += f", m={self.embedding[0]}, tau={self.embedding[1]}"
        if self.n_realizations:
            s += f", R={self.n_realizations}"
        s += "]"
        if self.truth is not None:
            s += f"\n  published lambda_1 = {self.truth:.4f}, error {self.error:+.1%}"
        if not self.reliable:
            s += "\n  single trajectory: upper bound, ~factor-2 uncertainty"
        return s


def estimate(series, method="auto", **kw) -> LyapunovEstimate:
    """Estimate lambda_1 for a dataloader.Series.

    method="auto" uses the ensemble estimator when the series carries
    realizations and falls back to Rosenstein otherwise. Force one with
    method="ensemble" or method="rosenstein".
    """
    has_ens = getattr(series, "ensemble", None) is not None
    if method == "auto":
        method = "ensemble" if has_ens else "rosenstein"
    if method == "ensemble" and not has_ens:
        raise ValueError(f"{series.key!r} has no realizations; "
                         "use method='rosenstein'")

    if method == "ensemble":
        t, curve = lyapunov_ensemble(series.ensemble, dt=series.dt, **kw)
        emb = None
    else:
        t, curve, emb = lyapunov_rosenstein(series.x, dt=series.dt, **kw)

    slope, win = fit_growth_window(t, curve)
    return LyapunovEstimate(
        lyap=slope, lyap_time=lyapunov_time(slope), method=method,
        t=t, curve=curve, window=win, time_unit=series.time_unit,
        truth=getattr(series, "lyap_true", None), embedding=emb,
        n_realizations=getattr(series, "n_realizations", 0))


if __name__ == "__main__":
    from dataloader import load_all

    for k, s in load_all().items():
        try:
            print(f"\n{k}  ({s.name})")
            print("  " + estimate(s).summary().replace("\n", "\n  "))
        except Exception as exc:                       # noqa: BLE001
            print(f"  failed: {exc}")
