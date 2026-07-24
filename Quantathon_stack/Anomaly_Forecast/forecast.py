"""forecast.py -- QRC glue for the Forecast-then-Detect pipeline (plan.md Step 1).

Replaces the missing ``experiments.py`` that ``predict_timeseries.ipynb`` imports.
Everything downstream (scorers, thresholds, evaluation) consumes the trajectories
this module produces, so the surface is deliberately small:

    make_reservoir(kind, ...)              "ising" | "xxz_hx" | "esn"
    fit_readout(x, res, ...)               -> Readout (W_out + scaler + split)
    forecast_1step(x, res, readout)        -> one-step-ahead predictions
    rollout(x, res, readout, origin, H)    -> recursive H-step trajectory
    skill_vs_horizon(...)                  -> NMSE(h) for the Step 2 kill-test

**Recursive rollout means the reservoir is driven by its own output.** After the
first predicted step the input is no longer data, so errors compound; that
compounding is what the skill-decay curve measures.

Three conventions everything downstream depends on:

* **The read-out works in scaled space.** Inputs must lie in [0, 1] for the
  amplitude encoding (``|psi_u> = sqrt(1-u)|0> + sqrt(u)|1>``), so the read-out
  predicts the *scaled* next value and its own output is directly feedable.
  Metrics are reported in original units. NMSE is invariant under that affine
  map; only RMSE cares.
* **The scaler is fitted on the training span only.** Fitting over the whole
  series leaks the test range backwards into training.
* **Feeding back requires clipping to [0, 1], and clipping is counted.** The
  encoding is undefined outside that interval, but clipping also caps extreme
  predictions -- precisely the excursions an anomaly pipeline exists to catch.
  :func:`rollout` therefore reports the *unclipped* trajectory while feeding
  the clipped value, and returns the clip count as a diagnostic.

Run this module directly to execute the Step 1 verification gates.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_STACK = _HERE.parent
for _p in (_STACK / "Hamiltonian_QRC", _STACK / "DataBase_Analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qrc_core import (ESN, QuantumReservoir, inject, nmse,  # noqa: E402
                      ridge_fit, ridge_predict)

__all__ = [
    "Scaler", "Readout", "RolloutResult",
    "make_reservoir", "make_driver", "drive",
    "fit_readout", "forecast_1step", "rollout", "skill_vs_horizon",
    "persistence_forecast", "climatology_forecast",
]

SEED = 7


# ----------------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------------
@dataclass
class Scaler:
    """Affine map into [0, 1], fitted on a training span only.

    A margin is left at both ends so test-era values slightly outside the
    training range do not all pile onto the clip boundary, where the encoding
    saturates and the reservoir stops distinguishing them.
    """

    lo: float
    hi: float

    @classmethod
    def fit(cls, x: np.ndarray, margin: float = 0.05) -> "Scaler":
        lo, hi = float(np.min(x)), float(np.max(x))
        span = hi - lo
        if span < 1e-12:
            raise ValueError("training span is constant; nothing to learn")
        return cls(lo - margin * span, hi + margin * span)

    def to_unit(self, x):
        return (np.asarray(x, float) - self.lo) / (self.hi - self.lo)

    def to_data(self, u):
        return np.asarray(u, float) * (self.hi - self.lo) + self.lo


# ----------------------------------------------------------------------
# Reservoir construction and stateful driving
# ----------------------------------------------------------------------
def make_reservoir(kind: str = "ising", n_qubits: int = 5, J: float = 1.0,
                   hx: float = 1.0, dt: float = 2.0, virtual_nodes: int = 4,
                   use_zz: bool = True, seed: int = SEED,
                   n_nodes: int | None = None):
    """Build a reservoir. ``kind`` is ``"ising"``, ``"xxz_hx"``, or ``"esn"``.

    The ESN defaults to a quantum reservoir's feature count, so the mandatory
    classical comparison trains the same number of read-out weights.
    """
    if kind == "esn":
        if n_nodes is None:
            probe = QuantumReservoir(n_qubits=n_qubits,
                                     virtual_nodes=virtual_nodes,
                                     use_zz=use_zz, seed=seed)
            n_nodes = probe.n_features
        return ESN(n_nodes, seed=seed)
    return QuantumReservoir(n_qubits=n_qubits, J=J, h=hx, dt=dt,
                            virtual_nodes=virtual_nodes, use_zz=use_zz,
                            seed=seed, hamiltonian=kind)


class _QRCDriver:
    """Single-step driver for :class:`QuantumReservoir`, with checkpointing.

    ``QuantumReservoir.run()`` takes a whole array and restarts from the
    maximally mixed state each call, so recursive rollout via repeated ``run()``
    on a growing sequence replays the entire history at every step. plan.md
    Open Item 1 allows that O(H^2) route as "acceptable at H<=50"; measured on
    this series it is not, because the replayed history dominates the cost:

        one H=24 rollout at origin t with ~640 steps of history
          = sum_i (640 + i) ~= 15,700 reservoir steps ~= 3.5 s
          x ~190 test origins ~= 11 min per configuration,
            before Step 3 multiplies it by K samples.

    Advancing one input at a time and *checkpointing* the state instead makes a
    rollout O(H) and the whole sweep O(N + origins*H). The arithmetic is
    identical to ``run()``, which :func:`verify_step1` asserts to machine
    precision -- a performance refactor with no semantic change.
    """

    def __init__(self, res: QuantumReservoir):
        self.res = res
        self.n_features = res.n_features
        self.reset()

    def reset(self) -> None:
        self.rho = self.res.initial_state()

    def get_state(self):
        return self.rho.copy()

    def set_state(self, state) -> None:
        self.rho = np.asarray(state).copy()

    def step(self, u: float) -> np.ndarray:
        """Advance one input step; return that step's feature row."""
        r = self.res
        self.rho = inject(self.rho, float(u), r.n)
        row = []
        for _ in range(r.V):
            self.rho = r.U_sub @ self.rho @ r.U_sub.conj().T
            row.extend(np.real(np.trace(op @ self.rho)) for op in r.Z_ops)
        row.extend(np.real(np.trace(op @ self.rho)) for op in r.ZZ_ops)
        return np.asarray(row)


class _ESNDriver:
    """Single-step driver for the ESN baseline; identical interface."""

    def __init__(self, res: ESN):
        self.res = res
        self.n_features = res.n_features
        self.reset()

    def reset(self) -> None:
        self.x = np.zeros(self.res.W.shape[0])

    def get_state(self):
        return self.x.copy()

    def set_state(self, state) -> None:
        self.x = np.asarray(state).copy()

    def step(self, u: float) -> np.ndarray:
        r = self.res
        pre = r.W @ self.x + r.W_in[:, 0] * float(u)
        self.x = (1 - r.leak) * self.x + r.leak * np.tanh(pre)
        return self.x.copy()


def make_driver(res):
    """Wrap a reservoir in the stateful driver interface."""
    return _ESNDriver(res) if isinstance(res, ESN) else _QRCDriver(res)


def drive(res, u: np.ndarray, checkpoints: bool = False):
    """Drive a reservoir over ``u``; optionally keep the state after each step.

    Returns ``(features, states)``. ``states[k]`` is the reservoir state after
    consuming ``u[k]`` -- what a rollout with origin ``k`` resumes from. Both
    the state *and* ``features[k]`` are needed to resume, because the feature
    row contains intermediate virtual-node readings that the final state alone
    does not determine.
    """
    d = make_driver(res)
    u = np.asarray(u, float)
    feats = np.empty((len(u), d.n_features))
    states = [] if checkpoints else None
    for k, uk in enumerate(u):
        feats[k] = d.step(uk)
        if checkpoints:
            states.append(d.get_state())
    return feats, states


# ----------------------------------------------------------------------
# Read-out
# ----------------------------------------------------------------------
@dataclass
class Readout:
    """A trained one-step-ahead read-out plus everything needed to reuse it."""

    w: np.ndarray
    scaler: Scaler
    washout: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    residuals: np.ndarray          # training residuals, in SCALED units
    n_features: int
    kind: str

    def predict_scaled(self, features: np.ndarray) -> np.ndarray:
        return ridge_predict(np.atleast_2d(features), self.w)


def fit_readout(x: np.ndarray, res, washout: int = 100, train_frac: float = 0.7,
                lam: float = 1e-6, scaler: Scaler | None = None,
                train_noise: float = 0.0,
                rng: np.random.Generator | None = None) -> Readout:
    """Train the ridge read-out for one-step-ahead prediction.

    ``x`` is the raw series; targets are the *scaled* next value so the
    read-out's own output can be fed straight back during rollout.

    ``train_noise`` implements the closed-loop robustness trick from
    Fujii & Nakajima (2017), Appendix A 3 (their Mackey-Glass task): fit the
    read-out on **noise-perturbed** reservoir features so that, when the loop is
    later closed and the reservoir is driven by its own imperfect output, the
    read-out has learned to correct back toward the trajectory instead of
    overfitting the exact clean training states. Without it, a recursive rollout
    of a mean read-out contracts to a fixed point (the free-run flat-line). The
    noise is zero-mean Gaussian, s.d. ``train_noise`` in scaled feature units;
    ``0.0`` reproduces the previous behaviour exactly. Residuals are computed on
    the **clean** features so the Step-3 bootstrap pool stays honest.
    """
    x = np.asarray(x, float)
    n = len(x)
    valid = np.arange(washout, n - 1)          # x[k+1] must exist
    if len(valid) < 50:
        raise ValueError(
            f"only {len(valid)} usable steps after washout={washout}")

    n_tr = int(train_frac * len(valid))
    train_idx, test_idx = valid[:n_tr], valid[n_tr:]

    if scaler is None:
        # Train span only: up to and including the last training target.
        scaler = Scaler.fit(x[:int(train_idx[-1]) + 2])

    u = scaler.to_unit(x)
    feats, _ = drive(res, np.clip(u, 0.0, 1.0))
    y = u[1:]                                   # scaled next value

    X_tr = feats[train_idx]
    if train_noise > 0.0:
        if rng is None:
            rng = np.random.default_rng(SEED)
        X_tr = X_tr + rng.normal(0.0, float(train_noise), size=X_tr.shape)

    w = ridge_fit(X_tr, y[train_idx], lam=lam)
    resid = y[train_idx] - ridge_predict(feats[train_idx], w)   # clean features

    return Readout(w=w, scaler=scaler, washout=washout, train_idx=train_idx,
                   test_idx=test_idx, residuals=resid,
                   n_features=feats.shape[1],
                   kind=getattr(res, "hamiltonian", "esn"))


def forecast_1step(x: np.ndarray, res, readout: Readout) -> np.ndarray:
    """One-step-ahead prediction in ORIGINAL units.

    ``yhat[k]`` predicts ``x[k+1]``; length ``len(x) - 1``.
    """
    u = np.clip(readout.scaler.to_unit(np.asarray(x, float)), 0.0, 1.0)
    feats, _ = drive(res, u)
    return readout.scaler.to_data(ridge_predict(feats[:-1], readout.w))


# ----------------------------------------------------------------------
# Recursive rollout
# ----------------------------------------------------------------------
@dataclass
class RolloutResult:
    """An H-step recursive forecast plus the diagnostics that qualify it."""

    yhat: np.ndarray          # ORIGINAL units, predictions of x[o+1..o+H]
    n_clipped: int            # steps whose fed-back value hit the [0,1] bound
    origin: int

    @property
    def clipped(self) -> bool:
        return self.n_clipped > 0


def rollout(x: np.ndarray, res, readout: Readout, origin: int, H: int,
            state=None, feat=None, rng: np.random.Generator | None = None,
            bootstrap: bool = False) -> RolloutResult:
    """Recursive ``H``-step forecast from information up to ``x[origin]``.

    Parameters
    ----------
    state, feat : optional
        Checkpoint from :func:`drive` (``states[origin]`` and
        ``features[origin]``). Supplying both skips re-driving the history and
        is what makes a sweep affordable; omit them and the history is replayed.
    rng, bootstrap :
        With ``bootstrap=True`` a residual drawn from the training pool is
        added at each step -- the Step 3 residual bootstrap. Default is the
        deterministic mean trajectory.

    Returns
    -------
    RolloutResult
        ``yhat`` holds the *unclipped* predictions in original units; the
        clipped value is what gets fed back. Reporting the clipped value would
        silently flatten exactly the excursions this pipeline is built to find.
    """
    x = np.asarray(x, float)
    sc = readout.scaler
    d = make_driver(res)

    if state is None or feat is None:
        # No checkpoint: replay the history into our own driver, keeping the
        # last feature row. This is the O(history) path the checkpoint avoids.
        u_hist = np.clip(sc.to_unit(x[:origin + 1]), 0.0, 1.0)
        cur = None
        for uk in u_hist:
            cur = d.step(uk)
    else:
        d.set_state(state)
        cur = np.asarray(feat)

    if bootstrap and rng is None:
        rng = np.random.default_rng(SEED)

    out = np.empty(H)
    n_clipped = 0
    for i in range(H):
        u_next = float(ridge_predict(cur[None, :], readout.w)[0])
        if bootstrap:
            u_next += float(rng.choice(readout.residuals))
        out[i] = u_next                          # report unclipped
        u_fed = min(1.0, max(0.0, u_next))       # feed clipped
        if u_fed != u_next:
            n_clipped += 1
        cur = d.step(u_fed)

    return RolloutResult(yhat=sc.to_data(out), n_clipped=n_clipped,
                         origin=int(origin))


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def persistence_forecast(x: np.ndarray, origin: int, H: int) -> np.ndarray:
    """Repeat the last observation: yhat(o+h) = x(o) for every h."""
    return np.full(H, float(np.asarray(x, float)[origin]))


def climatology_forecast(x: np.ndarray, origin: int, H: int, period: int,
                         train_end: int) -> np.ndarray:
    """Seasonal-mean forecast: the training-era mean for each phase of the cycle.

    A genuinely hard baseline on a seasonal series, and the one a climate
    forecast has to beat to be worth anything. Fitted on training data only.
    """
    x = np.asarray(x, float)
    phase_mean = np.array([
        np.mean(x[:train_end][np.arange(train_end) % period == p])
        for p in range(period)])
    return np.array([phase_mean[(origin + h) % period] for h in range(1, H + 1)])


# ----------------------------------------------------------------------
# Step 2: skill decay with horizon
# ----------------------------------------------------------------------
def skill_vs_horizon(x: np.ndarray, res, readout: Readout, H: int,
                     origins: np.ndarray | None = None,
                     period: int | None = None) -> dict:
    """NMSE against lead time for the reservoir and every baseline.

    Rolls out from each test-set origin, collects predictions of ``x[o+h]`` for
    ``h = 1..H``, and scores each horizon across origins. The reservoir, the
    persistence floor and (when ``period`` is given) climatology are all scored
    on the *identical* origin set, so the comparison is like-for-like.
    """
    x = np.asarray(x, float)
    n = len(x)
    if origins is None:
        origins = readout.test_idx[readout.test_idx + H < n]
    origins = np.asarray([o for o in origins if o + H < n], dtype=int)
    if len(origins) < 10:
        raise ValueError(f"only {len(origins)} origins admit an H={H} rollout")

    u_all = np.clip(readout.scaler.to_unit(x), 0.0, 1.0)
    feats, states = drive(res, u_all, checkpoints=True)

    train_end = int(readout.train_idx[-1]) + 1
    truth = np.empty((len(origins), H))
    pred_res = np.empty((len(origins), H))
    pred_per = np.empty((len(origins), H))
    pred_cli = np.empty((len(origins), H)) if period else None
    n_clipped = 0

    for i, o in enumerate(origins):
        truth[i] = x[o + 1: o + H + 1]
        r = rollout(x, res, readout, int(o), H,
                    state=states[o], feat=feats[o])
        pred_res[i] = r.yhat
        n_clipped += r.n_clipped
        pred_per[i] = persistence_forecast(x, int(o), H)
        if period:
            pred_cli[i] = climatology_forecast(x, int(o), H, period, train_end)

    def per_h(pred):
        return [float(nmse(truth[:, j], pred[:, j])) for j in range(H)]

    out = {"horizons": list(range(1, H + 1)),
           "n_origins": int(len(origins)),
           "nmse": {"reservoir": per_h(pred_res),
                    "persistence": per_h(pred_per)},
           "clip_rate": float(n_clipped / (len(origins) * H)),
           "kind": readout.kind}
    if period:
        out["nmse"]["climatology"] = per_h(pred_cli)
    return out


# ----------------------------------------------------------------------
# Step 1 verification gates (plan.md Step 1 "Verify")
# ----------------------------------------------------------------------
def verify_step1() -> None:
    """Assert the Step 1 gates. Raises AssertionError on any failure."""
    sys.path.insert(0, str(_STACK / "DataBase_Analysis"))
    from dataloader import load

    x = np.asarray(load("nino34").x, float)

    for kind in ("ising", "xxz_hx", "esn"):
        res = make_reservoir(kind, n_qubits=5, seed=SEED)

        # -- driver equivalence: stepping == run(), to machine precision.
        # This is what licenses the checkpointed rollout as a pure performance
        # refactor rather than a different model.
        u = np.clip(Scaler(-3.0, 3.0).to_unit(x[:120]), 0.0, 1.0)
        f_batch = res.run(u)
        f_step, states = drive(res, u, checkpoints=True)
        assert np.allclose(f_batch, f_step, atol=1e-12), \
            f"{kind}: stepwise features differ from run()"
        assert len(states) == len(u)

        ro = fit_readout(x, res, washout=100, train_frac=0.7)

        # -- gate: rollout(H=1) reproduces forecast_1step exactly, same index.
        y1 = forecast_1step(x, res, ro)
        for o in (150, 300, 500):
            r = rollout(x, res, ro, o, 1)
            assert abs(r.yhat[0] - y1[o]) < 1e-10, \
                f"{kind}: rollout(H=1) != forecast_1step at origin {o}"

        # -- gate: checkpointed rollout == replayed-history rollout.
        u_all = np.clip(ro.scaler.to_unit(x), 0.0, 1.0)
        feats, sts = drive(res, u_all, checkpoints=True)
        a = rollout(x, res, ro, 400, 8, state=sts[400], feat=feats[400])
        b = rollout(x, res, ro, 400, 8)
        assert np.allclose(a.yhat, b.yhat, atol=1e-10), \
            f"{kind}: checkpointed rollout differs from replayed"

        # -- gate: the read-out learned something on held-out data.
        e = nmse(x[ro.test_idx + 1], y1[ro.test_idx])
        assert np.isfinite(e), f"{kind}: non-finite test NMSE"
        print(f"  {kind:7s} n_features={ro.n_features:3d}  "
              f"1-step test NMSE={e:.4f}")

    print("Step 1 gates PASSED "
          "(driver equivalence, rollout(H=1) identity, checkpoint identity).")


if __name__ == "__main__":
    verify_step1()
