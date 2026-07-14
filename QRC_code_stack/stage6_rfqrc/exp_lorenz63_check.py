"""exp_lorenz63_check.py -- the declared-in-advance honesty check.

Part X: on a long-nonlinear-memory / smooth chaotic task at matched
degrees of freedom, the CLASSICAL reservoir is EXPECTED TO WIN. This
script measures valid prediction time (VPT) for closed-loop forecasting
of Lorenz-63 with RF-QRC, a feature-count-matched ESN, and NVAR. The
expected loss is declared before running; whatever happens is reported.

Protocol: one-step map trained on the attractor (inputs scaled to [0,1]
by TRAIN statistics only); closed-loop autonomous rollout from held-out
starting points; VPT at normalised-error threshold 0.4; median over
starts. Lorenz-63: sigma=10, rho=28, beta=8/3, dt=0.02, leading Lyapunov
exponent ~0.9056 -> 1 LT ~ 1.104 time units (55.2 steps).

Usage:
    python exp_lorenz63_check.py            full run
    python exp_lorenz63_check.py --check    reduced; exit 0 iff the
                                            pipeline runs end-to-end and
                                            every model beats persistence
                                            of the mean (VPT > 0)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from rfqrc_baselines import ESN, nvar_features, ridge_fit, ridge_predict
from rfqrc_metrics import valid_prediction_time
from rfqrc_reservoir import (RFQRCConfig, apply_leak, make_entangler_params,
                             step_features_exact)

SEED = 7
LORENZ_LYAP = 0.9056
DT = 0.02
DT_LT = DT * LORENZ_LYAP        # one step in Lyapunov times


def lorenz63(n_steps: int, dt: float = DT, seed: int = SEED,
             washout: int = 500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.array([1.0, 1.0, 1.0]) + 0.1 * rng.normal(size=3)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def f(v):
        return np.array([sigma * (v[1] - v[0]),
                         v[0] * (rho - v[2]) - v[1],
                         v[0] * v[1] - beta * v[2]])

    out = np.empty((n_steps + washout, 3))
    for i in range(n_steps + washout):
        k1 = f(x); k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = x
    return out[washout:]


class Scaler:
    """[0,1] min-max fit on the TRAIN span only (leakage discipline)."""

    def fit(self, x):
        self.lo, self.hi = x.min(axis=0), x.max(axis=0)
        return self

    def t(self, x):
        return np.clip((x - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def inv(self, u):
        return u * (self.hi - self.lo) + self.lo


def closed_loop_rfqrc(cfg, params, scaler, w_out, x0_hist, n_steps):
    """Autonomous rollout: reservoir state from own predictions.

    x0_hist: the last few TRUE states (>= window_m) that seed the loop.
    """
    preds = []
    hist = [scaler.t(x) for x in x0_hist]
    r = None
    for _ in range(n_steps):
        window = np.array(hist[-cfg.window_m:][::-1])   # window[0] = current
        phi = step_features_exact(window, cfg, params)
        r = cfg.leak_eps * phi if r is None else \
            (1 - cfg.leak_eps) * r + cfg.leak_eps * phi
        x_next = ridge_predict(r[None, :], w_out)[0]
        preds.append(x_next)
        hist.append(scaler.t(x_next))
    return np.array(preds)


def closed_loop_esn(esn, scaler, w_out, x0_hist, n_steps):
    s = np.zeros(esn.W.shape[0])
    for x in x0_hist:                                    # warm start on truth
        u = scaler.t(x)
        s = (1 - esn.leak) * s + esn.leak * np.tanh(esn.W @ s + esn.Win @ u)
    preds = []
    for _ in range(n_steps):
        x_next = ridge_predict(s[None, :], w_out)[0]
        preds.append(x_next)
        u = scaler.t(x_next)
        s = (1 - esn.leak) * s + esn.leak * np.tanh(esn.W @ s + esn.Win @ u)
    return np.array(preds)


def closed_loop_nvar(n_lags, max_feats, scaler, w_out, x0_hist, n_steps):
    hist = [scaler.t(x) for x in x0_hist]
    preds = []
    for _ in range(n_steps):
        emb = np.array(hist[-n_lags:][::-1]).ravel()     # current first
        d = len(emb)
        iu = np.triu_indices(d)
        F = np.concatenate([[1.0], emb, emb[iu[0]] * emb[iu[1]]])[:max_feats]
        x_next = ridge_predict(F[None, :], w_out)[0]
        preds.append(x_next)
        hist.append(scaler.t(x_next))
    return np.array(preds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    n_train = 800 if args.check else 3000
    n_test_steps = 300 if args.check else 600
    n_starts = 3 if args.check else 10
    cfg = RFQRCConfig(n_qubits=6, window_m=1, n_uploads=2,
                      entangler_layers=2, n_virtual_nodes=2,
                      readout="full_probs", leak_eps=1.0, seed=SEED)
    # leak_eps=1.0: closed-loop feedback supplies the memory; the QELM
    # posture is the published closed-loop choice. eps<1 rows belong to
    # the open-loop MFE study.
    cfg.print_config()
    print(f"lorenz63 config: dt={DT} n_train={n_train} "
          f"n_test_steps={n_test_steps} n_starts={n_starts} seed={SEED}")
    print("DECLARED IN ADVANCE (Part X): classical reservoir expected to "
          "WIN here at matched DoF.")

    data = lorenz63(n_train + n_starts * (n_test_steps + 5) + 100)
    train, rest = data[:n_train], data[n_train:]
    scaler = Scaler().fit(train)
    u_train = scaler.t(train)

    # ---- RF-QRC features on the train span (exact mode)
    params = make_entangler_params(cfg)
    raw = np.array([step_features_exact(u_train[k:k + 1], cfg, params)
                    for k in range(n_train - 1)])
    F_q = apply_leak(raw, cfg.leak_eps)
    n_feats = F_q.shape[1]
    lam = 1e-8
    w_q = ridge_fit(F_q, train[1:], lam)

    # ---- matched-capacity baselines (feature count = n_feats)
    esn = ESN(n_nodes=n_feats, n_inputs=3, seed=SEED)
    S = esn.run(u_train[:-1], washout=0)
    w_e = ridge_fit(S, train[1:], lam)

    n_lags = 4
    F_n = nvar_features(u_train[:-1], n_lags, max_features=n_feats)
    w_n = ridge_fit(F_n, train[1:], lam)
    print(f"feature-count matching: quantum {n_feats} = ESN nodes = "
          f"NVAR features (truncated)")

    # ---- closed-loop VPT from held-out starts
    vpts = {"rfqrc": [], "esn": [], "nvar": []}
    hist_len = max(cfg.window_m, n_lags, 25)
    for s in range(n_starts):
        base = s * (n_test_steps + 5)
        seg = rest[base:base + n_test_steps + hist_len]
        x_hist, truth = seg[:hist_len], seg[hist_len:]
        n_fc = len(truth)
        p_q = closed_loop_rfqrc(cfg, params, scaler, w_q, x_hist, n_fc)
        p_e = closed_loop_esn(esn, scaler, w_e, x_hist, n_fc)
        p_n = closed_loop_nvar(n_lags, n_feats, scaler, w_n, x_hist, n_fc)
        vpts["rfqrc"].append(valid_prediction_time(truth, p_q, DT_LT))
        vpts["esn"].append(valid_prediction_time(truth, p_e, DT_LT))
        vpts["nvar"].append(valid_prediction_time(truth, p_n, DT_LT))

    print(f"\nVPT medians over {n_starts} starts (threshold 0.4), in LT:")
    med = {k: float(np.median(v)) for k, v in vpts.items()}
    for k, v in med.items():
        print(f"  {k:6s} median VPT = {v:5.2f} LT   (all: "
              + " ".join(f"{x:.2f}" for x in vpts[k]) + ")")
    winner = max(med, key=med.get)
    print(f"\nresult: {winner} wins. Expected winner was classical "
          f"(esn/nvar); {'expectation CONFIRMED' if winner != 'rfqrc' else 'expectation VIOLATED - report as-is'}.")

    ok = all(v > 0 for v in med.values())
    print(f"\nexit gate (pipeline validity, not performance): every model "
          f"achieves VPT > 0 [{'PASS' if ok else 'FAIL'}]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
