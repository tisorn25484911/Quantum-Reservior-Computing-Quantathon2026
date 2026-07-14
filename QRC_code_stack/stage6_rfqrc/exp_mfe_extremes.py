"""exp_mfe_extremes.py -- the MFE extreme-event anchor study (Part X Phase 4).

Task (Ahmed et al., PRR 6 043082 (2024) protocol): forecast the nine MFE
mode amplitudes closed-loop; monitor k(t) = (1/2)|a|^2; extreme event
k >= k_e = 0.1. Metrics: median predictability horizon (PH, Racca-Magri
ladder) over test starts, event F-score versus prediction-time offset,
effective rank of the feature matrix (mandatory diagnostic -- Part X:
"the rank-versus-N_v curve is a primary figure whatever it shows").

PRE-REGISTERED SUCCESS CRITERIA (frozen in Part X sec. 6; evaluated ONCE,
on the full run only -- never on --check pilots): median PH of the best
swept configuration at n=11 exceeds the size-matched ESN plateau by
>= 1.0 LT, exceeds the Haar control by >= 0.5 LT, and is >= NVAR at
matched feature count. Any clause failing, the null is the result.
Claims banned regardless: quantum advantage, speed-up, hardware
implications from Aer.

Modes:
    --check   reduced pilot: n=6, tiny ensemble, single configuration +
              Haar/ESN/NVAR comparators. Gate = pipeline end-to-end,
              every number printed. NO claim evaluation.
    (default) medium run: n=8, one seed, reduced sweep (leak_eps x tau).
    --full    the pre-registered study: n in {8..11}, N_v in {1,2,4,8},
              eps log-grid, tau two-decade log grid, gamma set,
              m in {1,2,3}, five seeds; comparators incl. the windowed
              protocol and the recurrent ring. LONG (hours-days); writes
              mfe_results.json per configuration as it goes.

FAIRNESS NOTE (rubric #2): the ESN comparator must receive a matching
hyperparameter sweep (rho, leak, input scale) in the full run; the
--check pilot runs library defaults and its comparisons are NOT citable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import numpy as np

from mfe_model import (DT_DEFAULT, K_EXTREME, LT, generate_ensemble,
                       kinetic_energy)
from rfqrc_baselines import ESN, nvar_features, ridge_fit, ridge_predict
from rfqrc_metrics import (effective_rank, event_scores_vs_offset,
                           ph_ladder, predictability_horizon)
from rfqrc_reservoir import (RFQRCConfig, apply_leak, make_entangler_params,
                             step_features_exact)

SEED = 7
DT_LT = DT_DEFAULT / LT          # one step in Lyapunov times (~0.004)


class Scaler:
    def fit(self, x):
        self.lo, self.hi = x.min(axis=0), x.max(axis=0)
        span = self.hi - self.lo
        span[span < 1e-12] = 1.0
        self.span = span
        return self

    def t(self, x):
        return np.clip((x - self.lo) / self.span, 0.0, 1.0)


def train_features_rfqrc(u_train, cfg, params):
    raw = np.array([step_features_exact(u_train[k:k + 1], cfg, params)
                    for k in range(len(u_train))])
    return apply_leak(raw, cfg.leak_eps)


def closed_loop(model_step, x_hist_scaled, n_steps, predict):
    """Generic autonomous rollout. model_step(state, u) -> state;
    predict(state) -> next raw 9-vector; caller closes the loop by
    re-scaling."""
    state, hist = None, list(x_hist_scaled)
    preds = []
    for _ in range(n_steps):
        state = model_step(state, hist[-1])
        x_next = predict(state)
        preds.append(x_next)
        hist.append(x_next)   # caller-provided predict returns SCALED-space
    return preds


def run_one_config(cfg: RFQRCConfig, train_series, test_series,
                   n_forecast_steps: int, warm_steps: int,
                   comparators: bool = True, esn_kw: dict | None = None):
    """Train on the train series (one-step map on scaled amplitudes),
    then closed-loop forecast each test series; return per-model median
    PH + event scores + diagnostics. Every number returned is printed by
    the caller."""
    train_cat = np.vstack(train_series)
    scaler = Scaler().fit(train_cat)
    params = make_entangler_params(cfg)

    # ---- training features (per series, no cross-series leak carry-over)
    Fq, Yq = [], []
    for tr in train_series:
        u = scaler.t(tr)
        F = train_features_rfqrc(u[:-1], cfg, params)
        Fq.append(F)
        Yq.append(u[1:])
    Fq, Yq = np.vstack(Fq), np.vstack(Yq)
    n_feats = Fq.shape[1]
    lam = 1e-8
    w_q = ridge_fit(Fq, Yq, lam)
    eff_rank = effective_rank(Fq)

    models = {"rfqrc": None}
    if comparators:
        esn = ESN(n_nodes=n_feats, n_inputs=9, seed=cfg.seed,
                  **(esn_kw or {}))
        Fs, n_lags = [], 3
        Se, Ye = [], []
        Fn, Yn = [], []
        for tr in train_series:
            u = scaler.t(tr)
            Se.append(esn.run(u[:-1], washout=0))
            Ye.append(u[1:])
            Fn.append(nvar_features(u[:-1], n_lags, max_features=n_feats))
            Yn.append(u[1:])
        w_e = ridge_fit(np.vstack(Se), np.vstack(Ye), lam)
        w_n = ridge_fit(np.vstack(Fn), np.vstack(Yn), lam)
        cfg_h = RFQRCConfig(**{**cfg.__dict__, "entangler": "haar_control"})
        params_h = make_entangler_params(cfg_h)
        Fh = []
        for tr in train_series:
            u = scaler.t(tr)
            Fh.append(train_features_rfqrc(u[:-1], cfg_h, params_h))
        w_h = ridge_fit(np.vstack(Fh), Yq, lam)
        models.update({"esn": (esn, w_e), "nvar": (n_lags, w_n),
                       "haar": (cfg_h, params_h, w_h)})

    # ---- closed-loop forecasts
    ph = {m: [] for m in models}
    ktru, kpred = {m: [] for m in models}, {m: [] for m in models}
    for ts in test_series:
        u = scaler.t(ts)
        hist, truth = u[:warm_steps], ts[warm_steps:
                                         warm_steps + n_forecast_steps]
        k_true = kinetic_energy(truth)

        def rollout_rfqrc(c, p, w):
            # warm the leak integrator on the TRUE history first (open
            # loop), mirroring the ESN warm start -- the leak's zero init
            # must be washed out before the loop closes
            r = None
            for x_h in hist:
                phi = step_features_exact(np.array([x_h]), c, p)
                r = c.leak_eps * phi if r is None else \
                    (1 - c.leak_eps) * r + c.leak_eps * phi
            h = list(hist)
            preds = []
            for _ in range(len(truth)):
                x = np.clip(ridge_predict(r[None], w)[0], 0, 1)
                preds.append(x)
                h.append(x)
                phi = step_features_exact(np.array([h[-1]]), c, p)
                r = (1 - c.leak_eps) * r + c.leak_eps * phi
            return np.array(preds)

        for m in models:
            if m == "rfqrc":
                pu = rollout_rfqrc(cfg, params, w_q)
            elif m == "haar":
                pu = rollout_rfqrc(*models["haar"])
            elif m == "esn":
                esn_, w_ = models["esn"]
                s = np.zeros(esn_.W.shape[0])
                for x in hist:
                    s = ((1 - esn_.leak) * s
                         + esn_.leak * np.tanh(esn_.W @ s + esn_.Win @ x))
                preds = []
                for _ in range(len(truth)):
                    x = np.clip(ridge_predict(s[None], w_)[0], 0, 1)
                    preds.append(x)
                    s = ((1 - esn_.leak) * s
                         + esn_.leak * np.tanh(esn_.W @ s + esn_.Win @ x))
                pu = np.array(preds)
            else:  # nvar
                n_lags, w_ = models["nvar"]
                h = list(hist)
                preds = []
                for _ in range(len(truth)):
                    emb = np.array(h[-n_lags:][::-1]).ravel()
                    d = len(emb)
                    iu = np.triu_indices(d)
                    F = np.concatenate(
                        [[1.0], emb, emb[iu[0]] * emb[iu[1]]])[:n_feats]
                    x = np.clip(ridge_predict(F[None], w_)[0], 0, 1)
                    preds.append(x)
                    h.append(x)
                pu = np.array(preds)
            pred_raw = pu * scaler.span + scaler.lo
            k_p = kinetic_energy(pred_raw)
            ph[m].append(ph_ladder(predictability_horizon(
                k_true, k_p, K_EXTREME, DT_LT)))
            ktru[m].append(k_true)
            kpred[m].append(k_p)

    out = {"n_features": n_feats, "effective_rank": float(eff_rank),
           "median_ph": {m: float(np.median(v)) for m, v in ph.items()},
           "ph_all": {m: list(map(float, v)) for m, v in ph.items()}}
    # event scores in three offset bins (in steps)
    nb = n_forecast_steps
    bins = [0, nb // 3, 2 * nb // 3, nb]
    out["event_scores"] = {
        m: event_scores_vs_offset(np.array(ktru[m]), np.array(kpred[m]),
                                  K_EXTREME, bins)
        for m in models}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if args.check:
        # kept small enough for the run_tests.py promotion gate (~3 min);
        # comparisons at this scale are NOT citable (see module docstring)
        n_train_series, train_lt = 3, 2.0
        n_test, forecast_lt, n_q = 4, 1.5, 9
        sweep = [dict(leak_eps=0.3)]
    elif args.full:
        n_train_series, train_lt = 25, 20.0
        n_test, forecast_lt = 500, 12.0
        n_q = None  # swept below
        sweep = None
    else:
        n_train_series, train_lt = 10, 8.0
        n_test, forecast_lt, n_q = 40, 8.0, 8
        sweep = [dict(leak_eps=e, tau=t)
                 for e in (0.1, 0.3, 1.0) for t in (0.5, 1.0, 2.0)]

    steps_per_lt = int(round(LT / DT_DEFAULT))
    train_steps = int(train_lt * steps_per_lt)
    forecast_steps = int(forecast_lt * steps_per_lt)
    warm = 50

    print(f"exp_mfe_extremes config: seed={SEED} check={args.check} "
          f"full={args.full} train={n_train_series}x{train_lt}LT "
          f"test={n_test}x{forecast_lt}LT dt={DT_DEFAULT} "
          f"k_e={K_EXTREME} 1LT={steps_per_lt} steps")
    print("NOTE: pre-registered criteria are evaluated on --full ONLY.")

    series, disc = generate_ensemble(
        n_train_series + n_test, train_steps + forecast_steps + warm,
        seed=SEED)
    print(f"ensemble: kept {len(series)}, discarded {disc} laminarised")
    train_series = [s[:train_steps] for s in series[:n_train_series]]
    test_series = series[n_train_series:n_train_series + n_test]

    if args.full:
        results = {}
        grid = itertools.product((8, 9, 10, 11), (1, 2, 4, 8),
                                 np.geomspace(0.02, 1.0, 5),
                                 np.geomspace(0.1, 10.0, 5),
                                 (np.pi / 4, np.pi / 2, np.pi),
                                 (1, 2, 3), range(5))
        for n, nv, eps, tau, gamma, m, sd in grid:
            key = f"n{n}_nv{nv}_e{eps:.3f}_t{tau:.2f}_g{gamma:.2f}_m{m}_s{sd}"
            cfg = RFQRCConfig(n_qubits=n, n_virtual_nodes=nv,
                              leak_eps=float(eps), tau=float(tau),
                              gamma=float(gamma), window_m=m,
                              seed=SEED + sd)
            r = run_one_config(cfg, train_series, test_series,
                               forecast_steps, warm,
                               comparators=(sd == 0 and m == 1))
            results[key] = r
            with open("mfe_results.json", "w") as f:
                json.dump(results, f, indent=1)
            print(f"{key}: PH {r['median_ph']} rank "
                  f"{r['effective_rank']:.1f} ({time.time() - t0:.0f}s)")
        print("full sweep complete; evaluate the pre-registered criteria "
              "ONCE against mfe_results.json")
        return 0

    all_ok = True
    for kw in sweep:
        cfg = RFQRCConfig(n_qubits=n_q, n_virtual_nodes=2,
                          entangler_layers=2, readout="full_probs",
                          seed=SEED, **kw)
        r = run_one_config(cfg, train_series, test_series, forecast_steps,
                           warm, comparators=True)
        print(f"\nconfig {kw}: features={r['n_features']} "
              f"effective_rank={r['effective_rank']:.2f}")
        for m, v in r["median_ph"].items():
            print(f"  {m:6s} median PH = {v:5.2f} LT  "
                  f"(all: {' '.join(f'{x:.1f}' for x in r['ph_all'][m])})")
        for m, scores in r["event_scores"].items():
            row = " ".join(f"[{s['bin'][0]}-{s['bin'][1]})F1={s['f1']:.2f}"
                           + ("*" if s["degenerate"] else "")
                           for s in scores)
            print(f"  {m:6s} event F1 by offset: {row}  (* = degenerate)")
        all_ok &= np.isfinite(r["effective_rank"])

    print(f"\ntotal {time.time() - t0:.0f}s")
    print(f"exit gate (pipeline validity): {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
