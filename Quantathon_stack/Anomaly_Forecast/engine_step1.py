"""engine_step1.py -- Day 2: the headline Step-1 result on the CANONICAL engine.

Runs the marine-heatwave-intensity forecast on the sister repo's validated exact
Fujii-Nakajima reservoir (vendored under engine/), closed-loop, and scores it on
identical held-out origins against the classical field that actually exists:
persistence, a size-matched ESN, and NVAR. Reports the horizon taxonomy
(H_skill / H_beats-persistence / H_beats-classical) the engine repo argues for.

A fast stateful stepper is built on their engine (their features() replays the
whole history each call; recursive rollout needs single-step advancement), using
exactly their per-step ops: inject -> read V virtual nodes -> advance one tau.

    PYTHONPATH not needed; paths are wired below.
    .venv/bin/python Quantathon_stack/Anomaly_Forecast/engine_step1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "engine" / "QRC_single_time_series" / "src"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "DataBase_Analysis"))

from qrc_single_time_series.quantum.exact_qrc import ExactQRC       # noqa: E402
from qrc_single_time_series.quantum.hamiltonians import fc_tfi      # noqa: E402
from qrc_single_time_series.quantum.input_channels import inject    # noqa: E402
from qrc_single_time_series.models.readout import fit as eng_fit    # noqa: E402

from dataloader import load                                         # noqa: E402
from forecast import (Scaler, make_reservoir, fit_readout,          # noqa: E402
                      skill_vs_horizon, persistence_forecast)
from nvar_baseline import build_design, rollout_nvar                # noqa: E402
from itertools import combinations_with_replacement                 # noqa: E402
from qrc_core import nmse                                           # noqa: E402

RESULTS = _HERE / "results"
H = 24
SEEDS = (7, 11, 23)
MATERIAL = 0.05


class _EngineDriver:
    """Stateful single-step driver on the sister repo's ExactQRC (O(H) rollout)."""

    def __init__(self, qrc: ExactQRC):
        self.q = qrc
        self.reset()

    def reset(self):
        self.rho = self.q.initial_state()

    def get_state(self):
        return self.rho.copy()

    def set_state(self, s):
        self.rho = np.asarray(s).copy()

    def step(self, u: float) -> np.ndarray:
        q = self.q
        self.rho = inject(self.rho, float(u), q.N, site=q.inject_site)
        sigma = q._sigma(self.rho)
        obs = q._obs_at_subtimes(sigma)                 # (V, M)
        row = (0.5 * (1.0 + obs)).reshape(-1)           # x' = (1+<O>)/2
        self.rho = q.W @ (sigma * q._Phi_end) @ q.Wd    # advance one tau
        return row


def engine_features(qrc, u, checkpoints=False):
    d = _EngineDriver(qrc)
    rows = np.empty((len(u), qrc.M * qrc.V))
    states = [] if checkpoints else None
    for k, uk in enumerate(u):
        rows[k] = d.step(uk)
        if checkpoints:
            states.append(d.get_state())
    rows = np.hstack([rows, np.ones((len(u), 1))])      # bias, as their features()
    return rows, states


def engine_rollout(qrc, readout, sc, u_row_state, origin, Hh):
    state, feat = u_row_state
    d = _EngineDriver(qrc)
    d.set_state(state)
    cur = feat
    out = np.empty(Hh)
    for i in range(Hh):
        u_next = float(readout.predict(cur[None, :])[0])
        out[i] = u_next
        u_fed = min(1.0, max(0.0, u_next))
        row = d.step(u_fed)
        cur = np.append(row, 1.0)
    return sc.to_data(out)


def engine_skill(x, seed):
    """Multi-step NMSE curve for the engine QRC at one reservoir seed."""
    n = len(x)
    washout = 100
    valid = np.arange(washout, n - 1)
    n_tr = int(0.7 * len(valid))
    train_idx, test_idx = valid[:n_tr], valid[n_tr:]
    sc = Scaler.fit(x[:int(train_idx[-1]) + 2])
    u = np.clip(sc.to_unit(x), 0.0, 1.0)

    qrc = ExactQRC(fc_tfi(5, J=1.0, h=0.5, seed=seed), V=10, tau=2.0,
                   observable_kind="z_local")
    feats, states = engine_features(qrc, u, checkpoints=True)
    y = u[1:]
    ro = eng_fit(feats[train_idx], y[train_idx], lam=1e-6)

    origins = test_idx[test_idx + H < n]
    truth = np.empty((len(origins), H))
    pred = np.empty((len(origins), H))
    for i, o in enumerate(origins):
        truth[i] = x[o + 1:o + H + 1]
        pred[i] = engine_rollout(qrc, ro, sc, (states[o], feats[o]), int(o), H)
    return origins, truth, [float(nmse(truth[:, j], pred[:, j])) for j in range(H)]


def taxonomy(curve, persistence, classical, H):
    c = np.asarray(curve)
    skill = [h + 1 for h in range(H) if c[h] < 1.0]
    beats_p = [h + 1 for h in range(H) if c[h] < np.asarray(persistence)[h] * (1 - MATERIAL)]
    beats_c = [h + 1 for h in range(H) if c[h] < np.asarray(classical)[h] * (1 - MATERIAL)]
    worse_c = [h + 1 for h in range(H) if np.asarray(classical)[h] < c[h] * (1 - MATERIAL)]
    def rng(hs):
        if not hs:
            return "never"
        return (f"h=1-{max(hs)}" if hs == list(range(1, max(hs) + 1))
                else f"{len(hs)}/{H}")
    return dict(H_skill=max(skill) if skill else 0,
                H_beats_persistence=max(beats_p) if beats_p else 0,
                beats_classical=rng(beats_c), classical_beats=rng(worse_c))


def main():
    x = np.asarray(load("got_sst_mhwi").x, float)
    x = x[np.isfinite(x)][-6000:]
    n = len(x)
    print("=== Day 2: Step-1 on the CANONICAL engine (sister repo exact FN) ===")
    print(f"target=got_sst_mhwi  N={n}  H={H}  seeds={SEEDS}")

    # --- engine QRC, multi-seed ---
    eng_curves = []
    origins_ref = None
    for sd in SEEDS:
        origins, truth, curve = engine_skill(x, sd)
        eng_curves.append(curve)
        origins_ref = origins
        print(f"  engine seed {sd:3d} done")
    eng = np.array(eng_curves)
    eng_mean = eng.mean(0)

    # --- baselines on the IDENTICAL origins ---
    # ESN (our qrc_core ESN) via forecast.skill_vs_horizon, same origins
    esn_curves, per_curve = [], None
    for sd in SEEDS:
        res = make_reservoir("esn", n_qubits=5, seed=sd)
        ro = fit_readout(x, res, washout=100, train_frac=0.7)
        out = skill_vs_horizon(x, res, ro, H, origins=origins_ref)
        esn_curves.append(out["nmse"]["reservoir"])
        per_curve = out["nmse"]["persistence"]
    esn_mean = np.array(esn_curves).mean(0)
    per = np.array(per_curve)

    # NVAR on identical origins
    k, s = 4, 1
    pairs = list(combinations_with_replacement(range(k), 2))
    washout = (k - 1) * s + 1
    valid = np.arange(washout, n - 1)
    tr = valid[:int(0.7 * len(valid))]
    Phi = build_design(x, k, s, tr, pairs)
    W = np.linalg.solve(Phi.T @ Phi + 1e-4 * np.eye(Phi.shape[1]), Phi.T @ x[tr + 1])
    nvar_curve = []
    truthN = np.array([x[o + 1:o + H + 1] for o in origins_ref])
    predN = np.array([rollout_nvar(x, int(o), W, k, s, pairs, H) for o in origins_ref])
    nvar = np.array([float(nmse(truthN[:, j], predN[:, j])) for j in range(H)])

    # --- report ---
    print(f"\n  origins={len(origins_ref)}")
    print(f"  {'h':>3} {'ENGINE-QRC':>12} {'ESN':>9} {'NVAR':>9} {'persist':>9}")
    for j in range(H):
        if j < 4 or (j + 1) % 3 == 0:
            print(f"  {j+1:>3} {eng_mean[j]:>8.4f}+-{eng[:,j].std():.3f}"
                  f" {esn_mean[j]:>9.4f} {nvar[j]:>9.4f} {per[j]:>9.4f}")
    print(f"\n  mean NMSE 1..{H}:  engine-QRC {eng_mean.mean():.4f}  "
          f"ESN {esn_mean.mean():.4f}  NVAR {nvar.mean():.4f}  persistence {per.mean():.4f}")

    tax = taxonomy(eng_mean, per, np.minimum(esn_mean, nvar), H)
    print("\n  HORIZON TAXONOMY (engine-QRC):")
    print(f"    H_skill (NMSE<1)          = h=1..{tax['H_skill']}")
    print(f"    H_beats_persistence (>5%) = h=1..{tax['H_beats_persistence']}")
    print(f"    beats best-classical (>5%): {tax['beats_classical']}   "
          f"classical beats QRC: {tax['classical_beats']}")

    # cross-check vs our qrc_core result
    ss = RESULTS / "step2_seedsweep_got_sst_mhwi.json"
    if ss.exists():
        d = json.loads(ss.read_text())
        ours = np.array(d["mean"]["xxz_hx"])
        print(f"\n  cross-check: our qrc_core (xxz_hx) mean NMSE {ours.mean():.4f} "
              f"vs engine {eng_mean.mean():.4f} -- same story "
              f"({'consistent' if abs(ours.mean()-eng_mean.mean())<0.05 else 'differs'}).")

    verdict = ("parity with classical, no quantum advantage -- CONFIRMED on the "
               "validated engine" if tax["classical_beats"] == "never"
               else "engine-QRC beaten by classical at some leads -- report")
    print(f"\n  VERDICT: {verdict}.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "engine_step1_got_sst_mhwi.json").write_text(json.dumps(dict(
        target="got_sst_mhwi", engine="exact_fc_tfi_V10_zlocal", H=H, seeds=list(SEEDS),
        n_origins=int(len(origins_ref)),
        engine_qrc_mean=eng_mean.tolist(), engine_qrc_std=eng.std(0).tolist(),
        esn_mean=esn_mean.tolist(), nvar=nvar.tolist(), persistence=per.tolist(),
        taxonomy=tax,
    ), indent=2))
    print("\n  wrote results/engine_step1_got_sst_mhwi.json")


if __name__ == "__main__":
    main()
