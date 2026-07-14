"""experiments.py -- Phase 5: sweeps, shot budget, final metrics table.

Protocol (handbook 40.1/40.4 + CLAUDE.md):
  1. Operating-point sweep, EXACT NumPy reference: encoding gain
     gamma in {pi/4, pi/2, pi} x entangler scale s (digital JDt analogue)
     log-spaced over 2+ decades. Selection by NMSE on an inner
     chronological validation block WITHIN the train samples -- the test
     span is never touched until the final table.
  2. Shots curve at the selected config: S in {250, 1k, 4k, 16k} shots
     per basis, N_SAMPLING_SEEDS sampling seeds, NumPy multinomial
     sampling of the exact basis distributions (statistically identical
     to noiseless Aer sampling), against the exact-feature ceiling.
  3. Final table: full classical battery + QRC exact + QRC sampled
     (noiseless Aer, S_FINAL) + QRC noisy (Aer + depolarizing/readout,
     ECR/RZ/SX/X, ring coupling, S_FINAL). Event-conditional column.

Every plotted number is printed. Everything here is SIMULATION; the
ENSO data is real. Run from repo root:  python code/experiments.py
"""

import time

import numpy as np

import datasets
import figstyle
from baselines import (EVENT_THRESHOLD, HORIZON, L, chrono_split,
                       fit_predict_ridge, make_targets, run_battery, score)
from qrc_core import SEED, WindowedReservoir
from qrc_qiskit import (NOISE_P1, NOISE_P2, NOISE_RO, QiskitReservoir,
                        get_backend)

import matplotlib.pyplot as plt

GAMMAS = [np.pi / 4, np.pi / 2, np.pi]
ENT_SCALES = [0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0]   # 2.1 decades
SHOT_GRID = [250, 1000, 4000, 16000]
N_SAMPLING_SEEDS = 5
S_FINAL = 4096
INNER_TRAIN_FRAC = 0.8
FIGDIR = datasets.REPO / "figures"


def print_config():
    print(f"experiments config: gammas={[f'{g:.3f}' for g in GAMMAS]} "
          f"ent_scales={ENT_SCALES}")
    print(f"  shot grid={SHOT_GRID} sampling seeds={N_SAMPLING_SEEDS} "
          f"S_final={S_FINAL} inner_train_frac={INNER_TRAIN_FRAC} seed={SEED}")
    print(f"  selection rule: min inner-validation NMSE; "
          f"test span untouched until final table")
    print(f"  scope: simulation only (noise p1={NOISE_P1} p2={NOISE_P2} "
          f"ro={NOISE_RO}); ENSO data real")


# ------------------------------------------------------------------ setup
def harness(y):
    ks, yt = make_targets(y)
    tr, te = chrono_split(len(ks))
    icut = int(len(tr) * INNER_TRAIN_FRAC)
    itr, ival = tr[:icut], tr[icut:]
    print(f"samples: {len(ks)} | train {len(tr)} (inner-train {len(itr)}, "
          f"inner-val {len(ival)}) | test {len(te)}")
    return ks, yt, tr, te, itr, ival


def qrc_features(u_scaled, ks, gamma, s):
    res = WindowedReservoir(gamma=gamma, seed=SEED, ent_scale=s)
    return res.feature_matrix(u_scaled, L)[ks - (L - 1)], res


# ------------------------------------------------------------------ sweep
def sweep(u_scaled, ks, yt, itr, ival):
    print("\n== operating-point sweep (exact reference, inner validation) ==")
    print(f"{'gamma':>7s} {'s':>6s} {'val NMSE':>9s} {'val event':>10s}")
    grid = np.zeros((len(GAMMAS), len(ENT_SCALES)))
    best = (np.inf, None)
    for a, g in enumerate(GAMMAS):
        for b, s in enumerate(ENT_SCALES):
            Xf, _ = qrc_features(u_scaled, ks, g, s)
            pred, _ = fit_predict_ridge(Xf, yt, itr, ival)
            sc = score(yt[ival], pred, yt[itr])
            grid[a, b] = sc["nmse"]
            ev = f"{sc['nmse_event']:.3f}" if sc["nmse_event"] else "n/a"
            print(f"{g:7.3f} {s:6.2f} {sc['nmse']:9.3f} {ev:>10s}")
            if sc["nmse"] < best[0]:
                best = (sc["nmse"], (g, s))
    (g_star, s_star) = best[1]
    print(f"selected: gamma={g_star:.3f} s={s_star} "
          f"(inner-val NMSE {best[0]:.3f})")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for a, g in enumerate(GAMMAS):
        ax.plot(ENT_SCALES, grid[a], "o-", label=f"gamma={g:.2f}")
    ax.axhline(1.0, color="0.4", ls="--", lw=0.9, label="mean predictor")
    ax.plot([s_star], [best[0]], "k*", ms=12, label="selected")
    ax.set(xscale="log", xlabel="entangler scale s (JDt analogue)",
           ylabel="inner-val NMSE", title="operating-point sweep (exact)")
    ax.legend(fontsize=7)
    figstyle.save(fig, FIGDIR / "sweep_gamma_entscale.png")
    return g_star, s_star


# ------------------------------------------------------------ shots curve
def shots_curve(u_scaled, ks, yt, tr, te, g_star, s_star, ceiling):
    print(f"\n== shots curve at gamma={g_star:.3f} s={s_star} "
          f"(NumPy multinomial == noiseless-Aer statistics) ==")
    res = WindowedReservoir(gamma=g_star, seed=SEED, ent_scale=s_star)
    t0 = time.time()
    probs = [res.basis_probs(u_scaled[k - L + 1:k + 1]) for k in ks]
    print(f"basis distributions precomputed for {len(ks)} windows "
          f"({time.time() - t0:.1f} s)")
    means, sds = [], []
    for S in SHOT_GRID:
        vals = []
        for r in range(N_SAMPLING_SEEDS):
            rng = np.random.default_rng(100 + r)
            Xf = np.stack([res.sampled_features(None, S, rng, probs=p)
                           for p in probs])
            pred, _ = fit_predict_ridge(Xf, yt, tr, te)
            vals.append(score(yt[te], pred, yt[tr])["nmse"])
        m, sd = float(np.mean(vals)), float(np.std(vals))
        means.append(m)
        sds.append(sd)
        print(f"S={S:6d} shots/basis: test NMSE = {m:.3f} +- {sd:.3f} "
              f"({N_SAMPLING_SEEDS} sampling seeds)")
    print(f"exact ceiling (S=inf): {ceiling:.3f}")

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.errorbar(SHOT_GRID, means, yerr=sds, fmt="o-", capsize=3,
                label="sampled QRC")
    ax.axhline(ceiling, color="C2", ls=":", label="exact ceiling")
    ax.axhline(1.0, color="0.4", ls="--", lw=0.9, label="mean predictor")
    ax.set(xscale="log", xlabel="shots per feature basis S",
           ylabel="test NMSE", title="shot budget (simulation)")
    ax.legend(fontsize=7)
    figstyle.save(fig, FIGDIR / "shots_curve.png")


# ------------------------------------------------------------ final table
def qiskit_column(u_scaled, ks, yt, tr, te, res, noise):
    qr = QiskitReservoir(res=res)
    be = get_backend("aer", noise=noise, seed=SEED)
    t0 = time.time()
    Xf, meta = qr.sampled_feature_matrix(u_scaled, L, shots=S_FINAL,
                                         backend=be, verbose=False)
    Xf = Xf[ks - (L - 1)]
    pred, _ = fit_predict_ridge(Xf, yt, tr, te)
    print(f"  ({time.time() - t0:.0f} s, S={meta['shots_per_basis']}/basis)")
    return score(yt[te], pred, yt[tr])


def final_table(y, scaler, u_scaled, ks, yt, tr, te, g_star, s_star):
    print(f"\n== final metrics table (test span, n={len(te)}) ==")
    results, _ = run_battery(y, scaler, seed=SEED)

    res = WindowedReservoir(gamma=g_star, seed=SEED, ent_scale=s_star)
    Xf, _ = qrc_features(u_scaled, ks, g_star, s_star)
    pred, _ = fit_predict_ridge(Xf, yt, tr, te)
    results["qrc_exact"] = score(yt[te], pred, yt[tr])

    print("running qrc_sampled (noiseless Aer)...")
    results[f"qrc_sampled_S{S_FINAL}"] = qiskit_column(
        u_scaled, ks, yt, tr, te, res, noise=False)
    print("running qrc_noisy (Aer + noise, ECR/RZ/SX/X, ring)...")
    results[f"qrc_noisy_S{S_FINAL}"] = qiskit_column(
        u_scaled, ks, yt, tr, te, res, noise=True)

    print(f"\nmodel                 NMSE   event-NMSE   "
          f"(events |anom|>{EVENT_THRESHOLD} C: n={results['mean']['n_event']})")
    for name, sc in results.items():
        ev = f"{sc['nmse_event']:.3f}" if sc["nmse_event"] else "  n/a"
        print(f"{name:20s} {sc['nmse']:6.3f}      {ev}")
    print(f"\nscope: QRC config gamma={g_star:.3f}, ent_scale={s_star}, "
          f"n=5, L={L}, H={HORIZON}; sampled/noisy at S={S_FINAL} "
          f"shots/basis, 2 circuits/window; noise p1={NOISE_P1} "
          f"p2={NOISE_P2} ro={NOISE_RO}. All simulation; ENSO data real.")
    return results


if __name__ == "__main__":
    figstyle.apply()
    FIGDIR.mkdir(exist_ok=True)
    print_config()
    d = datasets.prepare()
    y, scaler = d["y"], d["scaler"]
    u_scaled = scaler(y)
    ks, yt, tr, te, itr, ival = harness(y)

    g_star, s_star = sweep(u_scaled, ks, yt, itr, ival)

    Xf, _ = qrc_features(u_scaled, ks, g_star, s_star)
    pred, _ = fit_predict_ridge(Xf, yt, tr, te)
    ceiling = score(yt[te], pred, yt[tr])["nmse"]

    shots_curve(u_scaled, ks, yt, tr, te, g_star, s_star, ceiling)
    final_table(y, scaler, u_scaled, ks, yt, tr, te, g_star, s_star)
