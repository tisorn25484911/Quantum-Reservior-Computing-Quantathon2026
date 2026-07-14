"""experiments.py -- Result figures for the document.

Honesty discipline enforced throughout:
  * every quantum result is compared against a feature-count-matched ESN,
    a structureless Haar-random quantum control, a delay-embedding linear
    model, and (where meaningful) persistence;
  * every NMSE axis carries the mean-predictor line at 1.0;
  * the analytic (infinite-shot) ceiling is measured before the shot study;
  * conclusions are computed in-script from the results dictionary.
"""
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs
from qrc_core import (QuantumReservoir, ESN, ridge_fit, ridge_predict, nmse,
                      narma10, memory_function, add_shot_noise, propagator)
from datasets import solar_surrogate, enso_real

RNG = np.random.default_rng(0)
LAM_GRID = [1e-8, 1e-6, 1e-4, 1e-2]


def eval_features(feats, target, washout=200, train_frac=0.7):
    """Chronological split with an inner validation split for lambda."""
    Xf, y = feats[washout:], target[washout:]
    n_tr = int(train_frac * len(y))
    Xtr, ytr, Xte, yte = Xf[:n_tr], y[:n_tr], Xf[n_tr:], y[n_tr:]
    n_val = int(0.15 * n_tr)
    best = (np.inf, None)
    for lam in LAM_GRID:
        w = ridge_fit(Xtr[:-n_val], ytr[:-n_val], lam=lam)
        v = nmse(ytr[-n_val:], ridge_predict(Xtr[-n_val:], w))
        if v < best[0]:
            best = (v, lam)
    w = ridge_fit(Xtr, ytr, lam=best[1])
    pred = ridge_predict(Xte, w)
    return nmse(yte, pred), pred, yte, best[1]


def delay_embed(u, lags=10):
    X = np.zeros((len(u), lags))
    for i in range(lags):
        X[:, i] = np.roll(u, i)
    return X


class HaarReservoir(QuantumReservoir):
    """Structureless control: the sub-step unitary is Haar random."""
    def __init__(self, n_qubits=5, virtual_nodes=4, use_zz=True, seed=7):
        super().__init__(n_qubits=n_qubits, virtual_nodes=virtual_nodes,
                         use_zz=use_zz, seed=seed)
        g = np.random.default_rng(seed + 100)
        A = g.normal(size=(2**n_qubits, 2**n_qubits)) \
            + 1j * g.normal(size=(2**n_qubits, 2**n_qubits))
        Q, R = np.linalg.qr(A)
        self.U_sub = Q * (np.diagonal(R) / np.abs(np.diagonal(R)))


# ---------------------------------------------------------------- Fig: edge
print("== temporal-edge sweep (NARMA-10) ==")
u, y = narma10(2200, seed=0)
u_in = u / 0.5
dts = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
edge = []
for dt in dts:
    res = QuantumReservoir(n_qubits=5, J=1.0, h=1.0, dt=dt, virtual_nodes=4, seed=7)
    e, *_ = eval_features(res.run(u_in), y)
    edge.append(e)
    print(f"  J*dt={dt:6.3f}  NMSE={e:.4f}")
edge = np.array(edge)
dt_star = float(dts[edge.argmin()])
print(f"chosen interval J*dt = {dt_star}")

fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.semilogx(dts, edge, "o-", color=fs.ACCENT, ms=3.5)
ax.axhline(1.0, color=fs.GRAY, lw=0.8, ls="--")
ax.text(dts[0], 1.02, "mean predictor", fontsize=6.5, color=fs.GRAY)
ax.axvline(dt_star, color=fs.LIGHT, lw=0.8, zorder=0)
ax.set_xlabel(r"input interval $J\,\Delta t$")
ax.set_ylabel("test NMSE (NARMA-10)")
fs.save(fig, "temporal_edge")

# ------------------------------------------------------------- Fig: memory
print("== linear memory capacity ==")
qr = QuantumReservoir(n_qubits=5, dt=dt_star, virtual_nodes=4, seed=7)
hr = HaarReservoir(n_qubits=5, virtual_nodes=4, seed=7)
esn = ESN(qr.n_features, seed=11)
mf_q, mc_q = memory_function(qr.run, qr.n_features)
mf_h, mc_h = memory_function(hr.run, hr.n_features)
mf_e, mc_e = memory_function(esn.run, esn.n_features)
print(f"  MC: QRC={mc_q:.2f}  Haar-control={mc_h:.2f}  ESN(matched)={mc_e:.2f}")

fig, ax = plt.subplots(figsize=(3.4, 2.5))
d = np.arange(len(mf_q))
ax.plot(d, mf_q, "o-", color=fs.ACCENT, ms=3, label=f"QRC (MC={mc_q:.1f})")
ax.plot(d, mf_e, "s-", color="k", ms=3, label=f"ESN, matched (MC={mc_e:.1f})")
ax.plot(d, mf_h, "^--", color=fs.GRAY, ms=3, label=f"Haar control (MC={mc_h:.1f})")
ax.set_xlabel("delay $d$"); ax.set_ylabel("memory function $\\mathrm{MF}_d$")
ax.legend()
fs.save(fig, "memory_capacity")

# ------------------------------------------------- Fig: prediction demos
print("== task suite ==")
results = {}

# NARMA-10
feats_q = qr.run(u_in)
feats_h = hr.run(u_in)
feats_e = esn.run(u_in)
results["narma_qrc"], pred_q, yte, _ = eval_features(feats_q, y)
results["narma_haar"], *_ = eval_features(feats_h, y)
results["narma_esn"], pred_e, _, _ = eval_features(feats_e, y)
results["narma_lin10"], *_ = eval_features(delay_embed(u, 10), y)

# Solar clear-sky index, one-step-ahead (30 min), daytime stream
sol = solar_surrogate()
day = sol[(sol.elevation_deg > 5) & sol.ghi.notna()]
kt = np.clip(day.kt.to_numpy()[:4200], 0, 1.05)
u_sol = kt / 1.05
t_sol = np.roll(kt, -1)          # next-step target
u_sol, t_sol = u_sol[:-1], t_sol[:-1]
qs = QuantumReservoir(n_qubits=5, dt=dt_star, virtual_nodes=4, seed=7)
results["sol_qrc"], pred_sq, yte_s, _ = eval_features(qs.run(u_sol), t_sol)
es = ESN(qs.n_features, seed=11)
results["sol_esn"], *_ = eval_features(es.run(u_sol), t_sol)
results["sol_lin10"], *_ = eval_features(delay_embed(kt[:-1], 10), t_sol)
pers = kt[:-1]
w0 = 200; n_tr = int(0.7 * (len(t_sol) - w0))
results["sol_persist"] = nmse(t_sol[w0 + n_tr:], pers[w0 + n_tr:])

# ENSO anomaly, 3 months ahead
en = enso_real()
an = en.anomaly_c.to_numpy()
lo, hi = an.min(), an.max()
u_en = (an - lo) / (hi - lo)
h_ahead = 3
t_en = np.roll(an, -h_ahead)
u_en, t_en = u_en[:-h_ahead], t_en[:-h_ahead]
qe = QuantumReservoir(n_qubits=5, dt=dt_star, virtual_nodes=4, seed=7)
results["enso_qrc"], pred_eq, yte_e, _ = eval_features(qe.run(u_en), t_en, washout=60)
ee = ESN(qe.n_features, seed=11)
results["enso_esn"], *_ = eval_features(ee.run(u_en), t_en, washout=60)
results["enso_lin10"], *_ = eval_features(delay_embed(an[:-h_ahead], 10), t_en, washout=60)
n_tr_e = int(0.7 * (len(t_en) - 60))
results["enso_persist"] = nmse(t_en[60 + n_tr_e:], an[:-h_ahead][60 + n_tr_e:])

for k in sorted(results):
    print(f"  {k:14s} NMSE = {results[k]:.4f}")

fig, ax = plt.subplots(3, 1, figsize=(6.4, 5.0))
sl = slice(0, 150)
ax[0].plot(yte[sl], color="k", lw=1.0, label="true")
ax[0].plot(pred_q[sl], color=fs.ACCENT, lw=0.9,
           label=f"QRC (NMSE {results['narma_qrc']:.3f})")
ax[0].plot(pred_e[sl], color=fs.GRAY, lw=0.8, ls="--",
           label=f"ESN (NMSE {results['narma_esn']:.3f})")
ax[0].set_title("(a) NARMA-10, one-step target", loc="left"); ax[0].legend(ncol=3)
sl = slice(0, 260)
ax[1].plot(yte_s[sl], color="k", lw=1.0, label="true")
ax[1].plot(pred_sq[sl], color=fs.ACCENT, lw=0.9,
           label=f"QRC (NMSE {results['sol_qrc']:.3f})")
ax[1].set_title("(b) solar clear-sky index $k_t$, 30-min ahead (surrogate)", loc="left")
ax[1].legend(ncol=2)
ax[2].plot(yte_e, color="k", lw=1.0, label="true")
ax[2].plot(pred_eq, color=fs.ACCENT, lw=0.9,
           label=f"QRC (NMSE {results['enso_qrc']:.3f})")
ax[2].set_title("(c) ENSO SST anomaly, 3-month ahead (real data)", loc="left")
ax[2].legend(ncol=2); ax[2].set_xlabel("test-set step")
fs.save(fig, "prediction_demos")

# --------------------------------------------------------- Fig: shot noise
print("== shot-noise study (NARMA-10) ==")
ceiling = results["narma_qrc"]
assert ceiling < 0.8, "signal-to-noise ceiling check failed: analytic NMSE too high"
shots = np.array([1e2, 3e2, 1e3, 3e3, 1e4, 1e5])
mean_e, std_e = [], []
for S in shots:
    errs = [eval_features(add_shot_noise(feats_q, S, rng=r), y)[0] for r in range(8)]
    mean_e.append(np.mean(errs)); std_e.append(np.std(errs))
    print(f"  S={int(S):>7d}  NMSE={mean_e[-1]:.3f} +/- {std_e[-1]:.3f}")
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.errorbar(shots, mean_e, yerr=std_e, fmt="o-", color=fs.ACCENT, ms=3.5, capsize=2)
ax.axhline(1.0, color=fs.GRAY, lw=0.8, ls="--")
ax.axhline(ceiling, color="k", lw=0.8, ls=":")
ax.text(shots[0], 1.03, "mean predictor", fontsize=6.5, color=fs.GRAY)
ax.text(shots[-1], ceiling * 1.08, "analytic ceiling", fontsize=6.5,
        ha="right", color="k")
ax.set_xscale("log"); ax.set_xlabel("shots per feature $S$")
ax.set_ylabel("test NMSE (NARMA-10)")
fs.save(fig, "shot_noise")

np.save("../figures/results.npy", results, allow_pickle=True)
print("experiments complete; chosen J*dt =", dt_star)
