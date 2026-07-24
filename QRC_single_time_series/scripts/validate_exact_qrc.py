"""scripts/validate_exact_qrc.py -- P2 exact-QRC anchors + FN Fig.5/Table-I checks.

Qualitative reproduction (order-of-magnitude only -- couplings are random and the
paper's pinv cutoff is unstated, errata G8) on one N=5, seed=7 reservoir:

  1. Short-term memory capacity C_STM(tau_B) and its total MC rise with the number
     of virtual nodes V and saturate (FN Fig. 5 direction).
  2. Parity/processing capacity C_PC(tau_B>=1) ~ 0 at V=1 (needs virtual nodes).
  3. NARMA2 with sine input: the quantum readout (QR) beats a linear readout on the
     raw input window (LR) -- FN Table-I ordering.

Results are printed and written to results/metrics/validate_exact_qrc.json. No
matplotlib dependency (figures are regenerated once mpl is installed, G1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.quantum.hamiltonians import fc_tfi
from qrc_single_time_series.quantum.exact_qrc import ExactQRC
from qrc_single_time_series.data.windows import align
from qrc_single_time_series.models import readout as R
from qrc_single_time_series.evaluation import metrics as M

N, SEED, TAU = 5, 7, 2.0


def _reservoir(V):
    return ExactQRC(fc_tfi(N, J=1.0, h=0.5, seed=SEED), V=V, tau=TAU)


def stm_capacity(V, L=2000, ntr=1400, max_delay=12):
    s = np.random.default_rng(0).uniform(0, 1, L)
    X = _reservoir(V).features(s, check_budget=False)
    caps = {}
    for d in range(1, max_delay + 1):
        Xd, yd = X[d:], s[: L - d]
        ro = R.fit(Xd[:ntr], yd[:ntr], lam=1e-8)
        caps[d] = M.capacity(yd[ntr:], ro.predict(Xd[ntr:]))
    return caps, float(sum(caps.values()))


def parity_capacity(V, L=2000, ntr=1400, max_delay=6):
    # PC(tau_B): reconstruct parity of the binarised input over the last tau_B steps.
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, L)
    s = bits.astype(float)                       # inputs in {0,1}
    X = _reservoir(V).features(s, check_budget=False)
    total = 0.0
    for d in range(1, max_delay + 1):
        par = np.array([bits[k - d:k + 1].sum() % 2 for k in range(d, L)], dtype=float)
        Xd = X[d:]
        ro = R.fit(Xd[:ntr - d], par[:ntr - d], lam=1e-6)
        total += M.capacity(par[ntr - d:], ro.predict(Xd[ntr - d:]))
    return float(total)


def narma2(u):
    y = np.zeros(len(u))
    for t in range(1, len(u) - 1):
        y[t + 1] = 0.4 * y[t] + 0.4 * y[t] * y[t - 1] + 0.6 * u[t] ** 3 + 0.1
    return y


def narma_qr_vs_lr(V=10, L=2500, ntr=1800, washout=100, window=6):
    t = np.arange(L)
    u = 0.1 * (np.sin(2 * np.pi * t / 8.3) + 1.0)      # sine input in [0, 0.2]
    y = narma2(u)
    # QR: reservoir readout
    X = _reservoir(V).features(u / u.max(), check_budget=False)
    Xa, ya = align(X, y, 1)
    ro = R.fit(Xa[washout:ntr], ya[washout:ntr], lam="gcv")
    qr = M.nmse_variance(ya[ntr:], ro.predict(Xa[ntr:]))
    # LR: linear regression on the raw input window
    Z = np.stack([np.roll(u, k) for k in range(window)], axis=1)
    Za, yb = align(Z, y, 1)
    rl = R.fit(Za[washout:ntr], yb[washout:ntr], lam="gcv")
    lr = M.nmse_variance(yb[ntr:], rl.predict(Za[ntr:]))
    return float(qr), float(lr)


def main():
    out = {"reservoir": {"N": N, "seed": SEED, "tau": TAU, "J": 1.0, "h": 0.5}}
    print(f"Exact QRC validation -- N={N} seed={SEED} tau={TAU}\n" + "-" * 52)

    print("STM total memory capacity vs V:")
    out["stm_total_mc"] = {}
    for V in (1, 2, 5, 10):
        caps, mc = stm_capacity(V)
        out["stm_total_mc"][V] = mc
        print(f"  V={V:2d}  MC={mc:6.3f}  C(1)={caps[1]:.3f} C(3)={caps[3]:.3f}")

    print("Parity/processing capacity (V=1 vs V=10):")
    pc1, pc10 = parity_capacity(1), parity_capacity(10)
    out["pc_total"] = {1: pc1, 10: pc10}
    print(f"  PC(V=1)={pc1:.3f}  PC(V=10)={pc10:.3f}")

    print("NARMA2 sine -- QR vs LR (NMSE, lower is better):")
    qr, lr = narma_qr_vs_lr()
    out["narma2"] = {"qr_nmse": qr, "lr_nmse": lr}
    print(f"  QR NMSE={qr:.4f}   LR NMSE={lr:.4f}   QR beats LR: {qr < lr}")

    dest = ROOT / "results" / "metrics" / "validate_exact_qrc.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print("-" * 52 + f"\nwrote {dest.relative_to(ROOT)}")

    # qualitative gates (FN Fig.5 direction / Table-I ordering)
    mc = out["stm_total_mc"]
    assert mc[10] > mc[1], "MC should grow with V"
    assert pc10 > pc1, "parity capacity needs virtual nodes"
    assert qr < lr, "QR should beat LR on NARMA2"
    print("qualitative FN checks PASSED")


if __name__ == "__main__":
    main()
