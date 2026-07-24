"""nvar_baseline.py -- NVAR / NG-RC baseline, borrowed from the sister repo's rigor.

The applied pipeline so far compares the QRC only against one ESN and persistence.
The sister project (QRC_single_time_series) carries a whole baseline battery; the
single most valuable import is **NVAR** (nonlinear vector autoregression, a.k.a.
next-generation reservoir computing; Gauthier et al., Nat. Commun. 12, 5564,
2021). It is cheap, has no random seed, and routinely matches or beats an ESN --
so it is a *harder* control than the ESN. If the QRC holds parity with NVAR the
"parity" claim is much stronger; if NVAR wins, that is a finding we must report.

NVAR feature map on a delay vector v_t = [x_t, x_{t-s}, ..., x_{t-(k-1)s}]:
    phi_t = [1] (+) v_t (+) {v_i * v_j : i <= j}         # const + linear + quadratic
Ridge readout predicts the next value; multi-step is the same recursive rollout
the QRC uses (feed the prediction back into the delay buffer), so the comparison
is like-for-like on closed-loop skill.

    python nvar_baseline.py --dataset got_sst_mhwi --horizon 24 --max-points 6000
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "DataBase_Analysis"))

from dataloader import load                                    # noqa: E402
from forecast import persistence_forecast                      # noqa: E402
from qrc_core import nmse                                       # noqa: E402

RESULTS = _HERE / "results"


def nvar_features(buf, pairs):
    """phi for one delay vector buf (length k): const + linear + quadratic."""
    quad = [buf[i] * buf[j] for i, j in pairs]
    return np.concatenate(([1.0], buf, quad))


def build_design(x, k, s, idx, pairs):
    """Feature matrix for predicting x[t+1] at each t in idx (all delays valid)."""
    Phi = np.empty((len(idx), 1 + k + len(pairs)))
    for r, t in enumerate(idx):
        buf = x[t - np.arange(k) * s]            # [x_t, x_{t-s}, ...]
        Phi[r] = nvar_features(buf, pairs)
    return Phi


def rollout_nvar(x, origin, W, k, s, pairs, H):
    """Recursive H-step forecast, feeding predictions back into the delay buffer."""
    buf = list(x[origin - np.arange(k) * s])     # most-recent-first
    out = np.empty(H)
    for i in range(H):
        yhat = float(W @ nvar_features(np.array(buf), pairs))
        out[i] = yhat
        buf = [yhat] + buf[:-1]                   # shift in the prediction
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="got_sst_mhwi")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--k", type=int, default=4, help="number of delay taps")
    ap.add_argument("--s", type=int, default=1, help="delay spacing")
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--train-frac", type=float, default=0.7)
    a = ap.parse_args()

    s_ = load(a.dataset)
    x = np.asarray(s_.x, float)
    x = x[np.isfinite(x)]
    if len(x) > a.max_points:
        x = x[-a.max_points:]
    H, n, k, s = a.horizon, len(x), a.k, a.s
    pairs = list(combinations_with_replacement(range(k), 2))
    washout = (k - 1) * s + 1

    valid = np.arange(washout, n - 1)
    n_tr = int(a.train_frac * len(valid))
    train_idx, test_idx = valid[:n_tr], valid[n_tr:]

    # fit ridge readout on the training rows (predict next value, original units)
    Phi = build_design(x, k, s, train_idx, pairs)
    y = x[train_idx + 1]
    A = Phi.T @ Phi + a.lam * np.eye(Phi.shape[1])
    W = np.linalg.solve(A, Phi.T @ y)

    origins = test_idx[test_idx + H < n]
    truth = np.empty((len(origins), H))
    pred = np.empty((len(origins), H))
    for r, o in enumerate(origins):
        truth[r] = x[o + 1:o + H + 1]
        pred[r] = rollout_nvar(x, int(o), W, k, s, pairs, H)
    nvar_nmse = [float(nmse(truth[:, j], pred[:, j])) for j in range(H)]

    # persistence on the identical origins, for context
    per = np.array([[persistence_forecast(x, int(o), H)[j] for j in range(H)]
                    for o in origins])
    per_nmse = [float(nmse(truth[:, j], per[:, j])) for j in range(H)]

    # pull the QRC + ESN numbers already computed for this dataset (if present)
    ss = RESULTS / f"step2_seedsweep_{s_.key}.json"
    qrc = esn = None
    if ss.exists():
        d = json.loads(ss.read_text())
        qrc = d["mean"].get("xxz_hx") or d["mean"].get("ising")
        esn = d["mean"].get("esn")

    print(f"=== NVAR baseline: {s_.name} ({s_.key}) ===")
    print(f"N={n} H={H} k={k} s={s} features={Phi.shape[1]} "
          f"origins={len(origins)}")
    hdr = f"  {'h':>3} {'NVAR':>8} {'persist':>8}"
    if qrc:
        hdr += f" {'QRC':>8} {'ESN':>8}"
    print("\n" + hdr)
    for j in range(H):
        if j < 6 or (j + 1) % 3 == 0:
            row = f"  {j+1:>3} {nvar_nmse[j]:>8.4f} {per_nmse[j]:>8.4f}"
            if qrc:
                row += f" {qrc[j]:>8.4f} {esn[j]:>8.4f}"
            print(row)

    if qrc:
        nv, q, e = np.array(nvar_nmse), np.array(qrc), np.array(esn)
        print(f"\n  mean NMSE 1..{H}:  NVAR {nv.mean():.4f}  QRC {q.mean():.4f}  "
              f"ESN {e.mean():.4f}  persistence {np.mean(per_nmse):.4f}")
        qrc_vs_nvar = [h + 1 for h in range(H) if q[h] < nv[h] * (1 - 0.05)]
        nvar_vs_qrc = [h + 1 for h in range(H) if nv[h] < q[h] * (1 - 0.05)]
        print(f"  QRC materially beats NVAR (>5%) at: "
              f"{qrc_vs_nvar if qrc_vs_nvar else 'never'}")
        print(f"  NVAR materially beats QRC (>5%) at: "
              f"{nvar_vs_qrc if nvar_vs_qrc else 'never'}")
        if not qrc_vs_nvar and not nvar_vs_qrc:
            print("  -> QRC and NVAR are at PARITY across the horizon "
                  "(neither wins by >5% anywhere).")
        verdict = ("QRC still competitive with the harder NVAR control"
                   if q.mean() <= nv.mean() * 1.05
                   else "NVAR beats the QRC on this target -- report it")
        print(f"  VERDICT: {verdict}.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"nvar_{s_.key}.json").write_text(json.dumps(dict(
        dataset=s_.key, H=H, k=k, s=s, n_features=int(Phi.shape[1]),
        n_origins=int(len(origins)), nvar_nmse=nvar_nmse, persistence=per_nmse,
        qrc_ref=qrc, esn_ref=esn,
    ), indent=2))
    print(f"\n  wrote results/nvar_{s_.key}.json")


if __name__ == "__main__":
    main()
