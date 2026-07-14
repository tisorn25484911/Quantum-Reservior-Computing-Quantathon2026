"""stacker.py -- the Phase-10 stacker ablation under the fairness rule.

Heads compared on IDENTICAL feature blocks: ridge (default-simple),
gradient boosting (sklearn HistGradientBoostingRegressor -- the tabular
default per the Part XI survey), and hooks for shallow TCN/LSTM/small
transformer (evaluated at study time, not assumed). The fairness rule:
every head offered to the hybrid block is offered to the classical-only
ablation IN THE SAME TABLE, and if the quantum delta is zero the table
says so (Phase-10 acceptance).
"""

from __future__ import annotations

import numpy as np


def _ridge(Ftr, ytr, Fte, lam=1e-3):
    Xd = np.hstack([Ftr, np.ones((len(Ftr), 1))])
    w = np.linalg.solve(Xd.T @ Xd + lam * np.eye(Xd.shape[1]), Xd.T @ ytr)
    return np.hstack([Fte, np.ones((len(Fte), 1))]) @ w


def _gbm(Ftr, ytr, Fte, seed=7):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(random_state=seed, max_depth=4,
                                      max_iter=200)
    m.fit(Ftr, ytr)
    return m.predict(Fte)

HEADS = {"ridge": _ridge, "gbm": _gbm}


def stacker_table(blocks_train: dict, blocks_test: dict,
                  y_train: np.ndarray, y_test: np.ndarray,
                  quantum_key: str = "quantum") -> dict:
    """blocks_*: named feature blocks, e.g. {'classical': F_c,
    'quantum': F_q}. Rows: every head x {hybrid, classical-only}.
    Returns the table with the quantum delta per head, stated plainly."""
    Fc_tr = np.hstack([v for k, v in sorted(blocks_train.items())
                       if k != quantum_key])
    Fc_te = np.hstack([v for k, v in sorted(blocks_test.items())
                       if k != quantum_key])
    Fh_tr = np.hstack([v for _, v in sorted(blocks_train.items())])
    Fh_te = np.hstack([v for _, v in sorted(blocks_test.items())])
    var = float(np.var(y_test))
    table = {}
    for name, head in HEADS.items():
        nm_h = float(np.mean((y_test - head(Fh_tr, y_train, Fh_te)) ** 2)
                     / var)
        nm_c = float(np.mean((y_test - head(Fc_tr, y_train, Fc_te)) ** 2)
                     / var)
        table[name] = {"hybrid_nmse": nm_h, "classical_only_nmse": nm_c,
                       "quantum_delta": nm_c - nm_h,
                       "delta_is_zero": abs(nm_c - nm_h) < 0.01}
    return table


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 2000
    Fc = rng.normal(size=(T, 6))
    Fq = rng.normal(size=(T, 4))
    y = Fc[:, 0] - 0.5 * Fc[:, 1] ** 2 + 0.8 * Fq[:, 0] \
        + 0.3 * rng.normal(size=T)
    tr = T // 2
    tab = stacker_table({"classical": Fc[:tr], "quantum": Fq[:tr]},
                        {"classical": Fc[tr:], "quantum": Fq[tr:]},
                        y[:tr], y[tr:])
    for k, v in tab.items():
        print(f"{k:6s} hybrid {v['hybrid_nmse']:.3f} classical-only "
              f"{v['classical_only_nmse']:.3f} delta "
              f"{v['quantum_delta']:+.3f} zero={v['delta_is_zero']}")
    assert all(v["quantum_delta"] > 0.05 for v in tab.values())
    print("stacker self-test PASS (synthetic quantum signal detected by "
          "both heads)")
