"""exp_tfim1d_noisy.py -- degradation curves for the simple reservoir (stage 3).

The reservoir is stage 5's verified brickwork-ZZ restart circuit (RX
encoding + RZ bias + exp(-i theta/2 ZZ) brickwork = a Trotterised
TFIM-1D-with-drive step), imported unchanged from qrc_experiment.py; the
task is its delayed-memory target y = x_{T-1-d}. This experiment adds the
noise axis only: NMSE versus 2q depolarizing strength p2 and versus shot
count S, with the exact (statevector) ceiling measured FIRST (playbook
rule) and the mean line at NMSE = 1.0 in every table.

Scope: honest FUNCTIONAL degradation study, all Aer simulation. The task
is linearly solvable from raw inputs (declared in qrc_experiment.py); the
numbers quantify noise response, not capability.

Usage:
    python exp_tfim1d_noisy.py            full grid + figure
    python exp_tfim1d_noisy.py --check    reduced grid, no figure; exit 0
                                          iff the promotion gates pass:
                                          (i)  p2=0 sampled NMSE agrees
                                               with the exact ceiling,
                                          (ii) NMSE degrades monotonically
                                               between the noise endpoints.

Requires stage5_qubit_reuse on PYTHONPATH (the run_tests.py driver
provides it).

CAVEAT (Part IX stage 3, measured in the reuse project): gate-local noise
cannot see reuse's depth inflation; never extrapolate stage-5 noise
claims from these curves.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from noise_models import NOISE_P1, NOISE_P2, NOISE_RO, depolarizing_model
from qrc_experiment import (SEED, build_qrc_circuit, counts_to_features,
                            make_reservoir_params)

BASIS = ["rz", "sx", "x", "cx"]


def make_task(n_seq: int, n_steps: int, delay: int, seed: int):
    """Random input sequences + delayed-memory target (qrc_experiment task)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_seq, n_steps))
    y = X[:, n_steps - 1 - delay].copy()
    return X, y


def ridge_nmse(F: np.ndarray, y: np.ndarray, train_frac: float = 0.7,
               lam: float = 1e-6) -> float:
    """Chronological split, ridge fit, variance-normalised test error."""
    n_tr = int(train_frac * len(y))
    Fd = np.hstack([F, np.ones((len(y), 1))])
    A, b = Fd[:n_tr], y[:n_tr]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
    pred = Fd[n_tr:] @ w
    resid = y[n_tr:] - pred
    return float(np.mean(resid ** 2) / np.var(y[n_tr:]))


def feature_matrix(circs, n_qubits: int, shots: int | None,
                   noise_model, seed: int) -> np.ndarray:
    """Features for every sequence circuit.

    shots=None -> exact statevector probabilities (the ceiling; noise
    model ignored). Otherwise transpile once to the native basis and
    sample on Aer (density_matrix method when noisy: channels exact,
    cost ~independent of shots).
    """
    if shots is None:
        feats = []
        for qc in circs:
            sv = Statevector(qc.remove_final_measurements(inplace=False))
            feats.append(counts_to_features(sv.probabilities_dict(), n_qubits))
        return np.array(feats)
    tqcs = [transpile(qc, basis_gates=BASIS, optimization_level=0)
            for qc in circs]
    be = AerSimulator(
        noise_model=noise_model, seed_simulator=seed,
        method="density_matrix" if noise_model is not None else "automatic")
    result = be.run(tqcs, shots=shots).result()
    return np.array([counts_to_features(result.get_counts(i), n_qubits)
                     for i in range(len(tqcs))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    n_qubits, n_steps, delay = 6, 5, 2
    n_seq = 64 if args.check else 160
    shots_ref = 2048 if args.check else 4096
    p2_grid = ([0.0, NOISE_P2, 5e-2] if args.check
               else [0.0, 1e-3, 3e-3, NOISE_P2, 2e-2, 5e-2])
    shot_grid = [256, 2048] if args.check else [256, 1024, 4096, 16384]

    print(f"exp_tfim1d_noisy config: n={n_qubits} steps={n_steps} "
          f"delay={delay} n_seq={n_seq} seed={SEED} basis={BASIS} "
          f"p1={NOISE_P1} ro={NOISE_RO} p2_grid={p2_grid} "
          f"shot_grid={shot_grid} check={args.check}")

    X, y = make_task(n_seq, n_steps, delay, SEED)
    params = make_reservoir_params(n_qubits, n_steps, seed=SEED)
    circs = [build_qrc_circuit(n_qubits, n_steps, X[s], params)
             for s in range(n_seq)]

    # ceiling first (playbook rule)
    F_exact = feature_matrix(circs, n_qubits, None, None, SEED)
    nmse_exact = ridge_nmse(F_exact, y)
    print(f"\nexact ceiling (S=inf, p=0):   NMSE = {nmse_exact:.4f}")
    print("mean predictor:               NMSE = 1.0000 (by construction)")

    # curve 1: NMSE vs p2 at fixed shots
    print(f"\nNMSE vs p2 (S={shots_ref}, p1={NOISE_P1}, ro={NOISE_RO}):")
    nmse_p = {}
    for p2 in p2_grid:
        nm = (depolarizing_model(p1=NOISE_P1, p2=p2, p_ro=NOISE_RO)
              if p2 > 0 else None)
        F = feature_matrix(circs, n_qubits, shots_ref, nm, SEED)
        nmse_p[p2] = ridge_nmse(F, y)
        print(f"  p2={p2:7.4f}  NMSE = {nmse_p[p2]:.4f}")

    # curve 2: NMSE vs shots at the project noise point
    print(f"\nNMSE vs shots (project noise p1={NOISE_P1} p2={NOISE_P2} "
          f"ro={NOISE_RO}):")
    nm_std = depolarizing_model()
    nmse_s = {}
    for S in shot_grid:
        F = feature_matrix(circs, n_qubits, S, nm_std, SEED)
        nmse_s[S] = ridge_nmse(F, y)
        print(f"  S={S:6d}  NMSE = {nmse_s[S]:.4f}")

    # promotion gates
    gate_ceiling = abs(nmse_p[p2_grid[0]] - nmse_exact) < 0.08
    gate_monotone = nmse_p[p2_grid[-1]] > nmse_p[p2_grid[0]]
    gate_shots = nmse_s[shot_grid[0]] >= nmse_s[shot_grid[-1]]
    print(f"\ngates: p2=0 sampled ~ exact ceiling "
          f"[{'PASS' if gate_ceiling else 'FAIL'}], "
          f"degradation p2 endpoint [{'PASS' if gate_monotone else 'FAIL'}], "
          f"more shots not worse [{'PASS' if gate_shots else 'FAIL'}]")
    print("caveat: gate-local noise; no statement about reuse depth "
          "inflation follows from these curves.")

    if not args.check:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        ps = [p for p in p2_grid if p > 0]
        axes[0].semilogx(ps, [nmse_p[p] for p in ps], "o-")
        axes[0].axhline(nmse_exact, ls="--", c="gray", label="exact ceiling")
        axes[0].axhline(1.0, ls=":", c="k", label="mean = 1.0")
        axes[0].set_xlabel("2q depolarizing p2")
        axes[0].set_ylabel("NMSE")
        axes[0].legend(fontsize=7)
        axes[1].semilogx(shot_grid, [nmse_s[S] for S in shot_grid], "s-")
        axes[1].axhline(nmse_exact, ls="--", c="gray")
        axes[1].axhline(1.0, ls=":", c="k")
        axes[1].set_xlabel("shots S (project noise)")
        fig.tight_layout()
        fig.savefig("tfim1d_noisy_curves.png", dpi=200)
        print("wrote tfim1d_noisy_curves.png")

    ok = gate_ceiling and gate_monotone and gate_shots
    print(f"\nexit: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
