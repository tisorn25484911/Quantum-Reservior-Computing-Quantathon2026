"""qrc_experiment.py -- Quantum Reservoir Computing test case for qubit reuse.

Scope statement (honesty discipline, qrc-project-playbook):
    This experiment is a FUNCTIONAL VALIDATION of the qubit-reuse compiler
    in an application setting -- it checks that a reuse-compiled QRC
    reservoir reproduces the same measurement-feature statistics, and hence
    the same downstream linear-readout performance, using fewer physical
    qubits.  It is NOT a quantum-advantage benchmark: the delayed-memory
    target y = x_{T-1-d} is literally one of the inputs, so a classical
    readout on the raw inputs solves it exactly.  All results are noiseless
    simulation (Qiskit Aer) with finite shots.

Reservoir (windowed restart protocol, one circuit per input sequence):
    per time step t:
        RX(input_scale * x_t + offset_i) on every qubit     (encoding)
        RZ(bias[t, i]) on every qubit                       (fixed random)
        CX-RZ(theta)-CX brickwork on alternating even/odd   (entangler,
            nearest-neighbour pairs                          = exp(-i th/2 ZZ)
                                                             up to phase)
    terminal Z-basis measurement of all qubits.

The brickwork is deliberately WIDE and SHALLOW: causal cones grow by ~2
qubits per entangling layer, so for n_steps << n_qubits the cones are
local and qubit reuse compresses the circuit.  A deep or ring-coupled
reservoir (e.g. the 5-qubit CZ-ring, L=24 ENSO reservoir from the
companion QRC-Climate project) has full-register cones and admits NO
compression -- `enso_style_template_check` demonstrates that honestly.

Qubit-reuse applicability note: reuse is valid here because the QRC
readout depends only on the classical measurement outcomes (measurement-
distribution equivalence).  It would NOT be valid for protocols that need
the unmeasured post-circuit quantum state.

Endianness: Qiskit count keys are LITTLE-ENDIAN (clbit 0 = rightmost
character).  `counts_to_features` is the single reversal point; nothing
else may reorder bits.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from sklearn.linear_model import Ridge

from qreuse_ir import from_qiskit
from qreuse_analysis import compute_output_cones
from qreuse_scheduler import (
    estimate_peak_active_qubits,
    greedy_with_first_qubit_search,
)
from qreuse_compiler import compile_with_reuse
from qreuse_validation import (
    circuit_stats,
    normalize_counts,
    run_counts,
    total_variation_distance,
    validate_reuse_circuit,
)

SEED = 7  # project-wide convention


# ---------------------------------------------------------------- circuit
#: Encoding gain gamma, SELECTED BY SWEEP (never defaulted -- QRC skill).
#: Reduced-size sweep (n_seq=48, S=600, seed 7), readout NMSE direct/reuse:
#:   gamma=pi   -> 1.10 / 1.26   (RX wraps a full period over x in [0,1];
#:                                linear readout cannot invert it)
#:   gamma=pi/2 -> 0.06 / 0.01
#:   gamma=pi/4 -> 0.03 / 0.03   (best, tightest direct/reuse agreement)
#: Reproduce with encoding_gain_sweep().
INPUT_SCALE = float(np.pi / 4.0)


def make_reservoir_params(
    n_qubits: int, n_steps: int, seed: int = SEED,
    input_scale: float = INPUT_SCALE,
) -> Dict:
    """Frozen random reservoir parameters (drawn once, reused everywhere)."""
    rng = np.random.default_rng(seed)
    return {
        "input_scale": float(input_scale),
        "input_offset": rng.uniform(0.1, 0.6, size=n_qubits),
        "rz_bias": rng.uniform(0.0, 2.0 * np.pi, size=(n_steps, n_qubits)),
        "zz_angle": rng.uniform(0.4, 1.1, size=(n_steps, n_qubits)),
    }


def build_qrc_circuit(
    n_qubits: int,
    n_steps: int,
    input_sequence: Sequence[float],
    reservoir_params: Dict,
    measure_all: bool = True,
    measured_qubits: Optional[List[int]] = None,
) -> QuantumCircuit:
    """1D brickwork restart reservoir driven by `input_sequence`.

    One circuit per sequence; all measurements are terminal (a requirement
    of the qubit-reuse compiler).  measure_all=False measures only
    `measured_qubits` (default: the middle qubit), mapping qubit q to
    clbit position within that list.
    """
    if len(input_sequence) < n_steps:
        raise ValueError("input_sequence shorter than n_steps")
    p = reservoir_params
    if measure_all:
        measured_qubits = list(range(n_qubits))
    elif measured_qubits is None:
        measured_qubits = [n_qubits // 2]

    qc = QuantumCircuit(n_qubits, len(measured_qubits))
    for t in range(n_steps):
        x = float(input_sequence[t])
        for q in range(n_qubits):
            qc.rx(p["input_scale"] * x + p["input_offset"][q], q)
        for q in range(n_qubits):
            qc.rz(p["rz_bias"][t, q], q)
        for i in range(t % 2, n_qubits - 1, 2):  # alternating brickwork
            j = i + 1
            theta = float(p["zz_angle"][t, i])
            qc.cx(i, j)
            qc.rz(theta, j)
            qc.cx(i, j)
    for pos, q in enumerate(measured_qubits):
        qc.measure(q, pos)
    return qc


# --------------------------------------------------------------- features
def counts_to_features(counts: Dict[str, int], n_bits: int) -> np.ndarray:
    """Counts -> feature vector [<Z_i>]*n + [<Z_i Z_{i+1}>]*(n-1).

    THE single endianness reversal point: key[::-1] puts clbit i at
    character i.  Both the direct and the reuse-compiled circuit write
    logical output q into clbit q, so features are in logical-qubit order
    for both.
    """
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("empty counts")
    any_key = next(iter(counts))
    if len(any_key.replace(" ", "")) < n_bits:
        raise ValueError(
            f"count keys have {len(any_key.replace(' ', ''))} bits, "
            f"fewer than the requested n_bits={n_bits}"
        )
    z = np.zeros(n_bits)
    zz = np.zeros(max(n_bits - 1, 0))
    for key, c in counts.items():
        bits = key.replace(" ", "")[::-1]
        vals = np.array([1.0 - 2.0 * int(ch) for ch in bits[:n_bits]])
        z += c * vals
        if n_bits > 1:
            zz += c * vals[:-1] * vals[1:]
    return np.concatenate([z, zz]) / shots


def effective_rank(features: np.ndarray) -> float:
    """Participation ratio of the feature-covariance spectrum.

    Diagnostic from the QRC skill: nominal feature count overstates the
    usable dimensions; identical dynamics must give identical ranks.
    """
    if features.ndim != 2 or features.shape[0] < 2:
        return 0.0
    cov = np.atleast_2d(np.cov(features.T))
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    s = eig.sum()
    return float(s ** 2 / np.sum(eig ** 2)) if s > 0 else 0.0


# ------------------------------------------------------------- experiment
def run_qrc_experiment(
    n_qubits: int = 12,
    n_steps: int = 3,
    n_sequences: int = 72,
    shots: int = 600,
    delay: int = 1,
    seed: int = SEED,
    train_frac: float = 2.0 / 3.0,
    ridge_alpha: float = 1e-2,
    n_baseline: int = 10,
    input_scale: float = INPUT_SCALE,
    verbose: bool = True,
) -> Dict:
    """Direct vs reuse-compiled QRC on the delayed-memory task.

    Target: y = x_{T-1-delay} (the input `delay` steps before the last).
    Readout: Ridge regression on measurement features; NMSE is normalized
    by the train-mean predictor's test MSE, so mean predictor = 1.0.
    """
    if not (0 < delay < n_steps):
        raise ValueError("delay must satisfy 0 < delay < n_steps")
    if n_sequences < 4:
        raise ValueError("need at least 4 sequences to form a train/test split")
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_sequences, n_steps))
    y = X[:, n_steps - 1 - delay].copy()
    params = make_reservoir_params(n_qubits, n_steps, seed=seed + 1,
                                   input_scale=input_scale)
    n_baseline = max(1, min(n_baseline, n_sequences))

    if verbose:
        print(
            f"qrc_experiment config: n_qubits={n_qubits} n_steps={n_steps} "
            f"n_sequences={n_sequences} shots={shots} delay={delay} "
            f"seed={seed} ridge_alpha={ridge_alpha} "
            f"input_scale={input_scale:.6f}"
        )

    # The circuit STRUCTURE is input-independent (only angles change), so
    # cones and the measurement order are computed once and reused.
    template = build_qrc_circuit(n_qubits, n_steps, X[0], params)
    ir0 = from_qiskit(template)
    cones = compute_output_cones(ir0)
    order = greedy_with_first_qubit_search(cones)
    n_phys = estimate_peak_active_qubits(order, cones)

    feats_direct: List[np.ndarray] = []
    feats_reuse: List[np.ndarray] = []
    feats_baseline: List[np.ndarray] = []  # direct rerun, new sampling seed
    tvds: List[float] = []
    tvd_baseline: List[float] = []
    stats_direct = stats_reuse = None

    for i in range(n_sequences):
        qc = build_qrc_circuit(n_qubits, n_steps, X[i], params)
        ir = from_qiskit(qc)
        compiled = compile_with_reuse(ir, order=order)
        if i == 0:
            validate_reuse_circuit(compiled, ir)
            stats_direct = circuit_stats(qc)
            stats_reuse = circuit_stats(compiled.circuit)
        cd = run_counts(qc, shots=shots, seed=seed + 1000 + i)
        cr = run_counts(compiled.circuit, shots=shots, seed=seed + 5000 + i)
        feats_direct.append(counts_to_features(cd, n_qubits))
        feats_reuse.append(counts_to_features(cr, n_qubits))
        tvds.append(
            total_variation_distance(normalize_counts(cd),
                                     normalize_counts(cr))
        )
        if i < n_baseline:  # shot-noise floor: direct vs direct, new seed
            cd2 = run_counts(qc, shots=shots, seed=seed + 9000 + i)
            tvd_baseline.append(
                total_variation_distance(normalize_counts(cd),
                                         normalize_counts(cd2))
            )
            feats_baseline.append(counts_to_features(cd2, n_qubits))

    Fd = np.array(feats_direct)
    Fr = np.array(feats_reuse)
    tvds_arr = np.array(tvds)
    tvd_base_arr = np.array(tvd_baseline)

    # Per-qubit marginal TVD: |P1_direct - P1_reuse| = |<Z>_d - <Z>_r| / 2.
    # Sharper equivalence statistic than full-bitstring TVD when the
    # outcome space (2^n) dwarfs the shot count.
    Fb = np.array(feats_baseline)
    marg = np.abs(Fd[:, :n_qubits] - Fr[:, :n_qubits]) / 2.0
    marg_base = np.abs(Fd[:len(Fb), :n_qubits] - Fb[:, :n_qubits]) / 2.0
    marginal_tvd_mean = float(marg.mean())
    marginal_tvd_baseline_mean = float(marg_base.mean())

    n_train = int(round(train_frac * n_sequences))
    n_train = min(max(n_train, 2), n_sequences - 1)  # non-empty train & test
    y_tr, y_te = y[:n_train], y[n_train:]
    mean_pred_mse = float(np.mean((y_te - y_tr.mean()) ** 2))

    def fit_eval(F: np.ndarray) -> Dict:
        model = Ridge(alpha=ridge_alpha).fit(F[:n_train], y_tr)
        pred_te = model.predict(F[n_train:])
        mse_te = float(np.mean((pred_te - y_te) ** 2))
        return {
            "model": model,
            "mse_train": float(
                np.mean((model.predict(F[:n_train]) - y_tr) ** 2)
            ),
            "mse_test": mse_te,
            "nmse_test": mse_te / mean_pred_mse,
            "pred_test": pred_te,
        }

    res_d = fit_eval(Fd)
    res_r = fit_eval(Fr)
    # transfer check: the DIRECT-trained readout applied to REUSE features
    pred_cross = res_d["model"].predict(Fr[n_train:])
    mse_cross = float(np.mean((pred_cross - y_te) ** 2))
    feature_corr = float(np.corrcoef(Fd.ravel(), Fr.ravel())[0, 1])

    results = {
        "config": {
            "n_qubits": n_qubits, "n_steps": n_steps,
            "n_sequences": n_sequences, "shots": shots, "delay": delay,
            "seed": seed, "n_train": n_train, "ridge_alpha": ridge_alpha,
        },
        "logical_qubits": n_qubits,
        "physical_qubits": n_phys,
        "measurement_order": order,
        "stats_direct": stats_direct,
        "stats_reuse": stats_reuse,
        "tvd_mean": float(tvds_arr.mean()),
        "tvd_baseline_mean": float(tvd_base_arr.mean()),
        "tvds": tvds_arr,
        "tvds_baseline": tvd_base_arr,
        "marginal_tvd_mean": marginal_tvd_mean,
        "marginal_tvd_baseline_mean": marginal_tvd_baseline_mean,
        "feature_corr": feature_corr,
        "effective_rank_direct": effective_rank(Fd),
        "effective_rank_reuse": effective_rank(Fr),
        "mse_test_direct": res_d["mse_test"],
        "mse_test_reuse": res_r["mse_test"],
        "nmse_test_direct": res_d["nmse_test"],
        "nmse_test_reuse": res_r["nmse_test"],
        "mse_test_cross": mse_cross,
        "nmse_test_cross": mse_cross / mean_pred_mse,
        "mean_pred_mse": mean_pred_mse,
        "y_test": y_te,
        "pred_test_direct": res_d["pred_test"],
        "pred_test_reuse": res_r["pred_test"],
        "features_direct": Fd,
        "features_reuse": Fr,
    }

    if verbose:
        _print_summary(results)
    return results


def _print_summary(r: Dict) -> None:
    sd, sr = r["stats_direct"], r["stats_reuse"]
    print("\n--- QRC direct vs reuse-compiled summary ---")
    print(f"qubits           : {r['logical_qubits']} logical -> "
          f"{r['physical_qubits']} physical "
          f"({r['physical_qubits'] / r['logical_qubits']:.0%} of original)")
    print(f"depth            : {sd['depth']} -> {sr['depth']}   "
          f"resets: {sr['resets']}   "
          f"mid-circuit measurements: {sr['mid_circuit_measurements']}")
    print(f"TVD(direct,reuse): {r['tvd_mean']:.4f} mean   "
          f"[shot-noise floor {r['tvd_baseline_mean']:.4f}]")
    print(f"marginal TVD     : {r['marginal_tvd_mean']:.4f} mean per qubit "
          f"[shot-noise floor {r['marginal_tvd_baseline_mean']:.4f}]")
    print(f"feature corr     : {r['feature_corr']:.4f}   "
          f"effective rank: {r['effective_rank_direct']:.2f} direct / "
          f"{r['effective_rank_reuse']:.2f} reuse")
    print(f"readout NMSE     : direct {r['nmse_test_direct']:.4f}   "
          f"reuse {r['nmse_test_reuse']:.4f}   "
          f"direct-model-on-reuse-features {r['nmse_test_cross']:.4f}   "
          f"(mean predictor = 1.0)")
    print(f"readout MSE      : direct {r['mse_test_direct']:.5f}   "
          f"reuse {r['mse_test_reuse']:.5f}")


# ---------------------------------------------------- compile-only scans
def encoding_gain_sweep(
    gains: Sequence[float] = (np.pi, np.pi / 2.0, np.pi / 4.0),
    n_sequences: int = 48,
    shots: int = 600,
    seed: int = SEED,
) -> List[Dict]:
    """Reduced-size sweep of the encoding gain gamma (QRC-skill rule:
    never default it).  Reproduces the numbers recorded at INPUT_SCALE.

    `input_scale` is a real parameter of run_qrc_experiment, so this sweep
    needs no monkeypatching of module globals.
    """
    out = []
    for g in gains:
        r = run_qrc_experiment(
            n_sequences=n_sequences, shots=shots, seed=seed,
            input_scale=float(g), verbose=False,
        )
        out.append({
            "gamma": float(g),
            "nmse_direct": r["nmse_test_direct"],
            "nmse_reuse": r["nmse_test_reuse"],
            "feature_corr": r["feature_corr"],
        })
        print(f"  gamma={g:.4f}  NMSE direct {r['nmse_test_direct']:.3f}  "
              f"reuse {r['nmse_test_reuse']:.3f}  "
              f"corr {r['feature_corr']:.4f}")
    return out


def qubit_scaling_scan(
    widths: Sequence[int] = (8, 12, 16, 20, 24, 28, 32, 40),
    n_steps: int = 3,
    seed: int = SEED,
) -> List[Dict]:
    """Compiled vs logical qubit count at fixed depth (no simulation)."""
    out = []
    for n in widths:
        params = make_reservoir_params(n, n_steps, seed=seed + 1)
        qc = build_qrc_circuit(n, n_steps, [0.5] * n_steps, params)
        cones = compute_output_cones(from_qiskit(qc))
        order = greedy_with_first_qubit_search(cones)
        out.append({
            "n_qubits": n,
            "n_steps": n_steps,
            "physical": estimate_peak_active_qubits(order, cones),
        })
    return out


def depth_scaling_scan(
    n_qubits: int = 16,
    steps_list: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10),
    seed: int = SEED,
) -> List[Dict]:
    """Compression vs reservoir depth at fixed width.

    Shows the honest limit: as causal cones grow with depth, compression
    vanishes (physical -> logical), exactly as for deep QAOA in the paper.
    """
    out = []
    for T in steps_list:
        params = make_reservoir_params(n_qubits, T, seed=seed + 1)
        qc = build_qrc_circuit(n_qubits, T, [0.5] * T, params)
        cones = compute_output_cones(from_qiskit(qc))
        order = greedy_with_first_qubit_search(cones)
        out.append({
            "n_qubits": n_qubits,
            "n_steps": T,
            "physical": estimate_peak_active_qubits(order, cones),
        })
    return out


def enso_style_template_check(
    n_qubits: int = 5, n_steps: int = 24, seed: int = SEED
) -> Dict:
    """Compile-only check on an ENSO-project-style reservoir template.

    Mirrors the companion QRC-Climate reservoir structure (per step:
    RY encoding on every qubit, CZ ring, RY/RZ entangler rotations;
    terminal Z measurement; n=5, L=24).  The ring coupling makes every
    causal cone the full register after ~n/2 steps, so NO compression is
    expected -- reported honestly rather than hidden.
    """
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.5, 1.0, size=n_qubits)
    a = rng.uniform(0.0, np.pi, size=n_qubits)
    b = rng.uniform(0.0, np.pi, size=n_qubits)
    u = rng.uniform(0.0, 1.0, size=n_steps)
    qc = QuantumCircuit(n_qubits, n_qubits)
    for t in range(n_steps):
        for q in range(n_qubits):
            qc.ry((np.pi / 4.0) * g[q] * u[t], q)
        for i in range(n_qubits):
            qc.cz(i, (i + 1) % n_qubits)  # ring
        for q in range(n_qubits):
            qc.ry(a[q], q)
            qc.rz(b[q], q)
    qc.measure(range(n_qubits), range(n_qubits))
    cones = compute_output_cones(from_qiskit(qc))
    order = greedy_with_first_qubit_search(cones)
    return {
        "n_qubits": n_qubits,
        "n_steps": n_steps,
        "physical": estimate_peak_active_qubits(order, cones),
        "cone_sizes": sorted(len(c) for c in cones.values()),
    }


# ----------------------------------------------------------------- plots
def make_figures(
    results: Dict,
    width_scan: List[Dict],
    depth_scan: List[Dict],
    path: str = "qrc_reuse_results.png",
) -> str:
    """Four-panel results figure; every number shown is computed in-run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    w_log = [s["n_qubits"] for s in width_scan]
    w_phys = [s["physical"] for s in width_scan]
    ax.plot(w_log, w_log, "k--", lw=1, label="no reuse (y = x)")
    ax.plot(w_log, w_phys, "o-", color="tab:blue",
            label=f"reuse-compiled (T={width_scan[0]['n_steps']})")
    d_T = [s["n_steps"] for s in depth_scan]
    d_phys = [s["physical"] for s in depth_scan]
    ax2 = ax.twiny()
    ax2.plot(d_T, d_phys, "s-", color="tab:red", alpha=0.7,
             label=f"depth scan (n={depth_scan[0]['n_qubits']})")
    ax2.set_xlabel("reservoir steps T (red)", color="tab:red")
    ax2.tick_params(axis="x", colors="tab:red")
    ax.set_xlabel("logical qubits (blue)")
    ax.set_ylabel("compiled physical qubits")
    ax.set_title("qubit-reuse compression: width helps, depth erodes")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)

    ax = axes[0, 1]
    Fd, Fr = results["features_direct"], results["features_reuse"]
    ax.plot([-1, 1], [-1, 1], "k--", lw=1)
    ax.plot(Fd.ravel(), Fr.ravel(), ".", ms=3, alpha=0.4, color="tab:blue")
    ax.set_xlabel("direct-circuit feature value")
    ax.set_ylabel("reuse-circuit feature value")
    ax.set_title(
        f"feature agreement (r = {results['feature_corr']:.4f}, "
        f"S = {results['config']['shots']})"
    )

    ax = axes[1, 0]
    y_te = results["y_test"]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(y_te, results["pred_test_direct"], "o", ms=5, alpha=0.75,
            label=f"direct (NMSE {results['nmse_test_direct']:.3f})")
    ax.plot(y_te, results["pred_test_reuse"], "s", ms=5, alpha=0.75,
            label=f"reuse (NMSE {results['nmse_test_reuse']:.3f})")
    ax.set_xlabel(f"true target  y = x[t-{results['config']['delay']}]")
    ax.set_ylabel("ridge readout prediction (test)")
    ax.set_title("delayed-memory readout (mean predictor = 1.0)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    bins = np.linspace(
        0.0,
        max(results["tvds"].max(), results["tvds_baseline"].max()) * 1.15,
        18,
    )
    ax.hist(results["tvds"], bins=bins, alpha=0.65, color="tab:blue",
            label=f"TVD(direct, reuse), mean {results['tvd_mean']:.3f}")
    ax.hist(results["tvds_baseline"], bins=bins, alpha=0.65,
            color="tab:gray",
            label=(f"TVD(direct, direct') shot-noise floor, "
                   f"mean {results['tvd_baseline_mean']:.3f}"))
    ax.set_xlabel("total variation distance")
    ax.set_ylabel("sequences")
    ax.set_title("measurement-distribution equivalence")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"QRC qubit-reuse test: {results['logical_qubits']} logical -> "
        f"{results['physical_qubits']} physical qubits "
        f"(simulation, {results['config']['shots']} shots)",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    res = run_qrc_experiment()
    wscan = qubit_scaling_scan()
    dscan = depth_scaling_scan()
    print("\nwidth scan :", [(s["n_qubits"], s["physical"]) for s in wscan])
    print("depth scan :", [(s["n_steps"], s["physical"]) for s in dscan])
    print("ENSO-style :", enso_style_template_check())
    print("figure     :", make_figures(res, wscan, dscan))
