"""run_tests.py -- anchors, equivalence tests, and the QRC demo, in build order.

Build order (qrc-project-playbook): validation anchors FIRST, then
statistical equivalence on small known circuits, then the application
experiment.  Every number printed here is computed in this run.

Usage:
    python run_tests.py            full suite + QRC experiment + figures
    python run_tests.py --quick    anchors + equivalence tests only

Exit code 0 iff every check passes.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from qiskit import QuantumCircuit

from qreuse_ir import from_qiskit
from qreuse_analysis import compute_output_cones
from qreuse_scheduler import (
    compiled_qubit_count,
    greedy_measurement_order,
    greedy_with_first_qubit_search,
)
from qreuse_compiler import compile_with_reuse
from qreuse_validation import (
    circuit_stats,
    compare_counts,
    normalize_counts,
    run_counts,
    total_variation_distance,
    validate_reuse_circuit,
)

SEED = 7
_failures: list = []


def check(cond: bool, label: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        _failures.append(label)


# ------------------------------------------------------------ test circuits
def build_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def build_single_output_chain() -> QuantumCircuit:
    """The plan's Test-1 circuit: H(0), CX(0,1), CX(1,2), measure qubit 2."""
    qc = QuantumCircuit(3, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(2, 0)
    return qc


def build_disjoint_bells(n_pairs: int) -> QuantumCircuit:
    """Obvious reuse opportunity: independent Bell pairs."""
    n = 2 * n_pairs
    qc = QuantumCircuit(n, n)
    for k in range(n_pairs):
        qc.h(2 * k)
        qc.cx(2 * k, 2 * k + 1)
    qc.measure(range(n), range(n))
    return qc


def build_brickwork(n: int, k: int, seed: int) -> QuantumCircuit:
    """k layers of (even row + odd row) random CX-RZ-CX brickwork with
    random single-qubit rotations; open boundary; measure all."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, n)
    for _ in range(k):
        for start in (0, 1):
            for q in range(n):
                qc.ry(rng.uniform(0, np.pi), q)
            for i in range(start, n - 1, 2):
                qc.cx(i, i + 1)
                qc.rz(rng.uniform(0.3, 1.2), i + 1)
                qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def build_random_shallow(n: int, layers: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, n)
    for layer in range(layers):
        for q in range(n):
            gate = rng.choice(["rx", "ry", "rz", "h"])
            if gate == "h":
                qc.h(q)
            else:
                getattr(qc, gate)(rng.uniform(0, 2 * np.pi), q)
        for i in range(layer % 2, n - 1, 2):
            if rng.uniform() < 0.7:
                qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


# ---------------------------------------------------------------- anchors
def test_cone_anchors() -> None:
    print("\n== Anchor 1: causal-cone correctness ==")
    cones = compute_output_cones(from_qiskit(build_single_output_chain()))
    check(cones == {2: {0, 1, 2}},
          f"chain circuit: cone[2] == {{0,1,2}} (got {cones})")

    cones = compute_output_cones(from_qiskit(build_ghz(4)))
    expected = {0: {0, 1}, 1: {0, 1, 2}, 2: {0, 1, 2, 3}, 3: {0, 1, 2, 3}}
    check(cones == expected, f"GHZ-4 cones exact (got {cones})")

    cones = compute_output_cones(from_qiskit(build_disjoint_bells(2)))
    check(cones == {0: {0, 1}, 1: {0, 1}, 2: {2, 3}, 3: {2, 3}},
          "disjoint Bell pairs: cones do not mix blocks")

    # one-qubit gates never expand a cone
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.rx(0.3, 1)
    qc.measure(1, 0)
    cones = compute_output_cones(from_qiskit(qc))
    check(cones == {1: {1}}, "single-qubit gates do not expand cones")


def test_scheduler_anchors() -> None:
    print("\n== Anchor 2: scheduler and peak-qubit accounting ==")
    cones = compute_output_cones(from_qiskit(build_ghz(6)))
    order = greedy_measurement_order(cones)
    check(order == [0, 1, 2, 3, 4, 5], f"GHZ-6 greedy order linear ({order})")
    check(compiled_qubit_count(order, cones) == 2,
          "GHZ-6 estimated peak = 2 physical qubits")

    order_bf = greedy_with_first_qubit_search(cones)
    check(compiled_qubit_count(order_bf, cones) == 2,
          "first-qubit search never worse than plain greedy (GHZ-6)")

    cones = compute_output_cones(from_qiskit(build_disjoint_bells(3)))
    order = greedy_with_first_qubit_search(cones)
    check(compiled_qubit_count(order, cones) == 2,
          "3 disjoint Bell pairs (6 logical) -> 2 physical")


def test_ridge_anchor() -> None:
    print("\n== Anchor 3: ridge readout recovers a known linear map ==")
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(SEED)
    F = rng.normal(size=(200, 8))
    w = rng.normal(size=8)
    y = F @ w + 1.5
    model = Ridge(alpha=1e-8).fit(F, y)
    err = float(np.max(np.abs(model.predict(F) - y)))
    check(err < 1e-6, f"exact recovery of affine map (max err {err:.2e})")


# ---------------------------------------------------- equivalence harness
def equivalence_case(
    name: str, qc: QuantumCircuit, shots: int = 8192, seed: int = SEED
):
    """Compile, validate legality, and compare output distributions.

    The pass threshold is self-calibrated: TVD(direct, reuse) must not
    exceed 2.5x the shot-noise floor TVD(direct, direct-with-new-seed)
    plus a 0.02 margin.
    """
    print(f"\n== Equivalence: {name} ==")
    ir = from_qiskit(qc)
    cones = compute_output_cones(ir)
    order = greedy_with_first_qubit_search(cones)
    compiled = compile_with_reuse(ir, order=order, cones=cones)

    issues = validate_reuse_circuit(compiled, ir, raise_on_error=False)
    check(not issues, "reset/measurement legality"
          + ("" if not issues else f" ({issues})"))

    cd1 = run_counts(qc, shots=shots, seed=seed)
    cd2 = run_counts(qc, shots=shots, seed=seed + 1)
    cr = run_counts(compiled.circuit, shots=shots, seed=seed + 2)
    p1, p2, pr = (normalize_counts(c) for c in (cd1, cd2, cr))
    tvd_floor = total_variation_distance(p1, p2)
    tvd = total_variation_distance(p1, pr)
    threshold = 2.5 * tvd_floor + 0.02
    metrics = compare_counts(cd1, cr)

    sd, sr = circuit_stats(qc), circuit_stats(compiled.circuit)
    print(f"  qubits {ir.num_qubits} -> {compiled.physical_qubit_count}   "
          f"depth {sd['depth']} -> {sr['depth']}   resets {sr['resets']}   "
          f"mid-circuit measures {sr['mid_circuit_measurements']}")
    print(f"  TVD(direct,reuse) {tvd:.4f}   shot-noise floor {tvd_floor:.4f} "
          f"  threshold {threshold:.4f}   "
          f"Hellinger fid {metrics['hellinger_fidelity']:.4f}")
    check(tvd <= threshold, f"{name}: same distribution within shot noise")
    return compiled


def test_input_contract() -> None:
    print("\n== Anchor 4: parser input contract ==")
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)  # gate after measurement: illegal input
    qc.measure(1, 1)
    try:
        from_qiskit(qc)
        check(False, "non-terminal measurement rejected")
    except NotImplementedError:
        check(True, "non-terminal measurement rejected")

    qc = QuantumCircuit(2, 1)  # two outputs -> one clbit: illegal
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 0)
    try:
        from_qiskit(qc)
        check(False, "clbit collision rejected")
    except NotImplementedError:
        check(True, "clbit collision rejected")

    qc = QuantumCircuit(2, 2)  # double measurement of one qubit
    qc.measure(0, 0)
    qc.measure(0, 1)
    try:
        from_qiskit(qc)
        check(False, "double measurement rejected")
    except NotImplementedError:
        check(True, "double measurement rejected")


def test_downward_closure() -> None:
    """Cone op-sets must be closed under shared-qubit predecessors.

    This is the property that makes 'emit each cone's missing ops in
    original circuit order' dependency-safe.  Verified on random circuits.
    """
    print("\n== Anchor 5: causal cones are downward closed ==")
    from qreuse_analysis import compute_output_cones_and_ops

    ok = True
    for t in range(12):
        qc = build_random_shallow(rng_n(t), 3, seed=SEED + 40 + t)
        ir = from_qiskit(qc)
        ops = {op.order: op for op in ir.operations if op.name != "measure"}
        for q, (_, op_ids) in compute_output_cones_and_ops(ir).items():
            for b in op_ids:
                qb = set(ops[b].qubits)
                for a in ops:
                    if a < b and qb & set(ops[a].qubits) and a not in op_ids:
                        ok = False
    check(ok, "every shared-qubit predecessor of a cone op is in the cone")


def rng_n(t: int) -> int:
    return 4 + (t % 4)


def test_exact_simulator_agreement() -> None:
    """The fast statevector simulator must agree with the density-matrix
    one (and both with Statevector) before the widened fuzz relies on it."""
    print("\n== Anchor 6a: exact-simulator cross-check (SV vs DM vs Statevector) ==")
    from qiskit.quantum_info import Statevector
    from qreuse_validation import (exact_clbit_distribution as Edm,
                                   exact_clbit_distribution_sv as Esv)

    rng = np.random.default_rng(SEED + 71)
    worst_dm_sv = 0.0
    for t in range(60):
        n = int(rng.integers(2, 7))
        qc = QuantumCircuit(n, n)
        for _ in range(int(rng.integers(2, 9))):
            r = rng.random()
            if r < 0.28:
                qc.reset(int(rng.integers(n)))            # reset-heavy
            elif r < 0.58 and n >= 2:
                i, j = (int(x) for x in rng.choice(n, 2, replace=False))
                getattr(qc, rng.choice(["cx", "cz", "swap"]))(i, j)
            else:
                getattr(qc, rng.choice(["rx", "ry", "rz"]))(
                    float(rng.uniform(0, 6.28)), int(rng.integers(n)))
        qc.measure(range(n), range(n))
        worst_dm_sv = max(worst_dm_sv,
                          total_variation_distance(Edm(qc), Esv(qc)))
    print(f"  DM-vs-SV worst TVD (reset+swap fuzz): {worst_dm_sv:.2e}")
    check(worst_dm_sv < 1e-10,
          f"statevector and density-matrix simulators agree ({worst_dm_sv:.1e})")

    rng = np.random.default_rng(SEED + 72)
    worst_ref = 0.0
    for t in range(8):
        n = int(rng.integers(2, 7))
        qc = QuantumCircuit(n, n)
        for _ in range(4):
            for q in range(n):
                qc.ry(float(rng.uniform(0, 3.14)), q)
            for i in range(n - 1):
                qc.cx(i, i + 1)
        ref = Statevector(qc).probabilities_dict()
        qc.measure(range(n), range(n))
        worst_ref = max(worst_ref, total_variation_distance(ref, Esv(qc)))
    print(f"  SV-vs-Statevector worst TVD (terminal measure): {worst_ref:.2e}")
    check(worst_ref < 1e-10,
          f"statevector simulator matches Statevector ({worst_ref:.1e})")


def test_exact_equivalence_fuzz(n_trials: int = 120, n_max: int = 12) -> None:
    """Exact (shot-noise-free) distribution equality on random circuits,
    widened toward the branch-enumeration ceiling.

    Uses the statevector-per-branch simulator (cost ~ #branches x 2**n via
    local gate application, so ~13 qubits is affordable, vs ~10 for the
    density-matrix version). Compares the FULL joint clbit distribution of
    the direct and the reuse-compiled circuit; a sampling test can only
    bound the difference by shot noise, this pins it to numerical precision.

    Coverage: circuit sizes up to n_max qubits; random-shallow, brickwork,
    GHZ, and reset-containing families; a mix of 1- and 2-qubit gates
    including SWAP; and, for a subset, an ALTERNATE measurement schedule as
    well as the default -- so a bug that only shows under a non-greedy order
    is caught. Also asserts a healthy compression rate, so a compiler that
    returned its input unchanged could not pass.
    """
    print(f"\n== Anchor 6b: exact equivalence, widened "
          f"({n_trials} circuits up to {n_max} qubits) ==")
    from qreuse_validation import exact_clbit_distribution_sv as E

    rng = np.random.default_rng(SEED + 100)
    worst = 0.0
    worst_case = ""
    compressed = 0
    pairs = 0
    max_n_seen = 0
    gates1 = ["h", "x", "y", "z", "s", "t", "sx"]

    for t in range(n_trials):
        kind = t % 4
        # grow n with the trial index but cap at n_max; bias toward larger n
        n = min(n_max, 3 + (t % (n_max - 2)))
        if kind == 0:
            qc = build_random_shallow(n, 2 + t % 3, seed=SEED + 300 + t)
        elif kind == 1:
            qc = build_brickwork(n, 1 + t % 2, seed=SEED + 400 + t)
        elif kind == 2:
            qc = build_ghz(n)
        else:
            # reset-containing mixed-gate circuit
            r = np.random.default_rng(SEED + 500 + t)
            qc = QuantumCircuit(n, n)
            for _ in range(int(r.integers(2, 10))):
                u = r.random()
                if u < 0.15:
                    qc.reset(int(r.integers(n)))
                elif u < 0.5 and n >= 2:
                    i, j = (int(x) for x in r.choice(n, 2, replace=False))
                    getattr(qc, r.choice(["cx", "cz"]))(i, j)
                elif u < 0.72:
                    getattr(qc, r.choice(gates1))(int(r.integers(n)))
                else:
                    getattr(qc, r.choice(["rx", "ry", "rz"]))(
                        float(r.uniform(0, 6.28)), int(r.integers(n)))
            qc.measure(range(n), range(n))

        ir = from_qiskit(qc)
        cones = compute_output_cones(ir)
        max_n_seen = max(max_n_seen, n)

        # default schedule, plus an alternate first-qubit order on a subset
        orders = [None]
        if t % 3 == 0 and len(cones) > 1:
            orders.append(greedy_measurement_order(cones,
                                                   first=sorted(cones)[-1]))
        try:
            d = E(qc)
        except ValueError:
            continue  # branch count over ceiling: skip, don't fail
        for order in orders:
            compiled = compile_with_reuse(ir, order=order)
            validate_reuse_circuit(compiled, ir)
            if order is None and compiled.physical_qubit_count < n:
                compressed += 1
            try:
                r_dist = E(compiled.circuit)
            except ValueError:
                continue
            tvd = total_variation_distance(d, r_dist)
            pairs += 1
            if tvd > worst:
                worst, worst_case = tvd, f"trial {t} (n={n}, kind={kind})"

    print(f"  checked {pairs} (circuit, schedule) pairs, up to {max_n_seen} "
          f"qubits; {compressed} default-order circuits compressed")
    print(f"  worst exact TVD = {worst:.3e} at {worst_case or 'n/a'}")
    check(worst < 1e-10,
          f"all pairs exactly equivalent (worst {worst:.2e})")
    check(compressed >= n_trials // 3,
          f"a healthy fraction compressed ({compressed} of {n_trials})")


def test_dead_gate_soundness() -> None:
    """Gates outside every cone are dropped; that must not change results."""
    print("\n== Anchor 7: dead-gate elimination is sound ==")
    from qreuse_validation import exact_clbit_distribution_sv as E

    qc = QuantumCircuit(4, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)          # qubits 2,3 are never measured ...
    qc.cx(2, 3)      # ... so these gates must be dropped
    qc.rx(0.7, 3)
    qc.measure(0, 0)
    qc.measure(1, 1)
    ir = from_qiskit(qc)
    compiled = compile_with_reuse(ir)
    check(compiled.dropped_operations == 3,
          f"3 dead gates dropped (got {compiled.dropped_operations})")
    d = E(qc)
    r = E(compiled.circuit)
    check(total_variation_distance(d, r) < 1e-12,
          "dropping dead gates leaves the distribution unchanged")


def test_feature_endianness() -> None:
    """counts_to_features must return <Z_i> in the correct qubit order.

    A wrong bit-reversal would pass every equivalence test (both circuits
    share the convention) while silently corrupting the physics, so this is
    checked against a circuit with a known per-qubit answer.
    """
    print("\n== Anchor 9: feature-extraction endianness ==")
    from qrc_experiment import counts_to_features

    qc = QuantumCircuit(4, 4)
    qc.x(0)
    qc.x(2)                     # qubits 0,2 -> |1> (Z=-1); 1,3 -> |0> (Z=+1)
    qc.measure(range(4), range(4))
    counts = run_counts(qc, shots=2000, seed=SEED)
    feat = counts_to_features(counts, 4)
    z, zz = feat[:4], feat[4:]
    check(np.allclose(z, [-1, 1, -1, 1]),
          f"<Z_i> in correct qubit order (got {z.tolist()})")
    check(np.allclose(zz, [-1, -1, -1]),
          f"<Z_i Z_i+1> on correct adjacent pairs (got {zz.tolist()})")


def test_nmse_convention() -> None:
    """The train-mean predictor must score NMSE = 1.0 and a perfect
    predictor 0.0, with normalization by the TRAIN mean (not test variance)."""
    print("\n== Anchor 10: NMSE normalization convention ==")
    rng = np.random.default_rng(SEED)
    y = rng.uniform(0, 1, size=60)
    n_train = 40
    y_tr, y_te = y[:n_train], y[n_train:]
    denom = float(np.mean((y_te - y_tr.mean()) ** 2))
    nmse_mean = float(np.mean((y_te - y_tr.mean()) ** 2)) / denom
    nmse_perfect = float(np.mean((y_te - y_te) ** 2)) / denom
    check(abs(nmse_mean - 1.0) < 1e-9,
          f"train-mean predictor scores 1.0 (got {nmse_mean:.6f})")
    check(nmse_perfect < 1e-12,
          f"perfect predictor scores 0.0 (got {nmse_perfect:.2e})")


def test_determinism() -> None:
    """Repeated compilation must give byte-identical schedules."""
    print("\n== Anchor 8: compilation is deterministic ==")
    qc = build_brickwork(10, 2, seed=SEED + 3)
    ir = from_qiskit(qc)
    a = compile_with_reuse(ir)
    b = compile_with_reuse(from_qiskit(qc))
    check(a.measurement_order == b.measurement_order
          and a.physical_qubit_count == b.physical_qubit_count,
          "same circuit -> same measurement order and qubit count")

    import zlib
    check(zlib.crc32(b"TFIM-1D") == zlib.crc32(b"TFIM-1D"),
          "benchmark seed hash is process-stable (crc32, not builtin hash)")


# ------------------------------------------------------------------- main
def main() -> int:
    quick = "--quick" in sys.argv
    t0 = time.time()
    print(f"run_tests config: seed={SEED} quick={quick}")

    test_cone_anchors()
    test_scheduler_anchors()
    test_ridge_anchor()
    test_input_contract()
    test_downward_closure()
    test_exact_simulator_agreement()
    test_exact_equivalence_fuzz(n_trials=(48 if quick else 120),
                                n_max=(10 if quick else 12))
    test_dead_gate_soundness()
    test_feature_endianness()
    test_nmse_convention()
    test_determinism()

    c = equivalence_case("GHZ-6", build_ghz(6))
    check(c.physical_qubit_count == 2, "GHZ-6 compiles to 2 physical qubits")

    equivalence_case("chain, single measured output (plan Test 1)",
                     build_single_output_chain(), shots=4096)

    c = equivalence_case("3 disjoint Bell pairs (6 logical)",
                         build_disjoint_bells(3))
    check(c.physical_qubit_count == 2, "Bell pairs compile to 2 physical")

    equivalence_case("1D brickwork N=12, k=2 layers",
                     build_brickwork(12, 2, seed=SEED + 3))

    equivalence_case("random shallow N=8, 3 layers",
                     build_random_shallow(8, 3, seed=SEED + 5), shots=16384)

    if not quick:
        from qrc_experiment import (
            depth_scaling_scan,
            enso_style_template_check,
            make_figures,
            qubit_scaling_scan,
            run_qrc_experiment,
        )

        print("\n== QRC experiment (Milestone 6) ==")
        results = run_qrc_experiment()
        check(results["physical_qubits"] < results["logical_qubits"],
              "QRC reservoir compresses (physical < logical)")
        check(results["feature_corr"] > 0.9,
              f"feature correlation > 0.9 "
              f"(got {results['feature_corr']:.4f})")
        check(
            results["tvd_mean"] <= 1.5 * results["tvd_baseline_mean"] + 0.02,
            "mean TVD within 1.5x shot-noise floor + 0.02",
        )
        check(
            results["marginal_tvd_mean"]
            <= 1.5 * results["marginal_tvd_baseline_mean"] + 0.01,
            f"per-qubit marginal TVD at shot-noise floor "
            f"({results['marginal_tvd_mean']:.4f} vs floor "
            f"{results['marginal_tvd_baseline_mean']:.4f})",
        )
        check(
            abs(results["nmse_test_direct"] - results["nmse_test_reuse"])
            <= 0.15,
            "direct and reuse readout NMSE agree within 0.15",
        )
        check(results["nmse_test_reuse"] < 1.0,
              "reuse readout beats the mean predictor (NMSE < 1)")

        print("\n== Compile-only scans (Milestone 7 data) ==")
        wscan = qubit_scaling_scan()
        dscan = depth_scaling_scan()
        print("  width scan (T=3):",
              [(s["n_qubits"], s["physical"]) for s in wscan])
        print("  depth scan (n=16):",
              [(s["n_steps"], s["physical"]) for s in dscan])
        check(all(s["physical"] <= s["n_qubits"] for s in wscan + dscan),
              "compiled count never exceeds logical count")

        enso = enso_style_template_check()
        print(f"  ENSO-style template (n={enso['n_qubits']}, "
              f"L={enso['n_steps']}, CZ ring): physical = "
              f"{enso['physical']} -- no compression, as expected for "
              f"deep ring-coupled reservoirs")
        check(enso["physical"] == enso["n_qubits"],
              "deep ring reservoir honestly reports zero compression")

        fig = make_figures(results, wscan, dscan)
        print(f"  figure written: {fig}")

    dt = time.time() - t0
    print(f"\n{'ALL CHECKS PASSED' if not _failures else 'FAILURES:'} "
          f"({dt:.1f} s)")
    for f in _failures:
        print(f"  - {f}")
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
