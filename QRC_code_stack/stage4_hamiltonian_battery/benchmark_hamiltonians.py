"""benchmark_hamiltonians.py -- qubit-reuse benchmark across Hamiltonian families.

Grid: for each Hamiltonian, exactly FOUR sampled runs
    (direct | reuse-compiled)  x  (ideal simulation | noisy simulation)
plus one exact statevector reference distribution (not a sampled run).

The reference is the exact output distribution OF THE TROTTER CIRCUIT
ITSELF, not of the true e^{-iHt}.  Trotter error is therefore irrelevant
here: direct and reuse circuits realise the same Trotter circuit, and the
reference is what a perfect device running that circuit would produce.

Hamiltonians (first-order Trotter time evolution, terminal measurement of
all qubits; every term is an arbitrary Pauli string, so any spin
interaction is expressible):
    TFIM-1D      transverse-field Ising chain            (local, shallow)
    Ising-mix    mixed longitudinal+transverse Ising     (non-integrable)
    XXZ          anisotropic Heisenberg chain + Z field  (XX+YY+ZZ)
    XY           XY chain + transverse field             (XX+YY)
    TFIM-2D      transverse-field Ising on a 3x4 grid    (2D brickwork)
    SK-all2all   Sherrington-Kirkpatrick random Ising, all-to-all ZZ
    SYK4         Sachdev-Ye-Kitaev q=4, 8 Majoranas -> 4 qubits via
                 Jordan-Wigner, all C(8,4)=70 quartic terms

Expected compression physics: qubit reuse exploits LOCAL causal cones, so
the chains and the 2D grid compress; SK-all2all and SYK4 are all-to-all /
maximally scrambling and are EXPECTED to show zero compression -- included
deliberately as honest negative controls, not hidden.

Noise model (conventions of the companion QRC-Climate project):
    depolarizing p1 = 3e-4 on sx/x (rz is virtual, noise-free)
    depolarizing p2 = 7e-3 on cx
    symmetric readout flip ro = 1.5e-2 on every measurement
      (mid-circuit measurements in reuse circuits pay it too)
    reset bit-flip p_reset = 5e-3 on every reset
      (a reuse-specific error channel; direct circuits have no resets)
Circuits are transpiled to the rz/sx/x/cx basis at optimization_level=0
for the noisy runs, identically for direct and reuse.

HONESTY NOTE on what this noise model can and cannot test: gate counts are
IDENTICAL between a circuit and its reuse compilation, so a gate-local
noise model probes "does mid-circuit measure/reset machinery degrade
results" (plus the reset-error penalty) -- it does NOT capture the
idle/memory error that grows with the reuse circuit's larger depth. That
requires duration-aware scheduling + thermal relaxation and is listed as
the top improvement in REFLECTION.md. No hardware claim is made; all
results are Aer simulation.

Anchors run first: the Pauli-rotation emitter is verified against Qiskit's
PauliEvolutionGate up to global phase, and every SYK term is checked
Hermitian with real coefficient.

Usage:  python3 benchmark_hamiltonians.py
Writes: benchmark_results.json, benchmark_hamiltonians.png (3-panel:
        compression / ideal TVD / noisy marginal error, independent
        y-scales per panel), tvd_ideal_vs_noisy.png (all four TVD-to-exact
        numbers -- ideal/noisy x direct/reuse -- on ONE shared y-axis, so
        the noise contribution and the reuse contribution are directly
        comparable in size).
Exit 0 iff all checks pass.
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    pauli_error,
)

from qreuse_ir import from_qiskit
from qreuse_analysis import compute_output_cones
from qreuse_scheduler import (
    compiled_qubit_count,
    greedy_with_first_qubit_search,
)
from qreuse_compiler import compile_with_reuse
from qreuse_validation import (
    circuit_stats,
    exact_clbit_distribution,
    exact_clbit_distribution_sv,
    hellinger_fidelity,
    normalize_counts,
    run_counts,
    total_variation_distance,
    validate_reuse_circuit,
)
from qrc_experiment import counts_to_features  # THE endianness reversal point

SEED = 7
SHOTS = 2048
NOISE_P1 = 3e-4
NOISE_P2 = 7e-3
NOISE_RO = 1.5e-2
NOISE_RESET = 5e-3

#: term = (coefficient, {qubit_index: 'X'|'Y'|'Z'})
PauliTerm = Tuple[float, Dict[int, str]]

_failures: List[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        _failures.append(label)


# ------------------------------------------------- Pauli-string evolution
def append_pauli_rotation(
    qc: QuantumCircuit, pmap: Dict[int, str], theta: float
) -> None:
    """Append exp(-i theta/2 * P) for the Pauli string `pmap`.

    Standard construction with whitelisted gates only: basis change
    (X: H; Y: Sdg,H), CX chain onto the last support qubit, RZ(theta),
    then undo.  Empty pmap = global phase: skipped.
    """
    support = sorted(pmap)
    if not support:
        return
    for q in support:  # W^dagger
        if pmap[q] == "X":
            qc.h(q)
        elif pmap[q] == "Y":
            qc.sdg(q)
            qc.h(q)
    for a, b in zip(support[:-1], support[1:]):
        qc.cx(a, b)
    qc.rz(theta, support[-1])
    for a, b in zip(reversed(support[:-1]), reversed(support[1:])):
        qc.cx(a, b)
    for q in support:  # W
        if pmap[q] == "X":
            qc.h(q)
        elif pmap[q] == "Y":
            qc.h(q)
            qc.s(q)


def trotter_circuit(
    n: int,
    terms: Sequence[PauliTerm],
    steps: int,
    dt: float,
    measure: bool = True,
    init: str = "zero",
) -> QuantumCircuit:
    """First-order Trotter circuit exp(-i H t) ~ prod_k prod_j exp(-i c_j dt P_j),
    t = steps*dt, terms applied in the given (cone-aware) order each step.

    init: "zero" -> |0...0>;  "neel" -> |0101...> (X on odd sites).
    The Neel quench matters for U(1)-symmetric models (XXZ, XY): |0...0> is
    an EXACT eigenstate of XX+YY (the hopping term annihilates it), so the
    ideal output distribution would be a delta and the equivalence test
    trivial.  Initial 1-qubit gates never expand causal cones, so the
    choice does not affect compression.
    """
    qc = QuantumCircuit(n, n if measure else 0)
    if init == "neel":
        for q in range(1, n, 2):
            qc.x(q)
    elif init != "zero":
        raise ValueError(f"unknown init '{init}'")
    for _ in range(steps):
        for coeff, pmap in terms:
            append_pauli_rotation(qc, pmap, 2.0 * coeff * dt)
    if measure:
        qc.measure(range(n), range(n))
    return qc


# ------------------------------------------------------------ Hamiltonians
def _chain_zz_edges(n: int) -> List[Tuple[int, int]]:
    """Open-chain edges in brickwork (even, then odd) order -- the ordering
    of commuting terms shapes the causal cones (paper Sec. VII)."""
    return [(i, i + 1) for i in range(0, n - 1, 2)] + \
           [(i, i + 1) for i in range(1, n - 1, 2)]


def spec_tfim_chain(n: int = 10, k: int = 2) -> Dict:
    J, h = 1.0, 1.0
    terms: List[PauliTerm] = [(-J, {i: "Z", j: "Z"})
                              for i, j in _chain_zz_edges(n)]
    terms += [(-h, {q: "X"}) for q in range(n)]
    return {"name": "TFIM-1D", "n": n, "terms": terms, "steps": k,
            "dt": 0.2, "expect_compression": True}


def spec_mixed_ising_chain(n: int = 10, k: int = 2) -> Dict:
    J, hx, hz = 1.0, 0.9, 0.4
    terms: List[PauliTerm] = [(-J, {i: "Z", j: "Z"})
                              for i, j in _chain_zz_edges(n)]
    terms += [(-hx, {q: "X"}) for q in range(n)]
    terms += [(-hz, {q: "Z"}) for q in range(n)]
    return {"name": "Ising-mix", "n": n, "terms": terms, "steps": k,
            "dt": 0.2, "expect_compression": True}


def spec_xxz_chain(n: int = 10, k: int = 2) -> Dict:
    Jx, Jy, Jz, hz = 1.0, 1.0, 0.5, 0.4
    terms: List[PauliTerm] = []
    for i, j in _chain_zz_edges(n):  # grouped per edge, brickwork order
        terms += [(Jx, {i: "X", j: "X"}),
                  (Jy, {i: "Y", j: "Y"}),
                  (Jz, {i: "Z", j: "Z"})]
    terms += [(-hz, {q: "Z"}) for q in range(n)]
    return {"name": "XXZ", "n": n, "terms": terms, "steps": k,
            "dt": 0.15, "expect_compression": True, "init": "neel"}


def spec_xy_chain(n: int = 10, k: int = 2) -> Dict:
    Jx, Jy, hz = 1.0, 1.0, 0.6
    terms: List[PauliTerm] = []
    for i, j in _chain_zz_edges(n):
        terms += [(Jx, {i: "X", j: "X"}), (Jy, {i: "Y", j: "Y"})]
    terms += [(-hz, {q: "Z"}) for q in range(n)]
    return {"name": "XY", "n": n, "terms": terms, "steps": k,
            "dt": 0.15, "expect_compression": True, "init": "neel"}


def spec_tfim_2d(rows: int = 3, cols: int = 4, k: int = 1) -> Dict:
    J, h = 1.0, 1.0
    idx = lambda r, c: r * cols + c  # noqa: E731
    horiz = [(idx(r, c), idx(r, c + 1))
             for c in range(0, cols - 1, 2) for r in range(rows)]
    horiz += [(idx(r, c), idx(r, c + 1))
              for c in range(1, cols - 1, 2) for r in range(rows)]
    vert = [(idx(r, c), idx(r + 1, c))
            for r in range(0, rows - 1, 2) for c in range(cols)]
    vert += [(idx(r, c), idx(r + 1, c))
             for r in range(1, rows - 1, 2) for c in range(cols)]
    terms: List[PauliTerm] = [(-J, {i: "Z", j: "Z"}) for i, j in horiz + vert]
    terms += [(-h, {q: "X"}) for q in range(rows * cols)]
    return {"name": "TFIM-2D", "n": rows * cols, "terms": terms, "steps": k,
            "dt": 0.2, "expect_compression": True}


def spec_sk_all_to_all(n: int = 8, k: int = 1, seed: int = SEED) -> Dict:
    rng = np.random.default_rng(seed + 11)
    terms: List[PauliTerm] = [
        (float(rng.normal(0.0, 1.0 / np.sqrt(n))), {i: "Z", j: "Z"})
        for i, j in combinations(range(n), 2)
    ]
    terms += [(-0.8, {q: "X"}) for q in range(n)]
    return {"name": "SK-all2all", "n": n, "terms": terms, "steps": k,
            "dt": 0.25, "expect_compression": False}


def _majorana(idx: int, nq: int) -> Pauli:
    """Jordan-Wigner Majorana: gamma_2j = Z..Z X_j, gamma_2j+1 = Z..Z Y_j."""
    j = idx // 2
    z = np.zeros(nq, dtype=bool)
    x = np.zeros(nq, dtype=bool)
    z[:j] = True
    x[j] = True
    if idx % 2 == 1:
        z[j] = True  # (z=1, x=1) site is Pauli Y
    return Pauli((z, x))


def syk4_terms(n_majorana: int = 8, seed: int = SEED) -> List[PauliTerm]:
    """H = sum_{a<b<c<d} J_abcd g_a g_b g_c g_d, J ~ N(0, 3! J^2 / N^3), J=1.

    Every quartic Majorana product is Hermitian (Pauli group phase 0 or 2),
    asserted below; the JW-mapped Pauli string and its real sign are
    extracted from Qiskit's phase bookkeeping.
    """
    nq = n_majorana // 2
    rng = np.random.default_rng(seed + 23)
    sigma = np.sqrt(6.0) / n_majorana ** 1.5
    terms: List[PauliTerm] = []
    for a, b, c, d in combinations(range(n_majorana), 4):
        p = _majorana(a, nq).dot(_majorana(b, nq)) \
            .dot(_majorana(c, nq)).dot(_majorana(d, nq))
        if p.phase not in (0, 2):  # (-i)^phase must be +/-1 (Hermitian)
            raise AssertionError(f"non-Hermitian SYK term phase {p.phase}")
        sign = 1.0 if p.phase == 0 else -1.0
        pmap = {}
        for q in range(nq):
            zq, xq = bool(p.z[q]), bool(p.x[q])
            if zq and xq:
                pmap[q] = "Y"
            elif zq:
                pmap[q] = "Z"
            elif xq:
                pmap[q] = "X"
        coeff = float(rng.normal(0.0, sigma)) * sign
        if pmap:
            terms.append((coeff, pmap))
    return terms


def spec_syk4(n_majorana: int = 8, k: int = 1) -> Dict:
    return {"name": "SYK4", "n": n_majorana // 2,
            "terms": syk4_terms(n_majorana), "steps": k, "dt": 1.0,
            "expect_compression": False}


ALL_SPECS = [
    spec_tfim_chain, spec_mixed_ising_chain, spec_xxz_chain, spec_xy_chain,
    spec_tfim_2d, spec_sk_all_to_all, spec_syk4,
]


# ----------------------------------------------------------------- anchors
def anchor_pauli_rotation() -> None:
    """My emitter must equal Qiskit's PauliEvolutionGate up to global phase."""
    from qiskit.circuit.library import PauliEvolutionGate

    print("\n== Anchor A: Pauli-rotation emitter vs PauliEvolutionGate ==")
    cases = ["Z", "X", "Y", "XY", "ZZ", "YXZ", "XIYZ"]
    ok, worst = True, 0.0
    for label in cases:
        nq = len(label)
        pmap = {i: ch for i, ch in enumerate(label[::-1]) if ch != "I"}
        theta = 0.7351
        mine = QuantumCircuit(nq)
        append_pauli_rotation(mine, pmap, theta)
        ref = QuantumCircuit(nq)
        ref.append(PauliEvolutionGate(Pauli(label), time=theta / 2.0),
                   range(nq))
        M = (Operator(ref).adjoint() @ Operator(mine)).data
        phase = M[0, 0]
        dev = float(np.max(np.abs(M - phase * np.eye(M.shape[0]))))
        worst = max(worst, dev)
        ok &= abs(abs(phase) - 1.0) < 1e-9 and dev < 1e-9
    check(ok, f"exp(-i theta/2 P) exact for {cases} "
              f"(worst deviation {worst:.2e})")


def anchor_syk_hermitian() -> None:
    print("\n== Anchor B: SYK4 Jordan-Wigner terms ==")
    terms = syk4_terms(8)
    check(len(terms) == 70, f"C(8,4) = 70 quartic terms (got {len(terms)})")
    check(all(np.isreal(c) for c, _ in terms),
          "all JW coefficients real (Hermitian terms)")
    weights = sorted({len(p) for _, p in terms})
    print(f"  Pauli-string weights present: {weights}")


# --------------------------------------------------------------- benchmark
def make_noise_model() -> NoiseModel:
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(NOISE_P1, 1),
                                   ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(NOISE_P2, 2), ["cx"])
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - NOISE_RO, NOISE_RO], [NOISE_RO, 1 - NOISE_RO]])
    )
    nm.add_all_qubit_quantum_error(
        pauli_error([("X", NOISE_RESET), ("I", 1 - NOISE_RESET)]), ["reset"]
    )
    return nm


_NOISY_SIM = AerSimulator(noise_model=make_noise_model())


def run_counts_noisy(qc: QuantumCircuit, shots: int, seed: int) -> Dict:
    tqc = transpile(qc, _NOISY_SIM, optimization_level=0)
    res = _NOISY_SIM.run(tqc, shots=shots, seed_simulator=seed).result()
    return res.get_counts()


def exact_reference(qc_no_meas: QuantumCircuit) -> Tuple[Dict, np.ndarray]:
    """Exact ideal distribution and per-qubit <Z_i> from the statevector."""
    probs = Statevector(qc_no_meas).probabilities_dict()
    n = qc_no_meas.num_qubits
    z = np.zeros(n)
    for key, p in probs.items():
        bits = key[::-1]  # same one-reversal convention as counts
        for i in range(n):
            z[i] += p * (1.0 - 2.0 * int(bits[i]))
    return probs, z


def marginal_mae(counts: Dict, z_exact: np.ndarray, n: int) -> float:
    """Mean per-qubit marginal TVD |P1_hat - P1_exact| = |z_hat - z|/2."""
    z_hat = counts_to_features(counts, n)[:n]
    return float(np.mean(np.abs(z_hat - z_exact)) / 2.0)


def run_one(spec: Dict, seed: int = SEED) -> Dict:
    name, n = spec["name"], spec["n"]
    print(f"\n== {name}: n={n}, {len(spec['terms'])} terms, "
          f"steps={spec['steps']}, dt={spec['dt']} ==")

    t0 = time.time()
    init = spec.get("init", "zero")
    qc = trotter_circuit(n, spec["terms"], spec["steps"], spec["dt"],
                         init=init)
    qc_ref = trotter_circuit(n, spec["terms"], spec["steps"], spec["dt"],
                             measure=False, init=init)
    ir = from_qiskit(qc)
    cones = compute_output_cones(ir)
    order = greedy_with_first_qubit_search(cones)
    compiled = compile_with_reuse(ir, order=order)
    t_compile = time.time() - t0
    issues = validate_reuse_circuit(compiled, ir, raise_on_error=False)
    check(not issues, f"{name}: reuse-circuit legality"
          + ("" if not issues else f" ({issues})"))

    sd = circuit_stats(qc)
    sr = circuit_stats(compiled.circuit)
    p_exact, z_exact = exact_reference(qc_ref)

    # zlib.crc32 is a stable, process-independent hash.  Python's builtin
    # hash() on str is randomized per process (PYTHONHASHSEED), which made
    # earlier runs of this benchmark NON-reproducible despite a fixed seed.
    base = seed + zlib.crc32(name.encode()) % 100000
    runs: Dict[str, Dict] = {}
    for mode, runner in (("ideal", run_counts), ("noisy", run_counts_noisy)):
        for variant, circ in (("direct", qc), ("reuse", compiled.circuit)):
            t1 = time.time()
            counts = runner(circ, shots=SHOTS,
                            seed=base + (0 if mode == "ideal" else 5)
                            + (0 if variant == "direct" else 1))
            dist = normalize_counts(counts)
            runs[f"{mode}_{variant}"] = {
                "tvd_to_exact": total_variation_distance(dist, p_exact),
                "hellinger_to_exact": hellinger_fidelity(dist, p_exact),
                "marginal_mae": marginal_mae(counts, z_exact, n),
                "sim_seconds": time.time() - t1,
                "_dist": dist,
            }

    tvd_dir_reuse_ideal = total_variation_distance(
        runs["ideal_direct"]["_dist"], runs["ideal_reuse"]["_dist"])
    tvd_dir_reuse_noisy = total_variation_distance(
        runs["noisy_direct"]["_dist"], runs["noisy_reuse"]["_dist"])
    for r in runs.values():
        del r["_dist"]

    print(f"  qubits {n} -> {compiled.physical_qubit_count}   "
          f"depth {sd['depth']} -> {sr['depth']}   "
          f"2q(cx) {sum(1 for i in qc.data if i.operation.name == 'cx')}   "
          f"resets {sr['resets']}   "
          f"mid-meas {sr['mid_circuit_measurements']}   "
          f"compile {t_compile:.2f}s")
    for key in ("ideal_direct", "ideal_reuse", "noisy_direct", "noisy_reuse"):
        r = runs[key]
        print(f"  {key:13s} TVD->exact {r['tvd_to_exact']:.4f}   "
              f"Hellinger {r['hellinger_to_exact']:.4f}   "
              f"marginal {r['marginal_mae']:.4f}   "
              f"({r['sim_seconds']:.2f}s)")
    print(f"  TVD(direct,reuse): ideal {tvd_dir_reuse_ideal:.4f}   "
          f"noisy {tvd_dir_reuse_noisy:.4f}")

    # -- exact, shot-noise-free equivalence -------------------------------
    # Statevector-per-branch simulator (cost ~ #branches x 2**n) reaches
    # larger Hamiltonians than the density-matrix version; guarded by a
    # branch cap that raises rather than exhausting memory.
    exact_tvd = None
    if n <= 12 and compiled.circuit.num_qubits <= 12:
        try:
            d_exact = exact_clbit_distribution_sv(qc)
            r_exact = exact_clbit_distribution_sv(compiled.circuit)
            exact_tvd = total_variation_distance(d_exact, r_exact)
            print(f"  EXACT TVD(direct, reuse) = {exact_tvd:.2e}  "
                  "(no shot noise)")
            check(exact_tvd < 1e-10,
                  f"{name}: exact distributions identical to 1e-10")
        except ValueError as exc:  # branch count over ceiling
            print(f"  exact check skipped: {exc}")

    # -- sampled checks ---------------------------------------------------
    ideal_floor = runs["ideal_direct"]["tvd_to_exact"]
    check(runs["ideal_reuse"]["tvd_to_exact"] <= 1.4 * ideal_floor + 0.02,
          f"{name}: ideal reuse TVD at the direct sampling floor")
    check(runs["ideal_reuse"]["marginal_mae"]
          <= 1.4 * runs["ideal_direct"]["marginal_mae"] + 0.01,
          f"{name}: ideal reuse marginals at the direct floor")
    dh = abs(runs["noisy_direct"]["hellinger_to_exact"]
             - runs["noisy_reuse"]["hellinger_to_exact"])
    check(dh <= 0.08,
          f"{name}: noisy direct/reuse Hellinger within 0.08 (d={dh:.4f})")
    if spec["expect_compression"]:
        check(compiled.physical_qubit_count < n,
              f"{name}: compresses (physical < logical)")
    else:
        check(compiled.physical_qubit_count == n,
              f"{name}: honest zero compression (all-to-all cones)")

    return {
        "name": name,
        "n_logical": n,
        "n_physical": compiled.physical_qubit_count,
        "n_terms": len(spec["terms"]),
        "trotter_steps": spec["steps"],
        "depth_direct": sd["depth"],
        "depth_reuse": sr["depth"],
        "cx_count": sum(1 for i in qc.data if i.operation.name == "cx"),
        "resets": sr["resets"],
        "mid_circuit_measurements": sr["mid_circuit_measurements"],
        "reuse_measurements": sr["reuse_measurements"],
        "dropped_operations": compiled.dropped_operations,
        "compile_seconds": t_compile,
        "tvd_direct_vs_reuse_ideal": tvd_dir_reuse_ideal,
        "tvd_direct_vs_reuse_noisy": tvd_dir_reuse_noisy,
        "exact_tvd_direct_vs_reuse": exact_tvd,
        "runs": runs,
    }


# ------------------------------------------------------------------ report
def make_benchmark_figure(rows: List[Dict], path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in rows]
    xs = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    ax.bar(xs - 0.2, [r["n_logical"] for r in rows], 0.4,
           label="logical", color="0.6")
    ax.bar(xs + 0.2, [r["n_physical"] for r in rows], 0.4,
           label="reuse-compiled", color="tab:blue")
    ax.set_ylabel("qubits")
    ax.set_title("compression by Hamiltonian family")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(xs - 0.2, [r["runs"]["ideal_direct"]["tvd_to_exact"]
                      for r in rows], 0.4, label="direct", color="0.6")
    ax.bar(xs + 0.2, [r["runs"]["ideal_reuse"]["tvd_to_exact"]
                      for r in rows], 0.4, label="reuse", color="tab:blue")
    ax.set_ylabel("TVD to exact distribution")
    ax.set_title(f"ideal sampling, S={SHOTS} (equal bars = equivalent)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(xs - 0.2, [r["runs"]["noisy_direct"]["marginal_mae"]
                      for r in rows], 0.4, label="direct", color="0.6")
    ax.bar(xs + 0.2, [r["runs"]["noisy_reuse"]["marginal_mae"]
                      for r in rows], 0.4, label="reuse", color="tab:orange")
    ax.set_ylabel("per-qubit marginal error vs exact")
    ax.set_title("noisy simulation (gate + readout + reset noise)")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    fig.suptitle(
        "qubit-reuse benchmark: direct vs reuse-compiled, ideal vs noisy "
        "(Aer simulation; gate-local noise -- see honesty note)",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_tvd_scale_figure(rows: List[Dict], path: str) -> str:
    """All four TVD-to-exact numbers (ideal/noisy x direct/reuse) on ONE
    shared y-axis, grouped by Hamiltonian.

    Purpose: the three-panel figure above puts ideal and noisy TVD on
    separate subplots with independent y-scales, which hides how much
    bigger the noise contribution is than the reuse contribution. Putting
    all four bars on one axis makes that comparison direct: within a
    color pair (ideal or noisy), direct vs reuse should be close --
    that's the equivalence claim. Across colors, noisy >> ideal is the
    dominant effect, and it scales with entangling (CX) gate count more
    than with which Hamiltonian family it is (compare XXZ/SYK4, both CX
    heavy, to TFIM-1D, CX-light).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in rows]
    xs = np.arange(len(rows))
    w = 0.2

    def tvd(r: Dict, mode: str, variant: str) -> float:
        return r["runs"][f"{mode}_{variant}"]["tvd_to_exact"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(xs - 1.5 * w, [tvd(r, "ideal", "direct") for r in rows], w,
           label="ideal, direct", color="#9ecae1")
    ax.bar(xs - 0.5 * w, [tvd(r, "ideal", "reuse") for r in rows], w,
           label="ideal, reuse", color="#3182bd")
    ax.bar(xs + 0.5 * w, [tvd(r, "noisy", "direct") for r in rows], w,
           label="noisy, direct", color="#fdae6b")
    ax.bar(xs + 1.5 * w, [tvd(r, "noisy", "reuse") for r in rows], w,
           label="noisy, reuse", color="#e6550d")

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("TVD to exact distribution")
    ax.set_title(
        "TVD to exact distribution: ideal vs noisy, direct vs reuse\n"
        f"(same scale; S={SHOTS} shots, seed={SEED})"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    t0 = time.time()
    print(f"benchmark config: seed={SEED} shots={SHOTS} "
          f"noise p1={NOISE_P1} p2={NOISE_P2} ro={NOISE_RO} "
          f"reset={NOISE_RESET}")

    anchor_pauli_rotation()
    anchor_syk_hermitian()

    rows = [run_one(spec()) for spec in ALL_SPECS]
    n_runs = 4 * len(rows)

    print("\n================ consolidated table ================")
    hdr = (f"{'Hamiltonian':<11} {'qubits':>9} {'depth':>11} {'cx':>5} "
           f"{'rst':>4} | {'TVD->exact ideal d/r':>21} | "
           f"{'marginal noisy d/r':>19}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<11} {r['n_logical']:>3} ->{r['n_physical']:>3} "
              f"{r['depth_direct']:>4} ->{r['depth_reuse']:>5} "
              f"{r['cx_count']:>5} {r['resets']:>4} | "
              f"{r['runs']['ideal_direct']['tvd_to_exact']:>10.4f} "
              f"{r['runs']['ideal_reuse']['tvd_to_exact']:>10.4f} | "
              f"{r['runs']['noisy_direct']['marginal_mae']:>9.4f} "
              f"{r['runs']['noisy_reuse']['marginal_mae']:>9.4f}")

    with open("benchmark_results.json", "w") as fh:
        json.dump({"config": {"seed": SEED, "shots": SHOTS,
                              "noise": {"p1": NOISE_P1, "p2": NOISE_P2,
                                        "ro": NOISE_RO,
                                        "reset": NOISE_RESET}},
                   "hamiltonians": rows}, fh, indent=2)
    fig = make_benchmark_figure(rows, "benchmark_hamiltonians.png")
    fig2 = make_tvd_scale_figure(rows, "tvd_ideal_vs_noisy.png")
    print(f"\nwrote benchmark_results.json and {fig} and {fig2}")
    print(f"{n_runs} sampled runs ({len(rows)} Hamiltonians x 4) in "
          f"{time.time() - t0:.1f} s")
    print(f"\n{'ALL CHECKS PASSED' if not _failures else 'FAILURES:'}")
    for f in _failures:
        print(f"  - {f}")
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
