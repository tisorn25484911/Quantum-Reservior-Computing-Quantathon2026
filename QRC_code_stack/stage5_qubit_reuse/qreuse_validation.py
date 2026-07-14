"""qreuse_validation.py -- legality checks and statistical comparison utilities.

Two jobs:

  1. Structural validation of a ReuseCompiledCircuit (reset/measurement
     legality, classical-bit bookkeeping, qubit-count sanity).
  2. Statistical comparison of measurement-count dictionaries between the
     direct and the reuse-compiled circuit: total variation distance,
     Hellinger fidelity, KL divergence.

Because both circuits write each logical output into the SAME classical bit
index, their Qiskit count dictionaries are directly comparable key-by-key
(no endianness handling needed here; keys are compared as opaque strings).

`run_counts` is the single simulation entry point (Qiskit Aer).  Circuits
are transpiled at optimization_level=0 only to map onto Aer's basis; the
reuse circuits contain mid-circuit measurements and resets, which Aer
executes natively.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

from qreuse_ir import LogicalCircuitIR, ReuseCompiledCircuit

_SIM = AerSimulator()


# ----------------------------------------------------------- distributions
def normalize_counts(counts: Dict[str, int]) -> Dict[str, float]:
    """Counts -> probability distribution (raises on empty/zero counts)."""
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("cannot normalize empty or zero-total counts")
    return {k: v / total for k, v in counts.items()}


def total_variation_distance(
    p: Dict[str, float], q: Dict[str, float]
) -> float:
    """TVD(p, q) = 0.5 * sum_k |p_k - q_k| over the union of outcomes.

    Keys are SORTED before summing.  Iterating a Python set of strings
    uses a per-process randomized hash, so an unsorted sum makes the
    last floating-point digits process-dependent -- enough to break
    bit-for-bit reproducibility of reported metrics.
    """
    keys = sorted(set(p) | set(q))
    return 0.5 * math.fsum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def hellinger_fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
    """(sum_k sqrt(p_k q_k))^2 -- 1.0 for identical distributions."""
    keys = sorted(set(p) | set(q))
    bc = math.fsum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys)
    return bc ** 2


def kl_divergence(
    p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12
) -> float:
    """D_KL(p || q), epsilon-smoothed for finite-shot zero counts."""
    keys = sorted(set(p) | set(q))
    terms = [
        p[k] * math.log((p[k] + eps) / (q.get(k, 0.0) + eps))
        for k in keys if p.get(k, 0.0) > 0.0
    ]
    return math.fsum(terms)


def compare_counts(
    counts_direct: Dict[str, int], counts_reuse: Dict[str, int]
) -> Dict[str, float]:
    """All comparison metrics between two raw count dictionaries."""
    p = normalize_counts(counts_direct)
    q = normalize_counts(counts_reuse)
    return {
        "tvd": total_variation_distance(p, q),
        "hellinger_fidelity": hellinger_fidelity(p, q),
        "kl_direct_vs_reuse": kl_divergence(p, q),
        "shots_direct": float(sum(counts_direct.values())),
        "shots_reuse": float(sum(counts_reuse.values())),
    }


# ------------------------------------------------------------- simulation
def run_counts(
    qc: QuantumCircuit, shots: int = 4096, seed: Optional[int] = None
) -> Dict[str, int]:
    """Run on Aer, return counts.  The single simulation entry point."""
    tqc = transpile(qc, _SIM, optimization_level=0)
    result = _SIM.run(tqc, shots=shots, seed_simulator=seed).result()
    return result.get_counts()


# ------------------------------------------------- exact (shot-noise-free)
def _gate_matrix(inst, qc) -> np.ndarray:
    """Dense unitary of a single instruction on ITS OWN qubits only
    (2**k x 2**k for a k-qubit gate), in Qiskit's little-endian qubit order
    for that gate's argument list."""
    sub = QuantumCircuit(len(inst.qubits))
    sub.append(inst.operation, list(range(len(inst.qubits))))
    return Operator(sub).data


def _embed_unitary(qc: QuantumCircuit, inst, n: int) -> np.ndarray:
    """Matrix of one instruction embedded in the full n-qubit register.
    Used by the density-matrix simulator, which needs the full operator for
    U rho U^dagger; the statevector simulator uses _apply_local instead."""
    sub = QuantumCircuit(n)
    sub.append(inst.operation, [qc.find_bit(b).index for b in inst.qubits])
    return Operator(sub).data


def _apply_local(psi: np.ndarray, U: np.ndarray, targets, n: int) -> np.ndarray:
    """Apply a k-qubit gate matrix U to statevector psi (length 2**n) acting
    on `targets` (Qiskit little-endian: targets[0] is the gate's least
    significant qubit).  Uses tensor reshaping, never a 2**n x 2**n matrix.

    Qiskit's statevector index bit j corresponds to qubit j with bit 0 the
    least significant.  Reshaping to (2,)*n gives axis a = qubit (n-1-a) in
    numpy's row-major (big-endian axis) convention, so qubit q lives on axis
    (n-1-q).  U is indexed in little-endian over the target list, matching
    _gate_matrix.
    """
    k = len(targets)
    psi_t = psi.reshape((2,) * n)
    axes = [n - 1 - q for q in targets]           # numpy axes of the targets
    # move target axes to the front, in little-endian (LSB first) order so
    # they line up with U's column indexing
    psi_t = np.moveaxis(psi_t, axes, range(k))
    shape_rest = psi_t.shape[k:]
    # U columns are little-endian over targets; our leading axes are ordered
    # targets[0], targets[1], ... which is also little-endian -> reverse to
    # big-endian for the matmul, apply, reverse back.
    psi_mat = psi_t.reshape(2 ** k, -1)
    # our leading-axis order is targets[0]..targets[k-1] = LSB..; U is indexed
    # with basis state sum_i b_i 2**i over the same target order, i.e. the
    # row/col index has targets[0] as bit 0.  numpy's reshape treats the first
    # axis as most significant, so build the permutation from big-endian axis
    # order to little-endian U order.
    perm = [k - 1 - i for i in range(k)]
    if k > 1:
        psi_kt = psi_t.transpose(*perm, *range(k, psi_t.ndim))
        psi_mat = psi_kt.reshape(2 ** k, -1)
    out = (U @ psi_mat).reshape(*( (2,) * k ), *shape_rest)
    if k > 1:
        inv = np.argsort(perm)
        out = out.transpose(*inv, *range(k, out.ndim))
    out = np.moveaxis(out, range(k), axes)
    return out.reshape(-1)


def exact_clbit_distribution(
    qc: QuantumCircuit,
    max_branches: int = 8192,
    prune: float = 1e-14,
) -> Dict[str, float]:
    """Exact probability distribution over classical bit strings.

    Simulates the circuit as a density matrix and branches on every
    measurement, tracking the classical record.  Handles mid-circuit
    measurement and reset exactly, so it works on reuse-compiled circuits
    where `Statevector` cannot be used.  This removes shot noise entirely:
    a correct compiler must reproduce the ORIGINAL distribution to
    numerical precision, not merely within a sampling threshold.

    Channels applied:
        gate      rho -> U rho U^dagger
        measure   branch: (P0 rho P0, c=0) and (P1 rho P1, c=1)
        reset     rho -> P0 rho P0 + X P1 rho P1 X   (no classical record)

    Each branch's trace is its probability; branches below `prune` are
    discarded.  Intended for small circuits -- cost grows with the number
    of surviving branches times 4^num_qubits.  Raises if the branch count
    would exceed `max_branches`.

    Returned keys use Qiskit's convention (clbit 0 is the rightmost char).
    """
    n = qc.num_qubits
    nc = qc.num_clbits
    if nc == 0:
        raise ValueError("circuit has no classical bits to report")

    dim = 1 << n
    idx = np.arange(dim)
    rho0 = np.zeros((dim, dim), dtype=complex)
    rho0[0, 0] = 1.0
    # branch key: tuple of clbit values (-1 = not yet written)
    branches: Dict[Tuple[int, ...], np.ndarray] = {(-1,) * nc: rho0}

    def masks(q: int):
        m1 = ((idx >> q) & 1).astype(bool)
        return ~m1, m1

    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = [qc.find_bit(b).index for b in inst.qubits]

        if name == "measure":
            q = qargs[0]
            c = qc.find_bit(inst.clbits[0]).index
            m0, m1 = masks(q)
            new: Dict[Tuple[int, ...], np.ndarray] = {}
            for key, rho in branches.items():
                for val, m in ((0, m0), (1, m1)):
                    sub = np.zeros_like(rho)
                    sel = np.ix_(m, m)
                    sub[sel] = rho[sel]
                    p = float(np.real(np.trace(sub)))
                    if p <= prune:
                        continue
                    k = list(key)
                    k[c] = val
                    nk = tuple(k)
                    new[nk] = new[nk] + sub if nk in new else sub
            branches = new
            if len(branches) > max_branches:
                raise ValueError(
                    f"branch count {len(branches)} exceeds max_branches "
                    f"{max_branches}; circuit too large for exact simulation"
                )
            continue

        if name == "reset":
            q = qargs[0]
            m0, m1 = masks(q)
            flip = idx ^ (1 << q)
            for key, rho in branches.items():
                out = np.zeros_like(rho)
                sel0 = np.ix_(m0, m0)
                out[sel0] = rho[sel0]
                sub1 = np.zeros_like(rho)
                sel1 = np.ix_(m1, m1)
                sub1[sel1] = rho[sel1]
                out += sub1[np.ix_(flip, flip)]  # X . P1 rho P1 . X
                branches[key] = out
            continue

        U = _embed_unitary(qc, inst, n)
        Ud = U.conj().T
        for key, rho in branches.items():
            branches[key] = U @ rho @ Ud

    dist: Dict[str, float] = {}
    for key in sorted(branches):
        rho = branches[key]
        p = float(np.real(np.trace(rho)))
        if p <= prune:
            continue
        if any(v < 0 for v in key):
            raise ValueError("some classical bit was never written")
        bits = "".join(str(key[i]) for i in reversed(range(nc)))
        dist[bits] = dist.get(bits, 0.0) + p

    total = sum(dist.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"exact distribution does not sum to 1 ({total})")
    return dist


def exact_clbit_distribution_sv(
    qc: QuantumCircuit,
    max_branches: int = 20000,
    prune: float = 1e-14,
) -> Dict[str, float]:
    """Exact clbit distribution via STATEVECTOR-per-branch enumeration.

    Same semantics and same result as `exact_clbit_distribution`, but each
    incoherent branch is carried as a pure statevector (cost 2**n) rather
    than folding everything into one density matrix (cost 4**n).  This
    pushes the affordable ceiling out by a few qubits on circuits that stay
    close to pure (few resets, moderate measurement fan-out).

    Correctness subtleties this implementation is careful about:

      * Branches are keyed by a UNIQUE id, never by their classical record.
        Two branches that happen to share a classical record are, in
        general, an *incoherent* mixture; adding their statevectors would be
        physically wrong.  They are summed only at the very end, and only as
        probabilities.
      * A reset splits a branch into two incoherent pieces --- the population
        already in |0> on the target, and the population in |1> mapped to |0>
        --- and keeps them as SEPARATE branches (a density matrix could add
        them into one operator; two statevectors cannot).  This is the only
        place the branch count can grow without a measurement.
      * Each branch stores an unnormalised statevector whose squared norm is
        that branch's probability, exactly mirroring the trace bookkeeping of
        the density-matrix version.

    Raises ValueError if the branch count exceeds `max_branches`.
    """
    n = qc.num_qubits
    nc = qc.num_clbits
    if nc == 0:
        raise ValueError("circuit has no classical bits to report")

    dim = 1 << n
    idx = np.arange(dim)
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0

    # each branch: (unique_id) -> (classical_record tuple, statevector)
    branches: Dict[int, Tuple[Tuple[int, ...], np.ndarray]] = {
        0: ((-1,) * nc, psi0)
    }
    next_id = 1

    def mask1(q: int) -> np.ndarray:
        return ((idx >> q) & 1).astype(bool)

    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = [qc.find_bit(b).index for b in inst.qubits]

        if name == "measure":
            q = qargs[0]
            c = qc.find_bit(inst.clbits[0]).index
            m1 = mask1(q)
            new: Dict[int, Tuple[Tuple[int, ...], np.ndarray]] = {}
            for (rec, psi) in branches.values():
                for val in (0, 1):
                    sel = m1 if val == 1 else ~m1
                    sub = np.where(sel, psi, 0.0)
                    p = float(np.vdot(sub, sub).real)
                    if p <= prune:
                        continue
                    k = list(rec)
                    k[c] = val
                    new[next_id] = (tuple(k), sub)
                    next_id += 1
            branches = new
            if len(branches) > max_branches:
                raise ValueError(
                    f"branch count {len(branches)} exceeds max_branches "
                    f"{max_branches}; circuit too large for exact simulation"
                )
            continue

        if name == "reset":
            q = qargs[0]
            m1 = mask1(q)
            flip = idx ^ (1 << q)
            new = {}
            for (rec, psi) in branches.values():
                keep0 = np.where(m1, 0.0, psi)          # already |0> on q
                moved = np.where(m1, psi, 0.0)[flip]     # |1> -> |0> on q
                for piece in (keep0, moved):
                    if float(np.vdot(piece, piece).real) > prune:
                        new[next_id] = (rec, piece)
                        next_id += 1
            branches = new
            if len(branches) > max_branches:
                raise ValueError(
                    f"branch count {len(branches)} exceeds max_branches "
                    f"{max_branches}; circuit too large after reset"
                )
            continue

        U = _gate_matrix(inst, qc)
        targets = qargs  # Qiskit little-endian over the gate's qubit list
        branches = {bid: (rec, _apply_local(psi, U, targets, n))
                    for bid, (rec, psi) in branches.items()}

    dist: Dict[str, float] = {}
    for bid in sorted(branches):
        rec, psi = branches[bid]
        p = float(np.vdot(psi, psi).real)
        if p <= prune:
            continue
        if any(v < 0 for v in rec):
            raise ValueError("some classical bit was never written")
        bits = "".join(str(rec[i]) for i in reversed(range(nc)))
        dist[bits] = dist.get(bits, 0.0) + p

    total = sum(dist.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"exact distribution does not sum to 1 ({total})")
    return dist


# -------------------------------------------------------------- structure
def circuit_stats(qc: QuantumCircuit) -> Dict[str, int]:
    """Size/shape summary used in test and experiment report tables.

    Definitions (made explicit because "mid-circuit" is ambiguous):
      mid_circuit_measurements
          measurements followed, anywhere later in the circuit, by at
          least one non-measurement operation.  These are the ones that
          require dynamic-circuit support on hardware.
      reuse_measurements
          measurements followed later by an operation on the SAME qubit,
          i.e. the qubit is genuinely recycled.  Always <= the above.
    """
    data = [inst for inst in qc.data if inst.operation.name != "barrier"]
    names = [inst.operation.name for inst in data]
    n_measure = names.count("measure")
    n_reset = names.count("reset")

    last_nonmeas = -1
    for k, name in enumerate(names):
        if name != "measure":
            last_nonmeas = k

    mid_measures = 0
    reuse_measures = 0
    for k, inst in enumerate(data):
        if inst.operation.name != "measure":
            continue
        if k < last_nonmeas:
            mid_measures += 1
        q = qc.find_bit(inst.qubits[0]).index
        for later in data[k + 1:]:
            if later.operation.name == "measure":
                continue
            if q in [qc.find_bit(b).index for b in later.qubits]:
                reuse_measures += 1
                break

    return {
        "num_qubits": qc.num_qubits,
        "depth": qc.depth(),
        "gates": len(data) - n_measure - n_reset,
        "measures": n_measure,
        "resets": n_reset,
        "mid_circuit_measurements": mid_measures,
        "reuse_measurements": reuse_measures,
    }


def validate_reuse_circuit(
    compiled: ReuseCompiledCircuit,
    ir: LogicalCircuitIR,
    raise_on_error: bool = True,
) -> List[str]:
    """Structural legality checks on a compiled circuit.

    Verifies:
      - no physical qubit receives a gate after measurement unless reset
        occurs first;
      - no physical qubit is measured twice without an intervening reset
        (i.e. no logical output is measured after its final measurement,
        and no measurement result is overwritten);
      - every classical bit is written exactly once, and the mapping of
        logical outputs to classical bits matches the original circuit;
      - the number of measurements equals the number of measured outputs;
      - the compiled circuit uses no more physical qubits than the
        original uses logical qubits.

    Returns the list of issues (empty if clean); raises AssertionError on
    any issue when raise_on_error is True.
    """
    issues: List[str] = []
    qc = compiled.circuit

    if compiled.physical_qubit_count > ir.num_qubits:
        issues.append(
            f"compiled circuit uses {compiled.physical_qubit_count} physical "
            f"qubits, more than the {ir.num_qubits} logical qubits"
        )
    if qc.num_qubits != compiled.physical_qubit_count:
        issues.append(
            "circuit width disagrees with reported physical_qubit_count"
        )

    state = {i: "fresh" for i in range(qc.num_qubits)}
    clbit_writes: Dict[int, int] = {}
    n_measures = 0
    emitted_clbits: List[int] = []

    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(b).index for b in inst.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            q = qs[0]
            if state[q] == "measured":
                issues.append(
                    f"physical qubit {q} measured twice without reset"
                )
            state[q] = "measured"
            c = qc.find_bit(inst.clbits[0]).index
            clbit_writes[c] = clbit_writes.get(c, 0) + 1
            emitted_clbits.append(c)
            n_measures += 1
        elif name == "reset":
            state[qs[0]] = "fresh"
        else:
            for q in qs:
                if state[q] == "measured":
                    issues.append(
                        f"gate '{name}' on physical qubit {q} after "
                        "measurement without reset"
                    )
                state[q] = "active"

    # The k-th emitted measurement must write the clbit that the ORIGINAL
    # circuit used for the k-th scheduled output.  This checks the actual
    # emitted circuit, not merely the bookkeeping dict.
    expected = [ir.output_to_clbit[q] for q in compiled.measurement_order]
    if emitted_clbits != expected:
        issues.append(
            f"emitted measurement clbit sequence {emitted_clbits} does not "
            f"match the original clbits for the measurement order {expected}"
        )

    if n_measures != len(ir.measured_outputs):
        issues.append(
            f"{n_measures} measurements emitted, expected "
            f"{len(ir.measured_outputs)}"
        )
    if set(compiled.classical_output_map) != set(ir.measured_outputs):
        issues.append(
            "classical_output_map keys do not equal the measured outputs"
        )
    for q, c in compiled.classical_output_map.items():
        if ir.output_to_clbit.get(q) != c:
            issues.append(
                f"logical output {q} mapped to clbit {c}, original circuit "
                f"used clbit {ir.output_to_clbit.get(q)}"
            )
    for c, cnt in clbit_writes.items():
        if cnt != 1:
            issues.append(f"classical bit {c} written {cnt} times")

    if raise_on_error and issues:
        raise AssertionError("reuse-circuit validation failed: "
                             + "; ".join(issues))
    return issues
