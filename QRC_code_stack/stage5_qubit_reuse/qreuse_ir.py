"""qreuse_ir.py -- backend-independent intermediate representation (IR).

The qubit-reuse compiler works on a flat, time-ordered list of operations
acting on integer *logical* qubit indices, decoupled from Qiskit internals.
Qiskit appears only at the boundary: `from_qiskit` (parsing, here) and the
lowering in qreuse_compiler.py.

Semantics contract (important):
    Input circuits must contain TERMINAL measurements only -- every gate on
    a qubit precedes that qubit's (single) measurement.  This is what makes
    qubit reuse legal: with terminal measurements, every operation touching
    logical qubit q lies in q's backward causal cone, so once q's cone has
    been emitted and q measured, no later operation can reference q and its
    physical qubit can be reset and reused.

Reference: DeCross, Chertkov, Kohagen, Foss-Feig, "Qubit-Reuse Compilation
with Mid-Circuit Measurement and Reset", Phys. Rev. X 13, 041057 (2023).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from qiskit import QuantumCircuit

#: Unitary gates the parser understands.  Every name maps 1:1 onto a
#: QuantumCircuit method of the same name with signature (*params, *qubits),
#: which is what the lowering in qreuse_compiler.py relies on.
SUPPORTED_GATES = {
    # one-qubit
    "id", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg",
    "rx", "ry", "rz", "p", "u",
    # two-qubit
    "cx", "cy", "cz", "ch", "cp", "crx", "cry", "crz", "swap",
    "rxx", "ryy", "rzz", "rzx",
}

#: Instructions silently dropped during parsing (no unitary action).
IGNORED_OPS = {"barrier", "delay"}


@dataclass(frozen=True)
class Op:
    """One circuit operation on logical qubits.

    `order` is the position in the time-ordered operation list and doubles
    as a unique id; frozen + tuple fields make Op hashable.
    """

    name: str
    targets: Tuple[int, ...]
    controls: Tuple[int, ...] = ()
    params: Tuple[float, ...] = ()
    order: int = -1

    @property
    def qubits(self) -> Tuple[int, ...]:
        """All logical qubits touched, controls first (Qiskit call order)."""
        return tuple(self.controls) + tuple(self.targets)


@dataclass
class LogicalCircuitIR:
    """A time-ordered logical circuit with terminal measurements."""

    num_qubits: int
    operations: List[Op]
    measured_outputs: List[int]
    #: logical output qubit -> classical bit index in the original circuit.
    #: Preserved by the compiler so count dictionaries from the direct and
    #: the reuse-compiled circuit are directly comparable key-by-key.
    output_to_clbit: Dict[int, int] = field(default_factory=dict)
    num_clbits: int = 0


@dataclass
class ReuseCompiledCircuit:
    """Result of qubit-reuse compilation (see qreuse_compiler.py)."""

    circuit: QuantumCircuit
    physical_qubit_count: int
    #: one entry per scheduled measurement: which logical output was
    #: measured, on which physical qubit, and the live logical->physical
    #: map at that moment.
    logical_to_physical_history: List[Dict]
    #: logical output qubit -> classical bit (same clbit as the original).
    classical_output_map: Dict[int, int]
    measurement_order: List[int]
    cones: Dict[int, Set[int]]
    #: operations of the original circuit that lie outside EVERY measured
    #: output's causal cone and were therefore dropped.  Sound: any op
    #: touching a measured qubit q is inside cone(q) (terminal measurement),
    #: so dropped ops act only on never-measured qubits; and by downward
    #: closure no dropped op precedes a retained op on a shared qubit.  A
    #: unitary on a disjoint, traced-out subsystem cannot change the joint
    #: distribution of the measured qubits.
    dropped_operations: int = 0


def from_qiskit(qc: QuantumCircuit) -> LogicalCircuitIR:
    """Extract a LogicalCircuitIR from a Qiskit QuantumCircuit.

    Supported: the gates in SUPPORTED_GATES, plus `measure` and `reset`.
    Barriers and delays are dropped.  Raises NotImplementedError for
    anything else (including classically-controlled operations and unbound
    symbolic parameters), and for circuits whose measurements are not
    terminal (a gate or reset acting on an already-measured qubit) or that
    measure a qubit more than once.
    """
    operations: List[Op] = []
    measured: List[int] = []
    output_to_clbit: Dict[int, int] = {}
    measured_set: Set[int] = set()
    used_clbits: Set[int] = set()
    order = 0

    for inst in qc.data:
        name = inst.operation.name
        if name in IGNORED_OPS:
            continue
        qidx = [qc.find_bit(qb).index for qb in inst.qubits]

        if name == "measure":
            q = qidx[0]
            if q in measured_set:
                raise NotImplementedError(
                    f"qubit {q} is measured more than once; only a single "
                    "terminal measurement per qubit is supported"
                )
            clbit = qc.find_bit(inst.clbits[0]).index
            if clbit in used_clbits:
                raise NotImplementedError(
                    f"classical bit {clbit} is written by more than one "
                    "measurement; each output needs its own classical bit "
                    "for the compiled counts to be comparable"
                )
            used_clbits.add(clbit)
            measured_set.add(q)
            measured.append(q)
            output_to_clbit[q] = clbit
            operations.append(Op("measure", (q,), (), (), order))
            order += 1
            continue

        # Any non-measure instruction touching classical bits implies
        # classical control / feed-forward, which the causal-cone analysis
        # does not model (a classical wire is an extra dependency edge).
        if len(inst.clbits) > 0 or getattr(inst.operation, "condition", None):
            raise NotImplementedError(
                f"instruction '{name}' uses classical bits or a classical "
                "condition; classically-controlled operations are not "
                "supported by the qubit-reuse compiler"
            )

        overlap = measured_set.intersection(qidx)
        if overlap:
            raise NotImplementedError(
                f"operation '{name}' acts on already-measured qubit(s) "
                f"{sorted(overlap)}; the input circuit must contain terminal "
                "measurements only"
            )

        if name == "reset":
            operations.append(Op("reset", (qidx[0],), (), (), order))
            order += 1
            continue

        if name not in SUPPORTED_GATES:
            raise NotImplementedError(
                f"gate '{name}' is not supported by the qubit-reuse parser; "
                f"supported gates: {sorted(SUPPORTED_GATES)}"
            )

        try:
            params = tuple(float(p) for p in inst.operation.params)
        except (TypeError, ValueError) as exc:
            raise NotImplementedError(
                f"gate '{name}' has unbound/symbolic parameters; bind all "
                "parameters before qubit-reuse compilation"
            ) from exc

        n_ctrl = int(getattr(inst.operation, "num_ctrl_qubits", 0) or 0)
        controls = tuple(qidx[:n_ctrl])
        targets = tuple(qidx[n_ctrl:])
        operations.append(Op(name, targets, controls, params, order))
        order += 1

    return LogicalCircuitIR(
        num_qubits=qc.num_qubits,
        operations=operations,
        measured_outputs=measured,
        output_to_clbit=output_to_clbit,
        num_clbits=qc.num_clbits,
    )
