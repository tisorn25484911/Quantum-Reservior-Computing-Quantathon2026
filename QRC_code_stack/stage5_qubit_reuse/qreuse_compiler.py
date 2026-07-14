"""qreuse_compiler.py -- greedy causal-cone qubit-reuse compiler (Qiskit lowering).

Given a LogicalCircuitIR with terminal measurements, a measurement order,
and the output causal cones, emit a new Qiskit QuantumCircuit that:

  - executes causal cones sequentially in the scheduled order,
  - measures each output qubit as soon as its cone is complete
    (mid-circuit measurement),
  - resets freed physical qubits and reuses them for later logical qubits,
  - writes every measurement into the SAME classical bit index as the
    original circuit, so count dictionaries are directly comparable.

Equivalence contract (Sec. II of PRX 13, 041057): the compiled circuit
R(C) is equivalent to the original C only at the level of the classical
measurement-output distribution, R(C) ~= C.  It does NOT reproduce the
pre-measurement quantum state of C (mid-circuit measurement and reset
destroy it), does not implement the same operator, and acts on a Hilbert
space of a different dimension.  No quantum post-processing of C's final
state can be carried over to R(C).

Implementation notes:
  - Gates outside every measured output's cone cannot influence any
    recorded measurement and are dropped (free dead-gate elimination).
  - Physical qubits are allocated lazily (first touch) and resets are
    deferred to the moment of reuse, so the reset count equals the number
    of actual reuses rather than the number of measurements.
  - Lowering uses qc.measure(phys, clbit) + qc.reset(phys), which Qiskit
    Aer and Quantinuum/IBM dynamic-circuit backends execute as mid-circuit
    measurement and reset.  On IBM Runtime backends that support it, the
    dedicated MidCircuitMeasure instruction is typically preferred over a
    plain measure for the non-terminal measurements because it is
    optimized for mid-circuit use; swap it in at the marked line below.

TODO hooks: CP-SAT exact optimizer, dual-circuit heuristic, hardware-aware
routing, noise-aware measurement scheduling, and a Qiskit transpiler-pass
wrapper all live behind the scheduler interface (see qreuse_scheduler.py).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set

from qiskit import QuantumCircuit

from qreuse_ir import LogicalCircuitIR, Op, ReuseCompiledCircuit
from qreuse_analysis import compute_output_cones_and_ops
from qreuse_scheduler import (
    compiled_qubit_count,
    greedy_with_first_qubit_search,
)


def _emit_gate(qc: QuantumCircuit, op: Op, phys: List[int]) -> None:
    """Lower one IR op onto physical qubits of `qc`.

    Every SUPPORTED_GATES name is a QuantumCircuit method taking
    (*params, *qubits) with controls first -- the same order Op.qubits uses.
    """
    if op.name == "reset":
        qc.reset(phys[0])
        return
    getattr(qc, op.name)(*op.params, *phys)


def compile_with_reuse(
    ir: LogicalCircuitIR,
    order: Optional[List[int]] = None,
    cones: Optional[Dict[int, Set[int]]] = None,
) -> ReuseCompiledCircuit:
    """Compile `ir` into a fewer-qubit circuit via mid-circuit measure/reset.

    Parameters
    ----------
    ir : LogicalCircuitIR
        Parsed original circuit (terminal measurements only).
    order : optional list of measured output qubits
        Causal-cone execution order.  Defaults to the greedy heuristic with
        brute-force first-qubit search.
    cones : optional {output: qubit set}
        Precomputed causal cones.  If given, they must match the cones
        recomputed from `ir` (they are a pure function of the circuit);
        a mismatch raises ValueError.

    Returns
    -------
    ReuseCompiledCircuit
    """
    if not ir.measured_outputs:
        raise ValueError("circuit has no measured outputs; nothing to compile")

    cones_and_ops = compute_output_cones_and_ops(ir)
    computed_cones = {q: set(c) for q, (c, _) in cones_and_ops.items()}
    cone_ops = {q: ops for q, (_, ops) in cones_and_ops.items()}

    if cones is not None:
        provided = {q: set(c) for q, c in cones.items()}
        if provided != computed_cones:
            raise ValueError(
                "provided cones do not match the cones computed from the IR"
            )
    cones = computed_cones

    if order is None:
        order = greedy_with_first_qubit_search(cones)
    if sorted(order) != sorted(ir.measured_outputs):
        raise ValueError(
            "measurement order must contain each measured output exactly once"
        )

    n_phys = compiled_qubit_count(order, cones)
    qc = QuantumCircuit(n_phys, ir.num_clbits)

    ops_by_id = {op.order: op for op in ir.operations}
    logical_to_physical: Dict[int, int] = {}
    physical_to_logical: Dict[int, int] = {}
    free_physical: deque = deque()
    measured_logicals: Set[int] = set()
    emitted: Set[int] = set()
    history: List[Dict] = []
    classical_output_map: Dict[int, int] = {}
    next_fresh = 0

    def ensure_assigned(logical: int) -> int:
        """Return the physical qubit hosting `logical`, allocating lazily."""
        nonlocal next_fresh
        if logical in measured_logicals:
            raise RuntimeError(
                f"internal error: logical qubit {logical} referenced after "
                "its measurement (input circuit not terminal-measured?)"
            )
        if logical in logical_to_physical:
            return logical_to_physical[logical]
        if free_physical:
            phys = free_physical.popleft()
            qc.reset(phys)  # deferred reset: reset exactly at reuse time
        else:
            phys = next_fresh
            next_fresh += 1
            if next_fresh > n_phys:
                raise RuntimeError(
                    "internal error: allocation exceeded the scheduler's "
                    "peak-active-qubit estimate"
                )
        logical_to_physical[logical] = phys
        physical_to_logical[phys] = logical
        return phys

    for step, q in enumerate(order):
        # 1-2. ensure cone qubits are mapped and emit its missing operations,
        #      in original circuit order (legal by downward closure).
        for op_id in sorted(cone_ops[q] - emitted):
            op = ops_by_id[op_id]
            phys = [ensure_assigned(l) for l in op.qubits]
            _emit_gate(qc, op, phys)
            emitted.add(op_id)

        # 3-4. measure the scheduled output into its ORIGINAL classical bit.
        phys_q = ensure_assigned(q)  # covers outputs with empty cones
        clbit = ir.output_to_clbit[q]
        # NOTE(IBM Runtime): replace this plain measure with the
        # MidCircuitMeasure instruction on backends that support it; it is
        # the preferred primitive for non-terminal measurements.
        qc.measure(phys_q, clbit)
        classical_output_map[q] = clbit
        history.append(
            {
                "step": step,
                "measured_logical": q,
                "physical": phys_q,
                "logical_to_physical": dict(logical_to_physical),
            }
        )

        # 5-6. free the physical qubit for later reuse (reset is deferred
        #      to the moment of reassignment inside ensure_assigned).
        measured_logicals.add(q)
        del logical_to_physical[q]
        del physical_to_logical[phys_q]
        free_physical.append(phys_q)

    if next_fresh != n_phys:
        raise RuntimeError(
            "internal error: allocated qubit count disagrees with the "
            f"scheduler estimate ({next_fresh} != {n_phys})"
        )

    # Completeness: every operation in every cone must have been emitted
    # exactly once, and nothing outside the cone union may have been.
    cone_union: Set[int] = set().union(*cone_ops.values()) if cone_ops \
        else set()
    if emitted != cone_union:
        raise RuntimeError(
            "internal error: emitted operation set differs from the union "
            f"of causal cones ({len(emitted)} vs {len(cone_union)})"
        )
    n_gate_ops = sum(1 for op in ir.operations if op.name != "measure")

    return ReuseCompiledCircuit(
        circuit=qc,
        physical_qubit_count=next_fresh,
        logical_to_physical_history=history,
        classical_output_map=classical_output_map,
        measurement_order=list(order),
        cones=cones,
        dropped_operations=n_gate_ops - len(cone_union),
    )
