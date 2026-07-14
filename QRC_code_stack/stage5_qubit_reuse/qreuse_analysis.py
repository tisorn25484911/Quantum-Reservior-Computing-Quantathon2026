"""qreuse_analysis.py -- causal-cone analysis.

For each terminally measured output qubit q, compute its backward causal
cone: the set of logical input qubits (and the set of operations) that can
influence the distribution sampled by q's measurement.

Reverse-pass algorithm (Sec. II of PRX 13, 041057):

    cone = {q}
    for op in reversed(circuit):
        if op touches any qubit in cone:
            cone |= qubits(op)

One-qubit gates never expand a cone; a two-qubit gate pulls its partner in
whenever either qubit is already in the cone.

Two structural facts the compiler relies on (both follow from the reverse
pass plus terminal measurements):

  1. Downward closure: if op B is in cone(q) and op A precedes B sharing a
     qubit with it, then A is in cone(q).  Hence emitting the not-yet-emitted
     ops of each cone in original circuit order always respects dependencies.
  2. Every operation touching q is in cone(q) (all of them precede q's
     terminal measurement).  Hence after cone(q) is emitted and q measured,
     no later operation references q -- its physical qubit is safely reusable.

Input `reset` operations are treated conservatively as ordinary one-qubit
operations (a reset actually *blocks* influence from earlier operations, so
this over-approximates cones; over-approximation is always correct, merely
less compressive).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Set, Tuple

from qreuse_ir import LogicalCircuitIR, Op


def compute_output_cones(ir: LogicalCircuitIR) -> Dict[int, Set[int]]:
    """Map each measured output qubit to its input causal-cone qubit set."""
    return {q: set(cone) for q, (cone, _) in
            compute_output_cones_and_ops(ir).items()}


def compute_output_cones_and_ops(
    ir: LogicalCircuitIR,
) -> Dict[int, Tuple[Set[int], FrozenSet[int]]]:
    """As compute_output_cones, but also return the op ids in each cone.

    Returns {output_qubit: (cone_qubits, frozenset_of_op_orders)}.
    Measurement ops are excluded from the walk (measuring another qubit
    never influences q under terminal measurement).
    """
    gate_ops: List[Op] = [op for op in ir.operations if op.name != "measure"]
    reversed_ops = list(reversed(gate_ops))
    reversed_qubit_sets = [frozenset(op.qubits) for op in reversed_ops]

    result: Dict[int, Tuple[Set[int], FrozenSet[int]]] = {}
    for q in ir.measured_outputs:
        cone: Set[int] = {q}
        op_ids: Set[int] = set()
        for op, qs in zip(reversed_ops, reversed_qubit_sets):
            if not qs.isdisjoint(cone):
                cone |= qs
                op_ids.add(op.order)
        result[q] = (cone, frozenset(op_ids))
    return result
