"""qreuse_scheduler.py -- measurement-order scheduling.

Implements the local greedy heuristic of Sec. III.B of PRX 13, 041057 and
its brute-force-first-qubit improvement.  The order in which output causal
cones are executed determines the peak number of simultaneously active
physical qubits, i.e. the compiled qubit count.

`estimate_peak_active_qubits` exactly simulates the allocator used by
qreuse_compiler.compile_with_reuse, so the estimate returned here equals
the physical qubit count of the compiled circuit for the same (order,
cones).

TODO hooks (deliberately unimplemented in this first version):
  - exact CP-SAT optimizer: binary variables m_qt / c_qt with constraints
    (C.1)-(C.7) of the paper, via Google OR-Tools; exact but scales
    superpolynomially (useful benchmark below ~30-50 qubits).
  - dual-circuit heuristic: compile the dual circuit C* (swap state preps
    and measurements, reverse time) and keep the better of R(C) and
    R(C*)* -- optimal compression of C and C* are provably equal
    (Sec. III.C), but the greedy heuristic need not be, so trying both
    is a free improvement.
  - brute-force over the first k measured qubits (O(N^k) overhead).
  - noise-aware objectives: trade qubit count against memory error by
    stopping reuse at the target device's qubit budget.
  - hardware-aware routing: apply reuse before SWAP insertion; co-optimize
    with the coupling map.
  - Qiskit TransformationPass wrapper so this runs inside a PassManager.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set


def greedy_measurement_order(
    cones: Dict[int, Set[int]], first: Optional[int] = None
) -> List[int]:
    """Local greedy causal-cone execution order.

    1. Measure the output qubit with the smallest causal cone (unless a
       specific `first` qubit is supplied).
    2. Repeatedly measure the remaining output whose cone adds the fewest
       new logical qubits to the union of all cones selected so far.

    Ties are broken by the smaller qubit index (deterministic output).
    """
    if not cones:
        return []
    remaining = set(cones)
    order: List[int] = []
    active_union: Set[int] = set()

    if first is None:
        first = min(remaining, key=lambda q: (len(cones[q]), q))
    elif first not in remaining:
        raise ValueError(f"first qubit {first} is not a measured output")

    order.append(first)
    active_union |= cones[first]
    remaining.discard(first)

    while remaining:
        nxt = min(remaining, key=lambda q: (len(cones[q] - active_union), q))
        order.append(nxt)
        active_union |= cones[nxt]
        remaining.discard(nxt)
    return order


def compiled_qubit_count(
    order: List[int], cones: Dict[int, Set[int]]
) -> int:
    """Width (physical qubit count) of the circuit compiled with `order`.

    Exact simulation of the compiler's allocator: at each step the new
    logical qubits of the current cone are drawn from the free pool first,
    fresh physical qubits otherwise; after each measurement one physical
    qubit returns to the pool.  The returned value is the total number of
    fresh allocations, which equals the compiled circuit's width because a
    physical qubit is never destroyed, only recycled.

    NOTE on naming: this is the compiled circuit WIDTH, i.e. the peak of
    (active + free-but-allocated), which upper-bounds the peak number of
    simultaneously *active* qubits.  Width is the quantity that matters --
    it is what the device must provide -- and it is what the compiler
    asserts against.  `estimate_peak_active_qubits` is a legacy alias.
    """
    seen: Set[int] = set()
    free = 0
    physical = 0
    for q in order:
        if q not in cones:
            raise KeyError(f"no causal cone for scheduled output {q}")
        n_new = len(cones[q] - seen)
        reused = min(free, n_new)
        free -= reused
        physical += n_new - reused
        seen |= cones[q]
        free += 1  # q is measured; its physical qubit is reclaimed
    return physical


#: Legacy alias kept for callers written against the original name.
estimate_peak_active_qubits = compiled_qubit_count


def greedy_with_first_qubit_search(cones: Dict[int, Set[int]]) -> List[int]:
    """Greedy heuristic with brute-force search over the first output qubit.

    Tries every possible first measured qubit, completes each ordering
    greedily, and returns the ordering with the smallest estimated peak
    active-qubit count (ties: the ordering found first, i.e. smallest
    first-qubit index).  O(N) times the cost of the plain greedy run;
    the paper reports ~13% average additional compression from this
    improvement on QAOA benchmarks.
    """
    best_order: Optional[List[int]] = None
    best_cost: Optional[int] = None
    for first in sorted(cones):
        order = greedy_measurement_order(cones, first=first)
        cost = compiled_qubit_count(order, cones)
        if best_cost is None or cost < best_cost:
            best_order, best_cost = order, cost
    return best_order if best_order is not None else []
