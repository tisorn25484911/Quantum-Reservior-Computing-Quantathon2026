"""qreuse_batch.py -- batch independent step circuits through the stage-5
reuse compiler (stage 7, Part X Phase 6a).

RF-QRC step circuits are mutually independent and carry terminal
measurements only, so a batch of B of them laid side by side on disjoint
qubit registers is parser-legal for the stage-5 compiler TODAY (Part X
sec. on the reuse fit). The compiler modules are imported FROZEN --
nothing in stage 5 is modified (Part IX promotion protocol).

The structural payoff: causal cones never cross block boundaries, so the
scheduler reuses physical qubits ACROSS blocks and the compiled width is
set by one block's cone structure, independent of the batch size B and --
because RF-QRC's memory lives in the classical leak -- independent of the
memory horizon. Contrast the windowed protocol's measured physical
count = T + 1 (stage-5 depth scan).

Endianness: unbatch_counts slices Aer count keys, which are little-endian
over clbits (clbit 0 = rightmost character). The slice is performed on
the reversed key so block b with clbits [lo, hi) takes characters
[lo, hi) -- then re-reversed so each block's counts look exactly like a
standalone run of that block. counts_to_features downstream stays the
single semantic reversal point (R3): this function only PARTITIONS keys,
it never reorders bits within a block.
"""

from __future__ import annotations

from typing import Dict, List

from qiskit import QuantumCircuit

from qreuse_analysis import compute_output_cones
from qreuse_compiler import compile_with_reuse
from qreuse_ir import ReuseCompiledCircuit, from_qiskit
from qreuse_scheduler import compiled_qubit_count, greedy_with_first_qubit_search
from qreuse_validation import validate_reuse_circuit


def batch_circuits(circs: List[QuantumCircuit]) -> QuantumCircuit:
    """Lay B independent circuits side by side on disjoint qubit and
    clbit ranges. Every input circuit must carry terminal measurements
    only (the stage-5 parser contract); measurement classical bits are
    offset per block in input order."""
    if not circs:
        raise ValueError("empty batch")
    n_q = sum(c.num_qubits for c in circs)
    n_c = sum(c.num_clbits for c in circs)
    if n_c == 0:
        raise ValueError("batched circuits must carry measurements")
    out = QuantumCircuit(n_q, n_c)
    q_off = c_off = 0
    for c in circs:
        out.compose(c,
                    qubits=range(q_off, q_off + c.num_qubits),
                    clbits=range(c_off, c_off + c.num_clbits),
                    inplace=True)
        q_off += c.num_qubits
        c_off += c.num_clbits
    return out


def compile_batch(circs: List[QuantumCircuit]) -> ReuseCompiledCircuit:
    """Batch, parse, compile, and structurally validate. Returns the
    stage-5 ReuseCompiledCircuit (frozen interface)."""
    batched = batch_circuits(circs)
    ir = from_qiskit(batched)
    compiled = compile_with_reuse(ir)
    validate_reuse_circuit(compiled, ir, raise_on_error=True)
    return compiled


def compiled_width_of_batch(circs: List[QuantumCircuit]) -> int:
    """Compiled physical qubit count without emitting the full circuit
    (scheduler dry run on the causal cones)."""
    ir = from_qiskit(batch_circuits(circs))
    cones = compute_output_cones(ir)
    order = greedy_with_first_qubit_search(cones)
    return compiled_qubit_count(order, cones)


def unbatch_counts(counts: Dict[str, int],
                   clbit_sizes: List[int]) -> List[Dict[str, int]]:
    """Marginalise joint batched counts back to per-block counts.

    clbit_sizes[b] = number of clbits of block b, in the same input order
    as batch_circuits. Keys are partitioned positionally; per-block keys
    come back in standalone little-endian convention (see module
    docstring)."""
    edges = [0]
    for s in clbit_sizes:
        edges.append(edges[-1] + s)
    out: List[Dict[str, int]] = [dict() for _ in clbit_sizes]
    for key, c in counts.items():
        rev = key.replace(" ", "")[::-1]          # rev[i] = clbit i
        if len(rev) < edges[-1]:
            raise ValueError(f"key {key!r} shorter than clbit total "
                             f"{edges[-1]}")
        for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            sub = rev[lo:hi][::-1]                 # standalone convention
            out[b][sub] = out[b].get(sub, 0) + c
    return out
