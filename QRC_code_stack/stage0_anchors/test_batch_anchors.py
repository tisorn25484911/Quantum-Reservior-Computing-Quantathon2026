"""Stage-7 anchors: batching preserves per-block distributions exactly;
unbatch_counts partitions keys correctly (hand-built case); the compiled
batch is legal and narrower than logical."""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for rel in ("stage7_integration", "stage6_rfqrc"):
    p = str(_ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qreuse_batch import batch_circuits, compile_batch, unbatch_counts


def _block(theta: float) -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_unbatch_counts_hand_built():
    """Joint key '10 01' style partition, hand-computed. Batched clbits:
    block0 = clbits 0-1, block1 = clbits 2-3. Aer key is little-endian,
    so key 'dcba' has a=clbit0. Key '1101' => clbits (a,b,c,d)=(1,0,1,1)
    => block0 standalone key '01', block1 standalone key '11'."""
    counts = {"1101": 7, "0000": 3}
    b0, b1 = unbatch_counts(counts, [2, 2])
    assert b0 == {"01": 7, "00": 3}
    assert b1 == {"11": 7, "00": 3}


def test_batching_preserves_block_distributions_exactly():
    """Marginals of the batched statevector equal each standalone block
    distribution to 1e-12 (blocks are independent by construction)."""
    blocks = [_block(0.7), _block(2.1)]
    batched = batch_circuits(blocks)
    sv = Statevector(batched.remove_final_measurements(inplace=False))
    joint = {k: float(v) for k, v in sv.probabilities_dict().items()}
    margs = unbatch_counts(joint, [2, 2])
    for blk, marg in zip(blocks, margs):
        ref = Statevector(
            blk.remove_final_measurements(inplace=False)).probabilities_dict()
        keys = set(ref) | set(marg)
        tvd = 0.5 * sum(abs(ref.get(k, 0.0) - marg.get(k, 0.0))
                        for k in keys)
        assert tvd < 1e-12


def test_compiled_batch_legal_and_narrower():
    """compile_batch runs the frozen stage-5 validator internally (raises
    on any structural issue) and must reuse qubits across independent
    blocks."""
    blocks = [_block(0.3), _block(1.1), _block(2.5)]
    compiled = compile_batch(blocks)
    assert compiled.physical_qubit_count < 6   # 3 blocks x 2 qubits logical
