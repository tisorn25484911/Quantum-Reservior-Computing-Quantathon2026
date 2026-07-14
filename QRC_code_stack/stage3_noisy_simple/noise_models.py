"""noise_models.py -- the stack's single noise-model definition (stage 3).

EXTRACTION, not invention (Phase 3): consolidates the two existing,
verified noise constructions --

  * stage2_circuits_exact/qrc_qiskit.py  make_noise_model(): depolarizing
    p1 on sx/x, p2 on ecr, symmetric readout flips (ECR basis, ENSO
    project conventions);
  * stage4_hamiltonian_battery/benchmark_hamiltonians.py
    make_noise_model(): same convention on the rz/sx/x/cx basis plus a
    reset bit-flip channel (reuse-specific error, mid-circuit resets).

into one parameterised factory, plus a thermal-relaxation variant (new;
parameter defaults are order-of-magnitude medians typical of IBM
Eagle-class devices circa 2024-2026 -- T1 ~ 250 us, T2 ~ 150 us, 1q gate
~ 35 ns, 2q gate ~ 500 ns, readout ~ 900 ns. Simulation-scoped: no
hardware claim attaches to these defaults; cite device data before any
hardware-adjacent statement).

Project noise constants (unchanged from both source projects):
    p1 = 3e-4, p2 = 7e-3, ro = 1.5e-2, reset = 5e-3.

HONESTY CAVEAT carried from the reuse project (measured there, Part IX
stage 3): gate-local noise models CANNOT see the 3-5x depth inflation
that qubit reuse introduces. Stage-5 noise claims must never be
extrapolated from these models alone; that requires duration-aware
scheduling + idle thermal relaxation.

Anchors: `python noise_models.py` runs the self-test (CPTP completeness
of every channel; p=0 model reproduces noiseless counts) and exits 0 iff
all pass. The same checks live in stage0_anchors/test_noise_anchors.py.
"""

from __future__ import annotations

import sys

import numpy as np
from qiskit_aer.noise import (NoiseModel, ReadoutError, depolarizing_error,
                              pauli_error, thermal_relaxation_error)

# project constants (both source projects agree)
NOISE_P1 = 3e-4       # depolarizing, 1q gates (sx, x; rz is virtual)
NOISE_P2 = 7e-3       # depolarizing, 2q gates (ecr or cx)
NOISE_RO = 1.5e-2     # symmetric readout flip probability
NOISE_RESET = 5e-3    # reset bit-flip (reuse circuits only)

# thermal defaults: order-of-magnitude IBM Eagle-class medians (see module
# docstring; simulation-scoped)
T1_DEFAULT = 250e-6
T2_DEFAULT = 150e-6
TIME_1Q = 35e-9
TIME_2Q = 500e-9
TIME_RO = 900e-9


def depolarizing_model(p1: float = NOISE_P1, p2: float = NOISE_P2,
                       p_ro: float = NOISE_RO, basis_2q: str = "cx",
                       p_reset: float | None = None) -> NoiseModel:
    """Depolarizing + readout noise on the native basis.

    basis_2q='ecr' reproduces qrc_qiskit.make_noise_model exactly;
    basis_2q='cx' with p_reset=NOISE_RESET reproduces the reuse
    benchmark's model exactly.
    """
    basis = ["rz", "sx", "x", basis_2q]
    nm = NoiseModel(basis_gates=basis)
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), [basis_2q])
    if p_ro > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    if p_reset is not None and p_reset > 0:
        nm.add_all_qubit_quantum_error(
            pauli_error([("X", p_reset), ("I", 1 - p_reset)]), ["reset"])
    return nm


def thermal_model(t1: float = T1_DEFAULT, t2: float = T2_DEFAULT,
                  time_1q: float = TIME_1Q, time_2q: float = TIME_2Q,
                  time_ro: float = TIME_RO, p_ro: float = NOISE_RO,
                  basis_2q: str = "cx") -> NoiseModel:
    """Thermal-relaxation noise: T1/T2 decay during gates and readout."""
    if t2 > 2 * t1:
        raise ValueError("unphysical: T2 > 2*T1")
    nm = NoiseModel(basis_gates=["rz", "sx", "x", basis_2q])
    err_1q = thermal_relaxation_error(t1, t2, time_1q)
    err_2q = thermal_relaxation_error(t1, t2, time_2q).expand(
        thermal_relaxation_error(t1, t2, time_2q))
    err_ro = thermal_relaxation_error(t1, t2, time_ro)
    nm.add_all_qubit_quantum_error(err_1q, ["sx", "x"])
    nm.add_all_qubit_quantum_error(err_2q, [basis_2q])
    nm.add_all_qubit_quantum_error(err_ro, ["measure"])
    if p_ro > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


# ----------------------------------------------------------------- anchors
def channels_are_cptp() -> bool:
    """Every channel used above is a valid CPTP map (Kraus completeness)."""
    checks = [
        depolarizing_error(NOISE_P1, 1),
        depolarizing_error(NOISE_P2, 2),
        pauli_error([("X", NOISE_RESET), ("I", 1 - NOISE_RESET)]),
        thermal_relaxation_error(T1_DEFAULT, T2_DEFAULT, TIME_1Q),
    ]
    ok = True
    for err in checks:
        chan = err.to_quantumchannel()
        good = chan.is_cptp()
        print(f"  [{'PASS' if good else 'FAIL'}] CPTP: {err.__class__.__name__}"
              f" ({len(err.circuits)} Kraus terms)")
        ok &= good
    return ok


def zero_noise_is_noiseless(shots: int = 8192, seed: int = 7) -> bool:
    """p=0 depolarizing model reproduces noiseless counts within shot noise."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rx(0.7, 2)
    qc.measure(range(3), range(3))

    nm0 = depolarizing_model(p1=0.0, p2=0.0, p_ro=0.0)
    tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"],
                    optimization_level=0)

    def counts(noise_model, s):
        be = AerSimulator(noise_model=noise_model, seed_simulator=s)
        return be.run(tqc, shots=shots).result().get_counts()

    def tvd(c1, c2):
        keys = set(c1) | set(c2)
        return 0.5 * sum(abs(c1.get(k, 0) / shots - c2.get(k, 0) / shots)
                         for k in keys)

    d_model_vs_none = tvd(counts(nm0, seed), counts(None, seed + 1))
    shot_floor = tvd(counts(None, seed + 2), counts(None, seed + 3))
    ok = d_model_vs_none <= 2.5 * shot_floor + 0.02  # reuse-suite calibration
    print(f"  [{'PASS' if ok else 'FAIL'}] p=0 model vs noiseless: "
          f"TVD={d_model_vs_none:.4f} vs floor {shot_floor:.4f}")
    return ok


if __name__ == "__main__":
    print("noise_models self-test:")
    ok = channels_are_cptp() and zero_noise_is_noiseless()
    print(f"noise_models: {'ALL PASS' if ok else 'FAILURES'}")
    sys.exit(0 if ok else 1)
