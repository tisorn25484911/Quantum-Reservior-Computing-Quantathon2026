"""noise_models.py -- synthetic channels + stored backend-snapshot noise model.

Two sources, per spec s11:
  - ``build_synthetic``: readout error + 1q/2q depolarising + optional thermal
    relaxation, all from ``noise_models.yaml`` knobs. Deterministic, controllable.
  - ``from_snapshot``: reconstruct a NoiseModel from a FROZEN backend snapshot
    (coupling map, basis gates, T1/T2, gate/readout errors) exported offline by
    ``scripts/export_backend_noise_model.py`` -- never live calibration.
Implemented in P3.
"""
from __future__ import annotations

import json
from pathlib import Path

from qiskit_aer.noise import (NoiseModel, ReadoutError, depolarizing_error,
                              thermal_relaxation_error)

_1Q = ["ry", "rz", "reset"]
_2Q = ["rxx"]


def build_synthetic(cfg):
    """NoiseModel from the ``synthetic`` block of ``noise_models.yaml``.

    Any channel set to 0 / omitted is skipped; an all-zero config yields an empty
    (ideal) NoiseModel.
    """
    nm = NoiseModel(basis_gates=["ry", "rz", "rxx", "reset"])
    p1 = float(cfg.get("depol_1q", 0.0) or 0.0)
    p2 = float(cfg.get("depol_2q", 0.0) or 0.0)
    ro = float(cfg.get("readout_p", 0.0) or 0.0)
    tr = cfg.get("thermal_relaxation") or {}

    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry", "rz"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["rxx"])
    if tr.get("t1_us") and tr.get("t2_us"):
        t1 = tr["t1_us"] * 1e3   # ns
        t2 = tr["t2_us"] * 1e3
        g1 = float(tr.get("gate_time_1q_ns", 50.0))
        g2 = float(tr.get("gate_time_2q_ns", 300.0))
        nm.add_all_qubit_quantum_error(
            thermal_relaxation_error(t1, t2, g1), ["ry", "rz"])
        nm.add_all_qubit_quantum_error(
            thermal_relaxation_error(t1, t2, g2).tensor(
                thermal_relaxation_error(t1, t2, g2)), ["rxx"])
    if ro > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))
    return nm


def from_snapshot(path):
    """Rebuild a NoiseModel from a frozen snapshot JSON (Aer ``to_dict`` format)."""
    p = Path(path)
    if p.is_dir():
        cands = sorted(p.glob("*.json"))
        if not cands:
            raise FileNotFoundError(f"no backend snapshot JSON under {p}")
        p = cands[-1]
    return NoiseModel.from_dict(json.loads(p.read_text()))
