"""latency_model.py -- per-step sequential-QPU latency ledger (spec s26).

The rewind architecture forces ONE QPU round trip per closed-loop forecast step:
the circuit for step t+1 depends on the RESULT of step t (the predicted value
re-enters the rewind window), so steps cannot be batched away. This module builds a
per-step ledger from two clearly separated sources:

  * MEASURED (local, this machine): circuit construct / parameter bind / transpile /
    readout maths / local-simulator execute / buffer update -- wall-clocked here.
  * ESTIMATED (remote, never live): queue / network / QPU execute -- read from the
    frozen ``latency_assumptions.yaml`` beside the backend snapshot. Heron-class
    execution anchors are recorded there as ASSUMPTIONS, not facts, and are labelled
    as such wherever they surface in results.

Strategy comparison A-G (spec s26): each strategy is a MAP saying which components a
single closed-loop step incurs (see ``STRATEGIES``). The consumer
(``sequential_execution``) multiplies the per-step total by the number of SEQUENTIAL
steps -- batched open-loop timing is explicitly NOT closed-loop evidence (acceptance
item 24). Implemented in P11.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..quantum.hamiltonians import nn_tfi
from ..quantum.schedule import build_block_schedule
from ..quantum.qiskit_qrc import block_circuit, _theta
from ..quantum import endianness as E
from ..utils.configuration import load_yaml

# ---------------------------------------------------------------- assumptions
_DEFAULT_ASSUMPTIONS = (
    Path(__file__).resolve().parents[3]
    / "results" / "backend_snapshots" / "latency_assumptions.yaml"
)


@dataclass(frozen=True)
class LatencyAssumptions:
    """Remote (never-live) timing anchors, read from the frozen assumptions file."""

    backend_name: str
    n_qubits: int
    basis_gates: tuple
    coupling: str
    per_shot_execution_s: float
    sampler_overhead_s: float
    queue_s: float
    network_send_s: float
    result_return_s: float
    refresh_period_steps: int
    sources: tuple = ()

    @classmethod
    def load(cls, path=None):
        d = load_yaml(path or _DEFAULT_ASSUMPTIONS)
        b = d["backend"]
        return cls(
            backend_name=b["name"],
            n_qubits=int(b["n_qubits"]),
            basis_gates=tuple(b["basis_gates"]),
            coupling=b["coupling"],
            per_shot_execution_s=float(d["per_shot_execution_s"]),
            sampler_overhead_s=float(d["sampler_overhead_s"]),
            queue_s=float(d["queue_s"]),
            network_send_s=float(d["network_send_s"]),
            result_return_s=float(d["result_return_s"]),
            refresh_period_steps=int(d["refresh_period_steps"]),
            sources=tuple(d.get("sources", [])),
        )

    def execute_s(self, shots):
        """ESTIMATED QPU execution for one job: session overhead + shots x anchor."""
        return self.sampler_overhead_s + shots * self.per_shot_execution_s


# ------------------------------------------------------- rewind step circuit
def build_rewind_step_circuit(N, t_w, schedule, window=None, parameterised=False,
                              readout="save_dm"):
    """One rewind-window forecast circuit: t_w (reset, Ry, Trotter block) injections.

    ``parameterised`` builds the injection angles as free ``Parameter``s (a reusable
    pre-transpiled template, strategy B); otherwise the concrete ``window`` angles are
    baked in (a fresh circuit, strategy A). ``readout`` selects how virtual nodes are
    read: ``"save_dm"`` saves a density matrix after every injection (simulation-only,
    G8-7; the ideal-path readout) while ``"measure"`` appends a terminal Z measurement
    -- the hardware-shaped form used for transpile timing (Aer save instructions do
    not translate to a device basis).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    if window is None:
        window = np.full(t_w, 0.5)
    inj = E.logical_to_wire(0, N)                       # injection at site 0 (G2)
    qc = QuantumCircuit(N, N) if readout == "measure" else QuantumCircuit(N)
    for i in range(t_w):
        qc.reset(inj)
        if parameterised:
            qc.ry(Parameter(f"u{i}"), inj)
        else:
            qc.ry(_theta(float(window[i])), inj)
        block_circuit(schedule, N, qc)
        if readout == "save_dm":
            qc.save_density_matrix(label=f"n_{i}")
    if readout == "measure":
        qc.measure(range(N), range(N))
    return qc


# --------------------------------------------------------- measured (local)
@dataclass
class MeasuredLocal:
    """Wall-clocked local components (seconds), median over repeats."""

    construct_s: float
    bind_s: float
    transpile_s: float
    readout_s: float
    local_execute_s: float
    buffer_s: float
    meta: dict = field(default_factory=dict)


def _median_time(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def measure_local_components(N=5, t_w=10, tau=1.0, shots=1024, repeats=5, seed=7):
    """Wall-clock the local ledger components for one rewind forecast step.

    Everything here runs on THIS machine (construct/bind/transpile on Qiskit,
    local execute on Aer, readout in numpy) so the numbers are genuine measurements,
    not estimates. Remote components are added later from ``LatencyAssumptions``.
    """
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    from qiskit.quantum_info import DensityMatrix, Pauli
    from qiskit_aer import AerSimulator

    res = nn_tfi(N, J=1.0, h=0.5, seed=seed)
    schedule = build_block_schedule(res, block_time=tau, kappa=1, order=1)
    rng = np.random.default_rng(seed)
    window = rng.uniform(0.0, 1.0, size=t_w)
    coupling = CouplingMap.from_line(N)
    basis = ["rz", "sx", "x", "cz"]

    # construct: build a fresh concrete circuit (strategy A pays this every step)
    construct_s = _median_time(
        lambda: build_rewind_step_circuit(N, t_w, schedule, window), repeats)

    # bind: assign angles into a pre-built parameterised template (strategy B)
    template = build_rewind_step_circuit(N, t_w, schedule, parameterised=True)
    params = list(template.parameters)
    angles = {p: _theta(float(window[int(p.name[1:])])) for p in params}
    bind_s = _median_time(lambda: template.assign_parameters(angles), repeats)

    # transpile: map a hardware-shaped circuit (terminal measurement, no Aer saves)
    # to the assumed basis/coupling
    qc_hw = build_rewind_step_circuit(N, t_w, schedule, window, readout="measure")
    transpile_s = _median_time(
        lambda: transpile(qc_hw, coupling_map=coupling, basis_gates=basis,
                          optimization_level=1, seed_transpiler=seed), repeats)

    # local execute: run the density-matrix circuit on Aer (strategy D uses this)
    sim = AerSimulator(method="density_matrix")
    qc_exec = build_rewind_step_circuit(N, t_w, schedule, window)
    local_execute_s = _median_time(lambda: sim.run(qc_exec).result(), repeats)

    # readout maths: expectations -> features -> ridge predict (numpy only)
    data = sim.run(qc_exec).result().data(0)
    M = N
    W = rng.standard_normal(M * t_w + 1)

    def _readout():
        rows = np.empty((t_w, M))
        for i in range(t_w):
            dm = DensityMatrix(data[f"n_{i}"])
            z = np.array([dm.expectation_value(Pauli("Z"),
                          [E.logical_to_wire(q, N)]).real for q in range(N)])
            rows[i] = 0.5 * (1.0 + z)
        feat = np.concatenate([rows.reshape(-1), [1.0]])
        return float(feat @ W)

    readout_s = _median_time(_readout, repeats)

    # buffer update: push one predicted value into a length-t_w ring buffer
    buf = np.zeros(t_w)

    def _buffer():
        buf[:-1] = buf[1:]
        buf[-1] = 0.123

    buffer_s = _median_time(_buffer, max(repeats, 50))

    return MeasuredLocal(
        construct_s=construct_s, bind_s=bind_s, transpile_s=transpile_s,
        readout_s=readout_s, local_execute_s=local_execute_s, buffer_s=buffer_s,
        meta={"N": N, "t_w": t_w, "tau": tau, "shots": shots,
              "circuit_depth": int(qc_hw.depth()), "repeats": repeats},
    )


# -------------------------------------------------------- strategy component map
# For each closed-loop step, a strategy incurs a subset of the local components and
# (if it touches the QPU) the remote ones. ``remote`` gates queue/network/execute;
# ``transpile``/``construct``/``bind`` toggle the per-step local build cost;
# ``closed_loop_capable`` records whether the strategy can actually roll the model
# forward on its OWN predictions (E cannot -- it only has offline features).
STRATEGIES = {
    "A_fresh_circuits": dict(
        label="A: fresh circuits (transpile every step)",
        remote=True, construct=True, bind=False, transpile=True,
        local_execute=False, surrogate=False, closed_loop_capable=True,
        note="worst case: full construct+transpile per step, one QPU round trip."),
    "B_pretranspiled_templates": dict(
        label="B: pre-transpiled parameterised template (bind only)",
        remote=True, construct=False, bind=True, transpile=False,
        local_execute=False, surrogate=False, closed_loop_capable=True,
        note="template transpiled once; per step only binds angles + one round trip."),
    "C_dynamic_circuits": dict(
        label="C: dynamic circuits (on-device feedforward)",
        remote=True, construct=False, bind=False, transpile=False,
        local_execute=False, surrogate=False, closed_loop_capable=True,
        requires_dynamic=True,
        note="needs snapshot-confirmed dynamic-circuit support; angles set on-device, "
             "still one QPU round trip per FORECAST step (cross-step dependency stays)."),
    "D_local_noisy_sim": dict(
        label="D: local noisy simulator (no QPU)",
        remote=False, construct=True, bind=False, transpile=True,
        local_execute=True, surrogate=False, closed_loop_capable=True,
        note="research mode: everything local; no queue/network."),
    "E_offline_features_classical_deploy": dict(
        label="E: offline QPU features + classical deploy",
        remote=False, construct=False, bind=False, transpile=False,
        local_execute=False, surrogate=False, closed_loop_capable=False,
        note="QPU paid ONCE offline; deploy is classical readout only. Cannot obtain "
             "quantum features for its OWN predicted inputs -> not closed-loop capable."),
    "F_classical_surrogate": dict(
        label="F: classical surrogate of the feature map (no QPU)",
        remote=False, construct=False, bind=False, transpile=False,
        local_execute=False, surrogate=True, closed_loop_capable=True,
        note="small model maps window->features locally; the Gate-7 cost comparator."),
    "G_periodic_refresh": dict(
        label="G: periodic quantum refresh (QPU every R steps)",
        remote=True, construct=False, bind=True, transpile=False,
        local_execute=False, surrogate=True, closed_loop_capable=True,
        periodic=True,
        note="amortised: one QPU round trip (template B) every R steps; surrogate "
             "carries the other R-1 steps."),
}


def step_ledger(strategy_key, measured, asm, shots, surrogate_predict_s=0.0):
    """Per-step latency ledger (seconds) for one strategy: components + total.

    Returns an ordered dict of component -> seconds, a ``total_s``, and provenance
    flags (``remote``, ``closed_loop_capable``, ``estimated``). ``surrogate_predict_s``
    is the measured cost of one surrogate feature evaluation (strategies F/G).
    """
    s = STRATEGIES[strategy_key]
    comp = {}

    # ---- local build components -------------------------------------------
    comp["construct"] = measured.construct_s if s["construct"] else 0.0
    comp["bind"] = measured.bind_s if s["bind"] else 0.0
    comp["transpile"] = measured.transpile_s if s["transpile"] else 0.0

    # ---- feature acquisition ----------------------------------------------
    if s.get("surrogate") and not s.get("periodic"):
        comp["surrogate_features"] = surrogate_predict_s
    if s["local_execute"]:
        comp["local_execute"] = measured.local_execute_s

    # ---- remote (ESTIMATED) round trip ------------------------------------
    if s["remote"] and not s.get("periodic"):
        comp["queue"] = asm.queue_s
        comp["network_send"] = asm.network_send_s
        comp["execute"] = asm.execute_s(shots)
        comp["result_return"] = asm.result_return_s

    # ---- always-local tail -------------------------------------------------
    comp["readout"] = measured.readout_s
    comp["buffer"] = measured.buffer_s

    total = float(sum(comp.values()))

    # ---- strategy G: amortise a full B round trip over R steps ------------
    if s.get("periodic"):
        R = max(1, asm.refresh_period_steps)
        refresh_step = step_ledger("B_pretranspiled_templates", measured, asm, shots)
        surrogate_step = step_ledger("F_classical_surrogate", measured, asm, shots,
                                     surrogate_predict_s=surrogate_predict_s)
        total = (refresh_step["total_s"] + (R - 1) * surrogate_step["total_s"]) / R
        comp = {"amortised_over_R": float(R),
                "refresh_step_total_s": refresh_step["total_s"],
                "surrogate_step_total_s": surrogate_step["total_s"]}

    return {
        "strategy": strategy_key,
        "label": s["label"],
        "components_s": comp,
        "total_s": total,
        "remote": bool(s["remote"]),
        "closed_loop_capable": bool(s["closed_loop_capable"]),
        "estimated": bool(s["remote"]),               # remote parts are ASSUMPTIONS
        "requires_dynamic": bool(s.get("requires_dynamic", False)),
        "note": s["note"],
    }


def total_time(step_total_s, n_steps):
    """Wall-clock for ``n_steps`` SEQUENTIAL closed-loop steps (no batching)."""
    return float(step_total_s) * int(n_steps)
