"""rfqrc_reservoir.py -- recurrence-free QRC with virtual-node readout.

Implements Part X of the handbook (design verified against Ahmed, Novoa,
Dalton & Magri, PRR 6 043082 (2024), arXiv:2405.03390, live-checked
2026-07-14):

    |psi_k> = V Phi(u_k) Phi(u_k) |0>^n          (double upload, eq. X.1)
    r_k     = (1 - eps) r_{k-1} + eps phi(u_k)   (classical leak, eq. X.2)
    y_hat_k = W_out r_k                          (ridge read-out, stage 6
                                                  Phase 2 / baselines)

phi(u_k) collects features measured at N_v intermediate evolution times
(virtual nodes): in exact mode, statevector snapshots at layer boundaries
(free); in shots mode, N_v truncated circuits sampled on Aer (the
reuse-fused realisation is a Stage-7 extension).

Endianness (R3): this module performs NO bit reordering of its own. The
shots path reuses stage 5's counts_to_features (THE single reversal
point); the exact path feeds Statevector.probabilities_dict() -- whose
keys use the same little-endian convention as Aer counts -- through the
same function. full_probs features are indexed by logical bitstring via
the same key[::-1] convention, in probs_to_features below, which is
anchor-tested against a hand-built basis state.

Echo-state property holds by construction for every eps in (0, 1]:
the leak map is affine in r with Lipschitz constant (1 - eps) < 1
(Part X Proposition; anchor-tested via the impulse response).

Requires stage5_qubit_reuse on PYTHONPATH (driver provides it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

from qrc_experiment import counts_to_features  # R3: single endianness point

SEED = 7


@dataclass
class RFQRCConfig:
    """Configuration surface, printed at runtime (Part X listing)."""
    n_qubits: int = 10
    window_m: int = 1            # 1 = pure RF-QRC; 2-3 = short in-circuit window
    n_uploads: int = 2           # RF-QRC signature; 1 = ablation
    entangler: str = "brickwork_zz"   # or "haar_control"
    entangler_layers: int = 2
    gamma: float = float(np.pi / 4)   # swept {pi/4, pi/2, pi}; or -1 => arcsin
    tau: float = 1.0             # global scale on entangler angles (J*dt)
    disorder_W: float = 0.0      # on-site RZ disorder inside the entangler
    n_virtual_nodes: int = 4     # snapshots at uniform layer boundaries
    readout: str = "full_probs"  # or "local_z", "local_zz"
    leak_eps: float = 0.3        # swept log-grid (0.02 .. 1.0]
    shots: Optional[int] = None  # None = exact snapshots
    denoise: str = "none"        # "svd:<k>", "savgol:<w>,<p>"
    seed: int = SEED

    def print_config(self) -> None:
        print("RFQRCConfig: " + " ".join(
            f"{k}={v}" for k, v in self.__dict__.items()))


def probs_to_features(probs: dict, n_qubits: int) -> np.ndarray:
    """Full-probability feature vector indexed by LOGICAL bitstring.

    Same key[::-1] reversal convention as counts_to_features (R3); the
    component for logical state |q0 q1 ... q_{n-1}> sits at index
    sum_i q_i 2^(n-1-i), i.e. qubit 0 is the most significant bit --
    matching the stage-1 reference convention (qubit 0 = leftmost tensor
    factor)."""
    out = np.zeros(2 ** n_qubits)
    for key, p in probs.items():
        bits = key.replace(" ", "")[::-1][:n_qubits]  # bits[i] = qubit i
        idx = int(bits, 2)  # qubit 0 -> MSB, matching stage-1 kron order
        out[idx] += p
    return out


# --------------------------------------------------------------- circuit
def make_entangler_params(cfg: RFQRCConfig) -> dict:
    """Frozen random reservoir parameters, drawn once (stage-5 pattern)."""
    rng = np.random.default_rng(cfg.seed)
    n, L = cfg.n_qubits, cfg.entangler_layers * cfg.n_uploads
    p = {
        "input_offset": rng.uniform(0.1, 0.6, size=n),
        "zz_angle": cfg.tau * rng.uniform(0.4, 1.1, size=(L, n)),
        "rz_disorder": rng.uniform(-cfg.disorder_W, cfg.disorder_W,
                                   size=(L, n)) if cfg.disorder_W > 0
                       else np.zeros((L, n)),
    }
    if cfg.entangler == "haar_control":
        from scipy.stats import unitary_group
        p["haar"] = [unitary_group.rvs(2 ** n, random_state=cfg.seed + i)
                     for i in range(L)]
    return p


def _encode(qc: QuantumCircuit, window: np.ndarray, cfg: RFQRCConfig,
            params: dict) -> None:
    """One data-encoding layer: the m most recent (multi-channel) inputs
    distributed across qubits; RY rotations.

    window has shape (m, n_channels) with window[0] = current input;
    slots = the m*n_channels flattened values. Assignment covers BOTH
    directions: when slots <= qubits, qubit q carries slot q mod n_slots
    (scalar replacement for a 1-channel drive); when slots > qubits,
    slot s lands on qubit s mod n as a stacked rotation, so EVERY channel
    is always encoded (a slot silently dropped is a blind reservoir --
    caught in the Phase-4 pilot, 2026-07-14)."""
    n = cfg.n_qubits
    flat = np.asarray(window, dtype=float).ravel()  # slot values in [0,1]
    n_slots = len(flat)
    for i in range(max(n, n_slots)):
        q = i % n
        x = float(flat[i % n_slots])
        if cfg.gamma < 0:   # arcsin encoding: wrap-free on [0, 1]
            angle = 2.0 * np.arcsin(np.sqrt(np.clip(x, 0.0, 1.0)))
        else:
            angle = cfg.gamma * x + params["input_offset"][q]
        qc.ry(angle, q)


def _entangle(qc: QuantumCircuit, layer: int, cfg: RFQRCConfig,
              params: dict) -> None:
    n = cfg.n_qubits
    if cfg.entangler == "haar_control":
        qc.unitary(params["haar"][layer], range(n), label=f"haar{layer}")
        return
    if cfg.entangler != "brickwork_zz":
        raise NotImplementedError(
            f"entangler {cfg.entangler!r}: ham_trotter families are a "
            "stage-7 extension (Part X)")
    for q in range(n):  # on-site disorder dial (parametric edge)
        if cfg.disorder_W > 0:
            qc.rz(params["rz_disorder"][layer, q], q)
    for i in range(layer % 2, n - 1, 2):   # alternating brickwork (stage-5
        j = i + 1                          #  factory pattern)
        theta = float(params["zz_angle"][layer, i])
        qc.cx(i, j)
        qc.rz(theta, j)
        qc.cx(i, j)


def build_step_circuit(window: np.ndarray, cfg: RFQRCConfig, params: dict,
                       n_layers: int | None = None,
                       measure: bool = False) -> QuantumCircuit:
    """The RF-QRC step circuit, optionally truncated after n_layers
    entangling layers (the shots-mode virtual-node realisation).

    Layer structure: for each of n_uploads uploads, one encoding layer
    followed by entangler_layers entangling layers. Total entangling
    layers L = n_uploads * entangler_layers; snapshots live at the L
    layer boundaries."""
    L_total = cfg.n_uploads * cfg.entangler_layers
    if n_layers is None:
        n_layers = L_total
    qc = QuantumCircuit(cfg.n_qubits, cfg.n_qubits if measure else 0)
    layer = 0
    for _ in range(cfg.n_uploads):
        if layer >= n_layers:
            break
        _encode(qc, window, cfg, params)
        for _ in range(cfg.entangler_layers):
            if layer >= n_layers:
                break
            _entangle(qc, layer, cfg, params)
            layer += 1
    if measure:
        qc.measure(range(cfg.n_qubits), range(cfg.n_qubits))
    return qc


def snapshot_layers(cfg: RFQRCConfig) -> list[int]:
    """N_v snapshot boundaries, uniformly spaced, always including the
    final layer."""
    L = cfg.n_uploads * cfg.entangler_layers
    nv = min(cfg.n_virtual_nodes, L)
    return sorted(set(int(round(L * (i + 1) / nv)) for i in range(nv)))


# -------------------------------------------------------------- features
def _features_from_probs(probs: dict, cfg: RFQRCConfig) -> np.ndarray:
    if cfg.readout == "full_probs":
        return probs_to_features(probs, cfg.n_qubits)
    feats = counts_to_features(probs, cfg.n_qubits)   # [<Z_i>, <Z_i Z_i+1>]
    if cfg.readout == "local_z":
        return feats[:cfg.n_qubits]
    if cfg.readout == "local_zz":
        return feats
    raise ValueError(f"unknown readout {cfg.readout!r}")


def step_features_exact(window: np.ndarray, cfg: RFQRCConfig,
                        params: dict) -> np.ndarray:
    """Exact path: statevector snapshots at the N_v layer boundaries."""
    from qiskit.quantum_info import Statevector
    feats = []
    for nl in snapshot_layers(cfg):
        qc = build_step_circuit(window, cfg, params, n_layers=nl)
        sv = Statevector(qc)
        feats.append(_features_from_probs(sv.probabilities_dict(), cfg))
    return np.concatenate(feats)


def step_features_shots(window: np.ndarray, cfg: RFQRCConfig, params: dict,
                        backend, shots: int, seed: int) -> np.ndarray:
    """Shots path: N_v truncated circuits, terminal measurement each
    (default realisation; reuse fusion is Stage 7)."""
    circs = [build_step_circuit(window, cfg, params, n_layers=nl,
                                measure=True)
             for nl in snapshot_layers(cfg)]
    result = backend.run(circs, shots=shots, seed_simulator=seed).result()
    feats = []
    for i in range(len(circs)):
        counts = result.get_counts(i)
        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()}
        feats.append(_features_from_probs(probs, cfg))
    return np.concatenate(feats)


# ------------------------------------------------------------- reservoir
def apply_leak(raw: np.ndarray, eps: float) -> np.ndarray:
    """r_k = (1-eps) r_{k-1} + eps raw_k, elementwise, r_{-1} = 0."""
    if not 0.0 < eps <= 1.0:
        raise ValueError("leak_eps must be in (0, 1]")
    r = np.zeros_like(raw)
    prev = np.zeros(raw.shape[1])
    for k in range(raw.shape[0]):
        prev = (1.0 - eps) * prev + eps * raw[k]
        r[k] = prev
    return r


def denoise_features(F: np.ndarray, spec: str) -> np.ndarray:
    """'svd:<k>' low-rank truncation | 'savgol:<w>,<p>' per-feature filter
    along time | 'none'."""
    if spec == "none":
        return F
    if spec.startswith("svd:"):
        k = int(spec.split(":")[1])
        U, s, Vt = np.linalg.svd(F, full_matrices=False)
        s[k:] = 0.0
        return (U * s) @ Vt
    if spec.startswith("savgol:"):
        from scipy.signal import savgol_filter
        w, p = (int(v) for v in spec.split(":")[1].split(","))
        return savgol_filter(F, window_length=w, polyorder=p, axis=0)
    raise ValueError(f"unknown denoise spec {spec!r}")


def run_reservoir(inputs: np.ndarray, cfg: RFQRCConfig,
                  washout: int = 20, verbose: bool = False) -> np.ndarray:
    """Full pipeline: per-step features -> optional denoise -> leak ->
    washout. `inputs` has shape (T, n_channels) scaled to [0, 1] with
    TRAIN statistics (caller's responsibility -- leakage discipline).

    Returns the reservoir state matrix of shape (T - washout, F).
    The T step circuits are mutually independent (embarrassingly
    parallel; batchable through the stage-5 compiler in Stage 7)."""
    inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
    if inputs.ndim == 2 and inputs.shape[0] == 1:
        inputs = inputs.T
    T, n_ch = inputs.shape
    params = make_entangler_params(cfg)

    backend = None
    if cfg.shots is not None:
        from qiskit_aer import AerSimulator
        backend = AerSimulator(seed_simulator=cfg.seed)

    raw = []
    for k in range(T):
        lo = max(0, k - cfg.window_m + 1)
        window = np.zeros((cfg.window_m, n_ch))
        window[:k - lo + 1] = inputs[lo:k + 1][::-1]   # window[0] = current
        if cfg.shots is None:
            raw.append(step_features_exact(window, cfg, params))
        else:
            raw.append(step_features_shots(window, cfg, params, backend,
                                           cfg.shots, cfg.seed + k))
        if verbose and (k + 1) % 200 == 0:
            print(f"  reservoir step {k + 1}/{T}")
    raw = np.array(raw)
    raw = denoise_features(raw, cfg.denoise)
    r = apply_leak(raw, cfg.leak_eps)
    return r[washout:]
