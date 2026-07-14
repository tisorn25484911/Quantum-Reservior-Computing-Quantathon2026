"""Stage-6 anchors: the Part X anchor table, each with an exact expected
value. Paths provided by conftest.py (stage6 added below since only this
file needs it)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage6_rfqrc"))

from qiskit.quantum_info import Statevector

from mfe_model import run_anchors as mfe_anchors
from rfqrc_baselines import run_anchors as baseline_anchors
from rfqrc_metrics import (event_scores_vs_offset, nmse, ph_ladder,
                           predictability_horizon)
from rfqrc_reservoir import (RFQRCConfig, apply_leak, build_step_circuit,
                             make_entangler_params, probs_to_features,
                             run_reservoir, snapshot_layers,
                             step_features_exact)


# ------------------------------------------------------------------ leak
def test_leak_impulse_response_exact():
    """r_d = eps (1-eps)^d exactly (Part X Proposition -> ESP)."""
    eps = 0.3
    raw = np.zeros((10, 1))
    raw[0] = 1.0
    r = apply_leak(raw, eps).ravel()
    expect = eps * (1 - eps) ** np.arange(10)
    assert np.allclose(r, expect, atol=1e-15)


def test_leak_memory_function_iid_input():
    """For iid features phi_k, cov(r_k, phi_{k-d}) = eps (1-eps)^d var(phi):
    the unrolled Volterra kernel (Part X eq. on additivity)."""
    eps, T = 0.25, 200_000
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(T, 1))
    r = apply_leak(phi, eps).ravel()
    phi = phi.ravel()
    for d in range(5):
        c = np.mean(r[d + 10:] * phi[10:len(phi) - d])
        expect = eps * (1 - eps) ** d
        assert abs(c - expect) < 6e-3, (d, c, expect)


def test_leak_rejects_invalid_eps():
    with pytest.raises(ValueError):
        apply_leak(np.zeros((3, 1)), 0.0)


# ------------------------------------------------- virtual nodes / circuit
def _tiny_cfg(**kw):
    base = dict(n_qubits=2, n_uploads=2, entangler_layers=1,
                n_virtual_nodes=2, readout="full_probs", seed=7)
    base.update(kw)
    return RFQRCConfig(**base)


def test_snapshot_probabilities_match_independent_statevector():
    """Snapshot features = |<b|psi(t_i)>|^2 computed independently from
    the truncated circuit's statevector, with the logical-index
    convention (qubit 0 = MSB, stage-1 kron order)."""
    cfg = _tiny_cfg()
    params = make_entangler_params(cfg)
    window = np.array([[0.37]])
    feats = step_features_exact(window, cfg, params)
    boundaries = snapshot_layers(cfg)
    dim = 2 ** cfg.n_qubits
    for j, nl in enumerate(boundaries):
        qc = build_step_circuit(window, cfg, params, n_layers=nl)
        sv = np.asarray(Statevector(qc))
        # qiskit statevector index bit i = qubit i (little-endian);
        # logical index has qubit 0 as MSB -> permute independently
        probs = np.zeros(dim)
        for idx in range(dim):
            bits = format(idx, f"0{cfg.n_qubits}b")[::-1]  # bit i = qubit i
            probs[int(bits, 2)] = abs(sv[idx]) ** 2
        assert np.allclose(feats[j * dim:(j + 1) * dim], probs, atol=1e-12)


def test_snapshot_vs_truncation_tvd():
    """Snapshot path and independently built truncated circuits give
    identical distributions, TVD < 1e-12."""
    cfg = _tiny_cfg(n_qubits=3, entangler_layers=2)
    params = make_entangler_params(cfg)
    window = np.array([[0.61]])
    feats = step_features_exact(window, cfg, params)
    dim = 2 ** cfg.n_qubits
    for j, nl in enumerate(snapshot_layers(cfg)):
        qc = build_step_circuit(window, cfg, params, n_layers=nl,
                                measure=True)
        sv = Statevector(qc.remove_final_measurements(inplace=False))
        p2 = probs_to_features(sv.probabilities_dict(), cfg.n_qubits)
        tvd = 0.5 * np.sum(np.abs(feats[j * dim:(j + 1) * dim] - p2))
        assert tvd < 1e-12


def test_endianness_known_basis_state():
    """probs_to_features on |q0=1, q1=0> puts the mass at logical index
    2 (qubit 0 = MSB): the single-reversal convention, hand-checked."""
    # qiskit little-endian key for q0=1, q1=0 is '01'
    probs = {"01": 1.0}
    f = probs_to_features(probs, 2)
    assert f[2] == 1.0 and f.sum() == 1.0


def test_haar_control_distinct_inputs_distinct_features():
    cfg = _tiny_cfg(entangler="haar_control")
    params = make_entangler_params(cfg)
    f1 = step_features_exact(np.array([[0.2]]), cfg, params)
    f2 = step_features_exact(np.array([[0.9]]), cfg, params)
    assert np.max(np.abs(f1 - f2)) > 1e-3


def test_reservoir_shapes_and_washout():
    cfg = _tiny_cfg(readout="local_zz")
    F = run_reservoir(np.linspace(0, 1, 12), cfg, washout=4)
    n_feats = (2 * cfg.n_qubits - 1) * len(snapshot_layers(cfg))
    assert F.shape == (8, n_feats)


# --------------------------------------------------------------- metrics
def test_ph_ladder_synthetic_ramp_hand_computed():
    """k_true constant 0.0, k_pred ramps linearly: with k_e = 0.1 and
    k_bar = 0, the error crosses 0.2 when k_pred > 0.02. Hand computation:
    with k_pred[j] = 0.001*j and dt_lt = 0.1, first failure at j = 21
    -> raw PH = 2.1 LT -> ladder snaps to 2.0 LT. Ladder terminates: a
    perfect prediction caps at the ladder start (10 LT)."""
    k_true = np.zeros(300)
    k_pred = 0.001 * np.arange(300)
    ph_raw = predictability_horizon(k_true, k_pred, k_e=0.1, dt_lt=0.1)
    assert abs(ph_raw - 2.1) < 1e-12
    assert ph_ladder(ph_raw) == 2.0
    ph_perfect = predictability_horizon(k_true, k_true, 0.1, dt_lt=0.1)
    assert ph_ladder(ph_perfect) == 10.0   # terminates at the start rung


def test_fscore_degenerate_event_free_window():
    """No events in truth or prediction: degenerate flag set, F1 = 0,
    no division error."""
    k = np.zeros((5, 40))
    out = event_scores_vs_offset(k, k, k_e=0.1, bin_edges_steps=[0, 20, 40])
    assert all(o["degenerate"] and o["f1"] == 0.0 for o in out)


def test_fscore_perfect_prediction():
    rng = np.random.default_rng(0)
    k = rng.uniform(0, 0.2, size=(20, 30))
    out = event_scores_vs_offset(k, k, k_e=0.1, bin_edges_steps=[0, 15, 30])
    assert all(o["f1"] == 1.0 for o in out if not o["degenerate"])


def test_nmse_mean_is_one():
    y = np.random.default_rng(1).normal(size=400)
    assert abs(nmse(y, np.full_like(y, y.mean())) - 1.0) < 1e-12


# --------------------------------------------- module-level anchor suites
def test_mfe_anchors():
    assert mfe_anchors(verbose=False)


def test_baseline_anchors():
    assert baseline_anchors(verbose=False)
