"""Qiskit-path anchors. The validation gate: the Qiskit circuit must
reproduce the NumPy reference EXACTLY (noiseless) before noise/shots
count for anything."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from qrc_core import WindowedReservoir
from qrc_qiskit import (QiskitReservoir, counts_to_features, get_backend,
                        run_counts)


def test_validation_gate_exact_features_match_reference():
    """Noiseless Aer/Statevector features == NumPy reference, same seed."""
    qr = QiskitReservoir(seed=7)
    rng = np.random.default_rng(11)
    for _ in range(3):
        w = rng.uniform(0, 1, size=24)
        assert np.max(np.abs(qr.exact_features(w) - qr.res.features(w))) < 1e-10


def test_validation_gate_holds_at_other_seeds_and_gamma():
    for seed in (7, 8):
        res = WindowedReservoir(gamma=np.pi / 4, seed=seed)
        qr = QiskitReservoir(res=res)
        w = np.random.default_rng(seed).uniform(0, 1, 24)
        assert np.max(np.abs(qr.exact_features(w) - res.features(w))) < 1e-10


def test_exact_feature_matrix_matches_reference_matrix():
    qr = QiskitReservoir(seed=7)
    s = np.random.default_rng(2).uniform(0, 1, 30)
    A = qr.exact_feature_matrix(s, 24)
    B = qr.res.feature_matrix(s, 24)
    assert A.shape == B.shape == (7, 20)
    assert np.max(np.abs(A - B)) < 1e-10


def test_counts_to_features_endianness():
    """Qiskit key '10000' = qubit 4 in |1> (little-endian). After the one
    reversal, reference qubit 4 must read -1 and qubits 0-3 read +1."""
    n = 5
    cz = {"10000": 1000}
    cx = {"00001": 1000}          # qubit 0 in |1> -> <X_0> block reads -1
    f = counts_to_features(cz, cx, n)
    z, x, zz = f[:n], f[n:2 * n], f[2 * n:]
    assert np.allclose(z, [1, 1, 1, 1, -1])
    assert np.allclose(x, [-1, 1, 1, 1, 1])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    expect = [(-1.0 if 4 in p else 1.0) for p in pairs]
    assert np.allclose(zz, expect)


def test_counts_to_features_averages():
    """50/50 mix of all-0 and all-1 -> Z block 0, ZZ block +1."""
    n = 5
    cz = {"00000": 500, "11111": 500}
    cx = {"00000": 1}
    f = counts_to_features(cz, cx, n)
    assert np.allclose(f[:n], 0.0)
    assert np.allclose(f[2 * n:], 1.0)


def test_sampled_noiseless_converges_to_exact():
    qr = QiskitReservoir(seed=7)
    w = np.random.default_rng(5).uniform(0, 1, 24)
    exact = qr.exact_features(w)
    be = get_backend("aer", noise=False)
    Xf, meta = qr.sampled_feature_matrix(w, 24, shots=8192, backend=be,
                                         verbose=False)
    assert meta["shots_per_basis"] == 8192
    # per-feature std <= 1/sqrt(S) ~ 0.011; 0.08 is a generous 7-sigma
    assert np.max(np.abs(Xf[0] - exact)) < 0.08


def test_transpiled_circuits_use_hardware_basis_only():
    qr = QiskitReservoir(seed=7)
    tz, tx, _ = qr.transpiled_templates(24)
    allowed = {"ecr", "rz", "sx", "x", "measure", "barrier"}
    for tc in (tz, tx):
        assert set(tc.count_ops()) <= allowed, tc.count_ops()
        assert tc.count_ops().get("ecr", 0) > 0     # entangler survived


def test_noisy_path_runs_and_degrades_gracefully():
    qr = QiskitReservoir(seed=7)
    w = np.random.default_rng(6).uniform(0, 1, 24)
    be = get_backend("aer", noise=True)
    Xf, _ = qr.sampled_feature_matrix(w, 24, shots=2048, backend=be,
                                      verbose=False)
    f = Xf[0]
    assert f.shape == (20,)
    assert np.all(np.isfinite(f)) and np.all(np.abs(f) <= 1.0)
    # noise must not scramble beyond recognition at these rates
    assert np.max(np.abs(f - qr.exact_features(w))) < 0.5


def test_ibm_hook_is_dormant_without_token(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    try:
        get_backend("ibm")
        assert False, "ibm path must refuse to run without a token"
    except RuntimeError as e:
        assert "IBM_QUANTUM_TOKEN" in str(e)
