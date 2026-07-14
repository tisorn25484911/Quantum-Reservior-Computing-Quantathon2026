"""Anchors for the ent_scale (digital JDt analogue) knob and the NumPy
full-feature shot sampler used by the shots curve."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from qrc_core import WindowedReservoir
from qrc_qiskit import QiskitReservoir


def test_ent_scale_one_is_the_legacy_reservoir():
    """s=1 must reproduce the original CZ-ring reservoir bit for bit."""
    a = WindowedReservoir(seed=7)
    b = WindowedReservoir(seed=7, ent_scale=1.0)
    w = np.random.default_rng(0).uniform(0, 1, 24)
    assert np.allclose(a.W, b.W)
    assert np.max(np.abs(a.features(w) - b.features(w))) < 1e-14


def test_ent_scale_zero_is_identity_entangler():
    r = WindowedReservoir(seed=7, ent_scale=0.0)
    assert np.max(np.abs(r.W - np.eye(2 ** r.n))) < 1e-12


def test_ent_scale_preserves_seed_draws():
    """gains and w_angles must be identical at every s (draw order fixed)."""
    a = WindowedReservoir(seed=7, ent_scale=0.1)
    b = WindowedReservoir(seed=7, ent_scale=4.0)
    assert np.allclose(a.gains, b.gains)
    assert np.allclose(a.w_angles, b.w_angles)


def test_validation_gate_extends_to_scaled_entangler():
    """Qiskit exact must match NumPy at s != 1 too."""
    for s in (0.25, 2.5):
        res = WindowedReservoir(seed=7, ent_scale=s)
        qr = QiskitReservoir(res=res)
        w = np.random.default_rng(3).uniform(0, 1, 24)
        assert np.max(np.abs(qr.exact_features(w) - res.features(w))) < 1e-10


def test_numpy_sampler_converges_to_exact():
    r = WindowedReservoir(seed=7)
    w = np.random.default_rng(4).uniform(0, 1, 24)
    exact = r.features(w)
    est = r.sampled_features(w, shots=200_000,
                             rng=np.random.default_rng(9))
    assert est.shape == (20,)
    assert np.max(np.abs(est - exact)) < 0.02


def test_numpy_sampler_probs_fast_path_matches_direct():
    r = WindowedReservoir(seed=7)
    w = np.random.default_rng(5).uniform(0, 1, 24)
    probs = r.basis_probs(w)
    a = r.sampled_features(w, 4096, np.random.default_rng(1))
    b = r.sampled_features(w, 4096, np.random.default_rng(1), probs=probs)
    assert np.allclose(a, b)
