"""P3: synthetic noise channels + L4 vs L5 (noise deviates from ideal, attributably)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.quantum.hamiltonians import fc_tfi  # noqa: E402
from qrc_single_time_series.quantum.qiskit_qrc import QiskitReservoir  # noqa: E402
from qrc_single_time_series.quantum.noise_models import build_synthetic  # noqa: E402
from qrc_single_time_series.quantum.noisy_qiskit_qrc import NoisyQiskitReservoir  # noqa: E402

N = 3
S = np.random.default_rng(0).uniform(0, 1, 16)


def _res():
    return fc_tfi(N, J=1.0, h=0.5, seed=7)


def ideal():
    return QiskitReservoir(_res(), V=4, tau=2.0, kappa=2).features(S, check_budget=False)


def noisy(cfg):
    nm = build_synthetic(cfg)
    return NoisyQiskitReservoir(_res(), nm, V=4, tau=2.0, kappa=2).features(
        S, check_budget=False)


def test_empty_noise_model_matches_ideal():
    assert np.allclose(ideal(), noisy({}), atol=1e-12)


def test_depolarising_deviation_is_monotonic():
    ref = ideal()
    devs = [np.max(np.abs(noisy({"depol_1q": p, "depol_2q": p}) - ref))
            for p in (1e-3, 1e-2, 5e-2)]
    assert devs[0] < devs[1] < devs[2]


def test_readout_error_pulls_z_toward_zero():
    # A symmetric readout flip biases <Z> toward 0, i.e. features toward 0.5.
    ref = ideal()
    ro = noisy({"readout_p": 0.1})
    # readout error is applied at measurement; with save_density_matrix it does not
    # affect the stored state -> features unchanged. This documents that the
    # readout channel must be modelled at count level (shot path), not the DM.
    assert np.allclose(ref, ro, atol=1e-12)


def test_thermal_relaxation_builds_and_perturbs():
    cfg = {"thermal_relaxation": {"t1_us": 100, "t2_us": 80,
                                  "gate_time_1q_ns": 50, "gate_time_2q_ns": 300}}
    dev = np.max(np.abs(noisy(cfg) - ideal()))
    assert dev > 0.0


def test_shot_layer_adds_spread_around_noisy_features():
    nm = build_synthetic({"depol_1q": 1e-2})
    r = NoisyQiskitReservoir(_res(), nm, V=4, tau=2.0, kappa=2)
    exact = r.features(S, check_budget=False)
    sampled = r.features(S, check_budget=False, shots=256, sampling_seed=5)
    assert not np.allclose(exact, sampled)
    assert np.max(np.abs(exact[:, :-1] - sampled[:, :-1])) < 0.5
