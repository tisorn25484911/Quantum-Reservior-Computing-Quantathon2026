"""P8: controlled-system integrators + Lyapunov audits (G8-1/2/3) + fidelity."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.dynamical_systems import integration as IN  # noqa: E402
from qrc_single_time_series.dynamical_systems import lorenz63 as LZ  # noqa: E402
from qrc_single_time_series.dynamical_systems import enso_ode as EN  # noqa: E402
from qrc_single_time_series.evaluation import dynamical_fidelity as DF  # noqa: E402
from qrc_single_time_series.evaluation.lyapunov import rosenstein  # noqa: E402


# ------------------------------------------------------------------ integrator
def test_rk4_matches_exponential_decay():
    # dx/dt = -x  ->  x(t) = x0 e^{-t}; RK4 is 4th-order accurate
    f = lambda s: -s
    traj = IN.integrate(f, [1.0], dt=0.01, n_steps=100, washout=0)
    assert traj[-1, 0] == pytest.approx(np.exp(-1.0), abs=1e-6)


def test_rk4_harmonic_energy_conserved():
    # d/dt [x, v] = [v, -x]; energy x^2+v^2 conserved to O(dt^4) over one period
    f = lambda s: np.array([s[1], -s[0]])
    traj = IN.integrate(f, [1.0, 0.0], dt=0.001, n_steps=6283, washout=0)
    e = traj[:, 0] ** 2 + traj[:, 1] ** 2
    assert np.allclose(e, 1.0, atol=1e-4)


# --------------------------------------------------------- G8-1 Lorenz beta audit
def test_lorenz_beta_audit_selects_8_3():
    ba = LZ.beta_audit(dt=0.01, n_steps=9000, washout=1500)
    assert ba["beta_8_3"]["lambda_max"] > 0.8         # ~0.9 chaotic
    assert ba["beta_3_8"]["lambda_max"] < 0.1         # non-chaotic
    assert ba["reproduces_0.906"] == "beta_8_3"


def test_lorenz_estimators_agree_in_sign():
    ba = LZ.beta_audit(dt=0.01, n_steps=9000, washout=1500)
    g = LZ.generate(n=6000, dt=0.02, washout=1500)
    lam_ros, _ = rosenstein(g["trajectory"][:, 0], m=5, lag=8, mean_period=15, max_t=40)
    # both estimators must be positive (chaotic); magnitudes need only share order
    assert ba["beta_8_3"]["lambda_max"] > 0 and lam_ros > 0


# --------------------------------------------------- G8-2/3 Vallis ENSO ODE audit
def test_vallis_bounded_and_chaotic():
    g = EN.generate(n=2000, dt=0.005, washout=3000)
    assert np.isfinite(g["trajectory"]).all()
    assert np.ptp(g["trajectory"][:, 1]) < 1e3       # bounded (no blow-up)
    a = EN.audit(dt=0.005, n_steps=15000, washout=3000)
    assert a["lambda_max"] > 1e-2 and a["chaotic"]   # computed in-repo, not 0.05-0.1


# ----------------------------------------------------------- dynamical fidelity
def test_fidelity_zero_for_identical_series():
    rng = np.random.default_rng(0)
    x = 0.5 + 0.3 * np.sin(np.linspace(0, 40, 500)) + 0.02 * rng.standard_normal(500)
    r = DF.fidelity_report(x, x)
    assert r["invariant_measure_L1"] == pytest.approx(0.0, abs=1e-9)
    assert r["spectral_L1"] == pytest.approx(0.0, abs=1e-9)
    assert r["acf_rms"] == pytest.approx(0.0, abs=1e-9)
    assert r["std_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_fidelity_positive_for_different_series():
    t = np.linspace(0, 40, 500)
    a = 0.5 + 0.3 * np.sin(t)
    b = 0.5 + 0.3 * np.sin(2.3 * t)                    # different spectrum
    r = DF.fidelity_report(a, b)
    assert r["spectral_L1"] > 0.1
    assert r["acf_rms"] > 0.05
