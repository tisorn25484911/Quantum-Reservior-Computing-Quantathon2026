"""P7: baseline battery -- correctness + the G5 no-future rule for every recursive model."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.models import autoregression as AR  # noqa: E402
from qrc_single_time_series.models import nvar as NV  # noqa: E402
from qrc_single_time_series.models import esn as ESN  # noqa: E402
from qrc_single_time_series.models import qlstm as QV  # noqa: E402
from qrc_single_time_series.models import lstm as LSTM  # noqa: E402
from qrc_single_time_series.models import gru as GRU  # noqa: E402
from qrc_single_time_series.models import statistical as STAT  # noqa: E402
from qrc_single_time_series.quantum.hamiltonians import fc_tfi, haar_reservoir  # noqa: E402
from qrc_single_time_series.quantum.exact_qrc import ExactQRC  # noqa: E402
from qrc_single_time_series.evaluation.autonomous import autonomous_from_series  # noqa: E402
from qrc_single_time_series.evaluation.metrics import effective_rank, nmse_variance  # noqa: E402


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 60, n)
    return 0.5 + 0.35 * np.sin(t) + 0.05 * rng.standard_normal(n)


def _ar2(n=2000, phi=(0.5, -0.3), c=0.2, seed=1):
    """A properly innovation-driven AR(2) so the coefficients are identifiable."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = c + phi[0] * y[t - 1] + phi[1] * y[t - 2] + 0.1 * rng.standard_normal()
    return y


# ---------------------------------------------------------------- correctness
def test_ridge_ar_recovers_coefficients():
    y = _ar2(phi=(0.5, -0.3), c=0.2)
    ar = AR.fit(y, 1600, p=2, lam=0.0)
    assert ar.W[0] == pytest.approx(0.5, abs=0.05)
    assert ar.W[1] == pytest.approx(-0.3, abs=0.05)
    assert ar.b == pytest.approx(0.2, abs=0.05)


def test_nvar_beats_mean_onestep():
    y = _series()
    nv = NV.fit(y, 300, k_lag=3, stride=1)
    pred = nv.predict_onestep(y)
    off = nv.span + 1
    nm = nmse_variance(y[300:-1], pred[300 - off: len(y) - 1 - off])
    assert nm < 0.5


def test_esn_size_matched_reaches_target():
    y = _series()
    # under a scalar drive the ESN state effective rank saturates around 2, so 2.0
    # is a reachable G7 target; the matching table is recorded as evidence.
    sm = ESN.size_matched(y, 300, target_eff_rank=2.0, seed=2)
    assert sm.eff_rank >= 2.0 - 1e-9
    assert isinstance(sm.match_table, list) and len(sm.match_table) >= 1


def test_windowed_vqc_learns():
    y = _series()
    vq = QV.WindowedVQC(window=5, n_qubits=3, layers=2, seed=0).fit(y, 300, maxiter=40)
    # trained circuit beats the constant (scaled-variance) predictor on train rows
    assert vq.train_mse < 0.9


def test_arima_fits_and_forecasts_finite():
    y = _series()
    m = STAT.fit(y, 300)
    st = m.warm(y[:300])
    st, yhat = m.step(st, y[299])
    assert np.isfinite(yhat)


# --------------------------------------------------------- G7 Haar control
def test_haar_control_finite_and_structureless():
    y = _series()
    haar = haar_reservoir(4, seed=3)
    qrc_h = ExactQRC(haar, V=5, tau=1.0)
    Xh = qrc_h.features(y[:200], check_budget=False)
    assert np.isfinite(Xh).all()
    # the Haar control genuinely differs from the structured TFI reservoir
    qrc_s = ExactQRC(fc_tfi(4, seed=3), V=5, tau=1.0)
    Xs = qrc_s.features(y[:200], check_budget=False)
    assert not np.allclose(Xh, Xs)


# --------------------------------------------------------- G5: no future access
def _factories():
    y = _series()
    fac = {
        "ridge_ar": lambda: AR.fit(y, 300, p=6),
        "nvar": lambda: NV.fit(y, 300, k_lag=3, stride=1),
        "esn": lambda: ESN.ESN(30, 0.9, 0.6, 0.5, 1).fit(y, 300),
        "windowed_vqc": lambda: QV.WindowedVQC(window=5, seed=0).fit(y, 300, maxiter=20),
    }
    return y, fac


@pytest.mark.parametrize("name", ["ridge_ar", "nvar", "esn", "windowed_vqc"])
def test_recursive_baseline_ignores_future(name):
    """G5: replacing everything at/after the origin with NaN must not change the rollout."""
    y, fac = _factories()
    model = fac[name]()
    origin, n_steps = 300, 15
    clean = autonomous_from_series(model, y, origin, n_steps)["predictions"]
    poisoned = y.copy()
    poisoned[origin:] = np.nan
    dirty = autonomous_from_series(model, poisoned, origin, n_steps)["predictions"]
    assert np.array_equal(clean, dirty)
    assert np.isfinite(clean).all()


@pytest.mark.skipif(not LSTM.AVAILABLE, reason="torch optional extra not installed")
def test_neural_baselines_available():
    y = _series()
    for mod in (LSTM, GRU):
        m = mod.fit(y, 300, window=8, hidden=8, max_epochs=30)
        assert np.isfinite(m.predict_onestep(y)).all()
