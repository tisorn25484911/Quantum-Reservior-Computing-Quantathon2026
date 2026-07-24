"""P6: rewind QRC (DM == Stinespring dilation, beats copy) + learned-map Lyapunov."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.quantum.hamiltonians import fc_tfi, nn_tfi  # noqa: E402
from qrc_single_time_series.quantum.exact_qrc import ExactQRC  # noqa: E402
from qrc_single_time_series.quantum.rewind_qrc import (  # noqa: E402
    RewindReservoir, copy_baseline_mse)
from qrc_single_time_series.dynamical_systems import mackey_glass as MG  # noqa: E402
from qrc_single_time_series.training.teacher_forcing import train_teacher_forced  # noqa: E402
from qrc_single_time_series.models import readout as R  # noqa: E402
from qrc_single_time_series.data.windows import align  # noqa: E402
from qrc_single_time_series.evaluation import metrics as MET  # noqa: E402
from qrc_single_time_series.evaluation import lyapunov as L  # noqa: E402


def _rewind(N=5, t_w=6):
    return RewindReservoir(nn_tfi(N, J=1.0, h=0.5, seed=7), t_w=t_w, tau=1.0)


@pytest.mark.parametrize("N,t_w", [(3, 4), (4, 5), (5, 4)])
def test_density_matrix_equals_stinespring_dilation(N, t_w):
    rw = _rewind(N, t_w)
    w = np.random.default_rng(0).uniform(0, 1, t_w)
    assert np.allclose(rw.window_features(w), rw.window_features_dilated(w), atol=1e-10)
    assert rw.n_dilated_qubits() == N + (t_w - 1)


def test_feature_dim():
    rw = _rewind(5, 8)
    assert rw.feature_dim == 5 * 8 + 1
    X = rw.features_series(np.random.default_rng(0).uniform(0, 1, 40))
    assert X.shape[1] == rw.feature_dim


def _beats_copy(series, lam="gcv"):
    s = np.asarray(series, float)
    rw = _rewind(5, 8)
    X = rw.features_series(s)
    y = s[rw.t_w:]
    X = X[:len(y)]
    ntr = int(0.7 * len(y))
    ro = R.fit(X[:ntr], y[:ntr], lam=lam)
    mse_model = np.mean((y[ntr:] - ro.predict(X[ntr:])) ** 2)
    mse_copy = copy_baseline_mse(s[rw.t_w + ntr - 1:])
    return mse_model, mse_copy


def test_rewind_beats_copy_mackey_glass():
    m, c = _beats_copy(MG.generate(17, n=1000, washout=800)["series"])
    assert m < c


def test_rewind_beats_copy_logistic():
    m, c = _beats_copy(L.logistic_series(r=3.9, n=1000))
    assert m < c


# --- learned-map Lyapunov machinery (validate on a known map first) ---------
def test_benettin_recovers_logistic_exponent():
    r = 3.9
    G = lambda x: np.array([r * x[0] * (1 - x[0])])
    lam = L.benettin_largest(G, np.array([0.4]), n_steps=3000, warmup=400)
    assert lam == pytest.approx(0.494, abs=0.05)


def test_benettin_agrees_with_rosenstein_on_logistic():
    lam_ben = L.benettin_largest(lambda x: np.array([3.9 * x[0] * (1 - x[0])]),
                                 np.array([0.4]), n_steps=3000, warmup=400)
    lam_ros, _ = L.rosenstein(L.logistic_series(r=3.9), m=3, lag=1,
                              mean_period=1, max_t=15)
    assert abs(lam_ben - lam_ros) < 0.05


def test_jacobian_matches_analytic():
    J = L.jacobian(lambda x: np.array([x[0] ** 2, x[0] * x[1]]), np.array([2.0, 3.0]))
    assert np.allclose(J, [[4.0, 0.0], [3.0, 2.0]], atol=1e-6)


def test_learned_delay_map_lyapunov_runs_on_interior():
    # The learned rewind F-map: build G and estimate its exponent on an interior
    # trajectory. This documents the (honest) finding that the stateless rewind map
    # does not reproduce the chaotic positive exponent -- recurrence captures it.
    s = MG.generate(17, n=900, washout=800)["series"]
    rw = _rewind(5, 6)
    X = rw.features_series(s)
    y = s[rw.t_w:]
    X = X[:len(y)]
    ntr = int(0.7 * len(y))
    ro = R.fit(X[:ntr], y[:ntr], lam="gcv")

    def predict(window):
        feat = np.concatenate([rw.window_features(window), [1.0]])
        return float(ro.predict(feat[None, :])[0])

    G = L.delay_map_from_prediction(predict)
    lam, n_int = L.benettin_largest_interior(G, s[300:300 + rw.t_w], n_steps=80)
    assert n_int >= 0                       # runs without error
    if n_int >= 10:
        assert np.isfinite(lam)


# --- FN recurrent vs rewind: both beat copy; recurrence carries the dynamics -
def test_fn_recurrent_and_rewind_both_beat_copy():
    d = MG.generate(17, n=1200, washout=800)["series"]
    ntr = 900
    qrc = ExactQRC(fc_tfi(5, J=1.0, h=0.5, seed=7), V=10, tau=4.0)
    _, info = train_teacher_forced(qrc, d, n_train=ntr)
    fn_nmse = info["nmse_dev_onestep"]

    rw = _rewind(5, 10)
    X = rw.features_series(d)
    y = d[rw.t_w:]
    X = X[:len(y)]
    ntr2 = ntr - rw.t_w
    ro = R.fit(X[:ntr2], y[:ntr2], lam="gcv")
    rw_nmse = MET.nmse_variance(y[ntr2:], ro.predict(X[ntr2:]))

    copy_nmse = copy_baseline_mse(d[ntr:]) / np.var(d[ntr:])
    assert fn_nmse < copy_nmse and rw_nmse < copy_nmse
