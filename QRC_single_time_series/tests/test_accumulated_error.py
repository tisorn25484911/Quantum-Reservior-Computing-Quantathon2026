"""P4: CNRMSE/CRMSE/CNMAE on hand-computed error sequences, train-only sigma."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qrc_single_time_series.evaluation import accumulated_error as A  # noqa: E402


def test_sigma_train_and_zero_guard():
    assert A.sigma_train([1.0, -1.0, 1.0, -1.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        A.sigma_train([3.0, 3.0, 3.0])


def test_instantaneous_ne():
    ne = A.instantaneous_ne([0, 0, 0], [0, 2, -4], sigma=2.0)
    assert np.allclose(ne, [0, 1, 2])


def test_crmse_is_running_rms():
    yt = np.zeros(4)
    yp = np.array([0.0, 2.0, 0.0, 0.0])       # single spike of size 2
    crmse = A.crmse(yt, yp)
    assert crmse[0] == pytest.approx(0.0)
    assert crmse[1] == pytest.approx(np.sqrt(4 / 2))     # sqrt(mean[0,4])
    assert crmse[2] == pytest.approx(np.sqrt(4 / 3))
    assert crmse[3] < crmse[2]                            # running mean dips


def test_cnrmse_and_cnmae_normalise_by_sigma():
    yt = np.zeros(3)
    yp = np.array([1.0, 1.0, 1.0])
    assert np.allclose(A.cnrmse(yt, yp, sigma=2.0), 0.5)
    assert np.allclose(A.cnmae(yt, yp, sigma=2.0), 0.5)
