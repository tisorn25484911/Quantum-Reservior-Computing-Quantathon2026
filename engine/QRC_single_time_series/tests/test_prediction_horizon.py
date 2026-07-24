"""P4: H_skill vs baseline, H_reliable from failure flag, integrated horizon."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qrc_single_time_series.evaluation import prediction_horizon as H  # noqa: E402
from qrc_single_time_series.evaluation import accumulated_error as A  # noqa: E402


def test_h_skill_when_model_beats_then_loses():
    base = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    model = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 0.4])   # loses sustainedly at idx 2
    assert H.h_skill(model, base, K=3) == 2


def test_h_skill_full_when_always_better():
    base = np.ones(10)
    model = 0.3 * np.ones(10)
    assert H.h_skill(model, base, K=3) == 10


def test_h_reliable_from_failure_step():
    assert H.h_reliable(first_failure_step=7, n_steps=20) == 7
    assert H.h_reliable(first_failure_step=None, n_steps=20) == 20


def test_integrated_horizon_from_error_curves():
    # build CNRMSE from a real error sequence and combine all three components.
    sigma = 1.0
    yt = np.zeros(12)
    yp = np.concatenate([np.zeros(6), np.full(6, 3.0)])   # diverges from step 6
    ne = A.instantaneous_ne(yt, yp, sigma)
    cn_model = A.cnrmse(yt, yp, sigma)
    cn_base = np.full(12, 1.2)
    he = H.h_error(ne, eps=1.0, K=3)
    hs = H.h_skill(cn_model, cn_base, K=3)
    out = H.h_effective(he, hs, H.h_reliable(None, 12))
    assert out["H_effective"] == min(he, hs)
    assert he == 6                                        # first 6 steps under eps
