"""P4: H_error prefix rule (no re-validation), H_effective components, survival."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qrc_single_time_series.evaluation import prediction_horizon as H  # noqa: E402


def test_h_error_basic():
    ne = np.array([0.1, 0.2, 0.3, 0.9, 0.9, 0.9, 0.1])  # sustained breach at idx 3
    assert H.h_error(ne, eps=0.5, K=3) == 3


def test_dip_back_under_eps_does_not_revalidate():
    # breaches, K-run declares failure, then dips back under eps: horizon stays.
    ne = np.array([0.1, 0.6, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1])
    assert H.h_error(ne, eps=0.5, K=3) == 1        # only step 0 valid
    # a LATER clean stretch must not extend the horizon
    ne2 = np.concatenate([ne, np.zeros(20)])
    assert H.h_error(ne2, eps=0.5, K=3) == 1


def test_no_sustained_failure_gives_full_length():
    ne = np.array([0.6, 0.1, 0.6, 0.1, 0.6, 0.1])   # never 3-in-a-row
    assert H.h_error(ne, eps=0.5, K=3) == len(ne)


def test_h_effective_is_min_but_reports_all():
    out = H.h_effective(h_err=8, h_skl=5, h_rel=12)
    assert out["H_effective"] == 5
    assert out["H_error"] == 8 and out["H_skill"] == 5 and out["H_reliable"] == 12


def test_survival_and_product():
    hes = [2, 5, 5, 8, 3]
    hs, s = H.survival_curve(hes, max_h=8)
    assert s[0] == 1.0 and s[2] == pytest.approx(0.8) and s[7] == pytest.approx(0.2)
    assert H.h_product(hes, max_h=8, p=0.8) == 3
    assert H.h_product(hes, max_h=8, p=1.0) == 2
