"""P4: inverse-transform -> rescale -> encode round trip is identity in-range."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qrc_single_time_series.data.preprocessing import (  # noqa: E402
    fit_scaler, invert_scaler, clip_report)


def test_decode_encode_identity_in_range():
    y = np.array([0.0, 2.0, 5.0, 9.0, 10.0])
    scale, bounds = fit_scaler(y, n_train=len(y))
    decode = invert_scaler(bounds)
    x = np.array([1.0, 3.5, 7.25])            # inside train range
    assert np.allclose(decode(scale(x)), x, atol=1e-12)


def test_encode_endpoints_map_to_unit_interval():
    y = np.array([-3.0, 0.0, 4.0])
    scale, bounds = fit_scaler(y, n_train=3)
    assert scale(-3.0) == pytest.approx(0.0)
    assert scale(4.0) == pytest.approx(1.0)


def test_out_of_range_is_clipped_and_reported():
    y = np.array([0.0, 10.0])
    scale, bounds = fit_scaler(y, n_train=2)
    decode = invert_scaler(bounds)
    x = np.array([-5.0, 15.0])                # outside -> clipped, NOT identity
    assert not np.allclose(decode(scale(x)), x)
    raw = (x - 0.0) / 10.0                     # pre-clip encoder values
    rep = clip_report(raw)
    assert rep["n_clipped"] == 2 and rep["frac_clipped"] == 1.0
