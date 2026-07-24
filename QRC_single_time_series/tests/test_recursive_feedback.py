"""P4: recursive baselines feed their own predictions back through the one engine."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qrc_single_time_series.evaluation.autonomous import rollout  # noqa: E402
from qrc_single_time_series.models.recursive_forecaster import (  # noqa: E402
    AR1, EncodedRecursive)


def test_ar1_rollout_matches_closed_form():
    phi, y0, n = 0.6, 4.0, 10
    r = rollout(AR1(phi), [y0], n)
    expected = y0 * phi ** np.arange(1, n + 1)
    assert np.allclose(r["predictions"], expected)


def test_ar1_with_intercept_converges_to_fixed_point():
    phi, c = 0.5, 2.0
    r = rollout(AR1(phi, c), [0.0], 40)
    assert r["predictions"][-1] == pytest.approx(c / (1 - phi), abs=1e-6)


def test_feedback_identity_encoded_recursive():
    # encode/decode identity + readout that returns the decoded input -> pure
    # persistence recursion: every prediction equals the warmup's last value.
    enc = lambda u: u
    dec = lambda s: s
    feat = lambda s, st: (None, np.array([s]))
    read = lambda f: f[0]
    m = EncodedRecursive(enc, dec, feat, read)
    r = rollout(m, [3.0, 7.0], 5)
    assert np.allclose(r["predictions"], 7.0)
