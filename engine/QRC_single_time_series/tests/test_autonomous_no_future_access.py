"""P4/G5: rollout reads only the warmup slice; the future cannot change outputs."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qrc_single_time_series.evaluation.autonomous import (  # noqa: E402
    rollout, autonomous_from_series, make_policy)
from qrc_single_time_series.models.recursive_forecaster import AR1  # noqa: E402


def test_future_mutation_does_not_change_outputs():
    series = np.arange(30.0)
    a = autonomous_from_series(AR1(0.7), series, 12, 8)
    s2 = series.copy()
    s2[12:] = np.nan                      # garbage future
    b = autonomous_from_series(AR1(0.7), s2, 12, 8)
    assert np.array_equal(a["predictions"], b["predictions"])


def test_future_garbage_values_identical():
    series = np.sin(np.arange(40.0) / 3)
    a = autonomous_from_series(AR1(0.9), series, 20, 10)
    s2 = series.copy()
    s2[20:] = 1e9
    b = autonomous_from_series(AR1(0.9), s2, 20, 10)
    assert np.array_equal(a["predictions"], b["predictions"])


def test_rollout_never_indexes_beyond_history_length():
    # A model that peeks at history[len] would raise; rollout must not pass it.
    hist = np.arange(10.0)
    r = rollout(AR1(0.5), hist, 5)
    assert len(r["predictions"]) == 5


def test_policy_telemetry_populated():
    series = np.arange(10.0)
    pol = make_policy("hard_clip", lo=0.0, hi=5.0)
    r = rollout(AR1(1.2), series, 8, policy=pol)   # phi>1 -> grows -> clips
    t = r["telemetry"]
    assert t["n"] == 8 and t["n_clipped"] > 0 and t["first_clip_t"] is not None
