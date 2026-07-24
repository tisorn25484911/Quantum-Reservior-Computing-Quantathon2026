"""P9: observational-climate EDA + Gate-1 skill-bootstrap machinery."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.data import eda as EDA  # noqa: E402
from qrc_single_time_series.data.loaders import load_raw, split  # noqa: E402
from qrc_single_time_series.evaluation.statistics import skill_bootstrap_ci  # noqa: E402


def _seasonal_series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 0.5 + 0.3 * np.sin(2 * np.pi * t / 12) + 0.05 * rng.standard_normal(n)


# --------------------------------------------------------------------- EDA
def test_eda_detects_annual_period_and_season():
    y = _seasonal_series()
    rep = EDA.eda_report(y, n_train=480)
    assert 10.0 <= rep["dominant_period"] <= 14.0        # ~12-month cycle
    assert rep["seasonal_strength"] > 0.5                # clearly seasonal
    assert rep["decorrelation_time"] >= 1


def test_eda_decorrelation_time_reasonable_on_real_index():
    s = load_raw("soi")
    cut = split(len(s))["cut"]
    rep = EDA.eda_report(s.values.astype(float), cut)
    assert 1 <= rep["decorrelation_time"] <= 60


# ------------------------------------------------------- Gate-1 skill bootstrap
def test_skill_ci_perfect_model_is_one():
    n = 200
    se_base = np.abs(np.random.default_rng(0).standard_normal(n)) + 0.1
    se_model = np.zeros(n)
    pt, lo, hi = skill_bootstrap_ci(se_model, se_base, block_length=6)
    assert pt == pytest.approx(1.0)
    assert lo > 0.9


def test_skill_ci_equal_model_is_zero_and_ci_brackets_zero():
    n = 300
    se = np.abs(np.random.default_rng(1).standard_normal(n)) + 0.1
    pt, lo, hi = skill_bootstrap_ci(se, se.copy(), block_length=6)
    assert pt == pytest.approx(0.0, abs=1e-9)
    assert lo <= 0.0 <= hi


def test_skill_ci_worse_model_is_negative():
    n = 300
    rng = np.random.default_rng(2)
    se_base = np.abs(rng.standard_normal(n)) + 0.1
    se_model = se_base * 2.0                              # twice the error
    pt, lo, hi = skill_bootstrap_ci(se_model, se_base, block_length=6)
    assert pt < 0 and hi < 0.0
