"""Leakage anchor (Phase 0): every fitted transform saw train rows only."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from qrc_single_time_series.data import loaders, preprocessing
from qrc_single_time_series.data.leakage_checks import assert_scaler_train_only


def test_climatology_uses_train_months_only():
    s = loaders.load_raw("enso")
    n_train = loaders.split(len(s))["cut"]
    clim = preprocessing.monthly_climatology(s.month, s.values, n_train)
    s2 = s.values.copy(); s2[n_train:] += 1e6           # corrupt the future
    clim2 = preprocessing.monthly_climatology(s.month, s2, n_train)
    assert np.allclose(clim, clim2), "climatology leaked test months"


def test_scaler_train_only_all_series():
    for name in ("enso", "pdo", "soi"):
        s = loaders.load_raw(name)
        n_train = loaders.split(len(s))["cut"]
        y, _ = preprocessing.anomaly(s.month, s.values, n_train)
        scale, _ = preprocessing.fit_scaler(y, n_train)
        assert_scaler_train_only(scale, y, n_train)
