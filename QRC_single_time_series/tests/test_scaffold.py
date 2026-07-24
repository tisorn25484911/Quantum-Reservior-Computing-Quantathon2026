"""Phase-0 tests: package imports, data manifests, and the frozen chronology.

These are real, passing tests (not skipped stubs) — they verify the Phase-0
exit criteria: importable package, checksummed manifests matching the raw files,
and a deterministic, non-overlapping, leak-free chronological split.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.data import loaders, manifests, preprocessing  # noqa: E402
from qrc_single_time_series.data.leakage_checks import assert_scaler_train_only  # noqa: E402

NAMES = ["enso", "pdo", "soi"]


def test_package_imports():
    import qrc_single_time_series as pkg
    assert pkg.__version__ == "0.0.0"


@pytest.mark.parametrize("name", NAMES)
def test_manifest_checksum_matches_raw(name):
    assert manifests.verify_checksum(name), f"{name}: manifest sha256 != raw file"


@pytest.mark.parametrize("name", NAMES)
def test_series_loads_and_is_monthly(name):
    s = loaders.load_raw(name)
    assert len(s) > 700 and s.month.min() == 1 and s.month.max() == 12
    assert not np.isnan(s.values).any()


@pytest.mark.parametrize("name", NAMES)
def test_split_is_deterministic_and_nonoverlapping(name):
    n = len(loaders.load_raw(name))
    a, b = loaders.split(n), loaders.split(n)
    assert np.array_equal(a["dev"], b["dev"]) and np.array_equal(a["test"], b["test"])
    assert set(a["dev"].tolist()).isdisjoint(a["test"].tolist())
    assert len(a["dev"]) + len(a["test"]) == n
    assert a["test"][0] == a["cut"] and a["dev"][-1] == a["cut"] - 1
    # final 20% within one row
    assert abs(len(a["test"]) / n - 0.20) < 1.5 / n


@pytest.mark.parametrize("name", NAMES)
def test_rolling_origin_stays_in_dev_and_is_causal(name):
    n = len(loaders.load_raw(name))
    cut = loaders.split(n)["cut"]
    for tr, va in loaders.rolling_origin_folds(cut):
        assert tr.max() < va.min()          # train strictly before val
        assert va.max() < cut               # never touches the test span


@pytest.mark.parametrize("name", NAMES)
def test_scaler_and_anomaly_are_train_only(name):
    s = loaders.load_raw(name)
    n_train = loaders.split(len(s))["cut"]
    y_anom, clim = preprocessing.anomaly(s.month, s.values, n_train)
    scale, _ = preprocessing.fit_scaler(y_anom, n_train)
    # perturbing the test tail must not change how train rows scale (no leak)
    assert_scaler_train_only(scale, y_anom, n_train)
