"""Reproducibility anchors (Phase 0 scope): manifests + split are deterministic."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from qrc_single_time_series.data import loaders
from qrc_single_time_series.utils.seeds import DEFAULT, SeedBundle


def test_seed_bundle_is_frozen_and_reproducible():
    a = SeedBundle(**DEFAULT.as_dict())
    assert a.as_dict() == DEFAULT.as_dict()
    assert (a.rng("reservoir_seed").random(3) == DEFAULT.rng("reservoir_seed").random(3)).all()


def test_test_fraction_is_frozen():
    assert loaders.TEST_FRACTION == 0.20
