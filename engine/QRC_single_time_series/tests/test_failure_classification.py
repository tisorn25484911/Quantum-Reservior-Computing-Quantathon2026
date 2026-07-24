"""P4: failure taxonomy on constructed pathologies (preregistered thresholds)."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qrc_single_time_series.evaluation import failure_modes as F  # noqa: E402
from qrc_single_time_series.utils.configuration import load_yaml  # noqa: E402

RNG = np.random.default_rng(0)
STATS = F.TrainStats.from_series(RNG.standard_normal(1000))    # mu~0, sigma~1
RULES = F.FailureRules()


def test_rules_load_from_preregistration():
    cfg = load_yaml(ROOT / "configs" / "preregistration.yaml")
    r = F.FailureRules.from_config(cfg)
    assert r.window == 24 and r.blowup_range_mult == 1.5 and r.var_explosion_mult == 4.0


def test_blowup_detected_with_first_step():
    traj = np.concatenate([RNG.standard_normal(20), [50.0], RNG.standard_normal(20)])
    labels = F.classify(traj, STATS, RULES)
    assert "blowup" in labels and labels["blowup"] == 20


def test_fixed_point_flat_line():
    traj = np.full(60, 0.3)
    labels = F.classify(traj, STATS, RULES)
    assert "fixed_point" in labels


def test_variance_explosion():
    traj = np.concatenate([RNG.standard_normal(24), 6 * RNG.standard_normal(24)])
    labels = F.classify(traj, STATS, RULES)
    assert "variance_explosion" in labels


def test_spurious_limit_cycle():
    t = np.arange(200)
    traj = 2.0 * np.sin(2 * np.pi * t / 12)
    labels = F.classify(traj, STATS, RULES)
    assert "spurious_cycle" in labels


def test_saturation_from_clip_budget():
    traj = RNG.standard_normal(50)
    labels = F.classify(traj, STATS, RULES, clip_fraction=0.5, clip_budget=0.2)
    assert "saturation" in labels
    labels2 = F.classify(traj, STATS, RULES, boundary_dwell_fraction=0.4,
                         boundary_budget=0.3)
    assert "saturation" in labels2


def test_stochastic_instability():
    # bounded ensemble mean, but seed-to-seed spread exceeds sigma
    base = np.sin(np.arange(60) / 5)
    ens = [base + RNG.standard_normal(60) * 3 for _ in range(8)]
    labels = F.classify(np.mean(ens, axis=0), STATS, RULES, ensemble=ens)
    assert "stochastic_instability" in labels


def test_stable_trajectory_trips_nothing_catastrophic():
    # in-range oscillation with train-like variance: no blowup / saturation
    traj = np.sin(np.arange(120) / 7)
    labels = F.classify(traj, STATS, RULES)
    assert "blowup" not in labels and "saturation" not in labels
    assert "variance_explosion" not in labels


def test_multiple_labels_allowed():
    traj = np.full(60, 0.0)                    # flat at the mean
    labels = F.classify(traj, STATS, RULES)
    assert {"fixed_point", "mean_collapse"} <= set(labels)
