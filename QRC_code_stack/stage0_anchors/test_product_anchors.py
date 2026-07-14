"""Stage-8 anchors: every product module's self-test must pass.

Each stage8 module carries its own __main__ self-test with hand-checked
expectations (firewall refusal, isotonic monotonicity, fatigue caps,
rank-collapse detection, quantum-demotion path, ...). Running them via
subprocess keeps a single source of truth."""

import subprocess
import sys
from pathlib import Path

import pytest

STAGE8 = Path(__file__).resolve().parents[1] / "stage8_product"

MODULES = [
    "data_plane/connectors.py",
    "data_plane/qc.py",
    "data_plane/transforms.py",
    "serving/heads.py",
    "serving/conformal.py",
    "serving/shadow_battery.py",
    "decision/thresholds.py",
    "decision/alerts.py",
    "decision/regime_router.py",
    "decision/stacker.py",
    "governance/monitors.py",
    "governance/promotion_gates.py",
    "governance/model_cards.py",
]


@pytest.mark.parametrize("mod", MODULES)
def test_module_selftest(mod):
    result = subprocess.run([sys.executable, str(STAGE8 / mod)],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (mod, result.stdout[-1500:],
                                    result.stderr[-1500:])
