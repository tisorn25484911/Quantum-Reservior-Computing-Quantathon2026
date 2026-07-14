"""Stage-5 anchors: the reuse suite's own anchor + equivalence section.

The suite is a single verified driver (run_stage5_tests.py, moved in
unchanged per Part IX); rather than fork its anchor code into a second
copy that can drift, this test runs it in --quick mode and requires
exit 0. Single source of truth."""

import subprocess
import sys
from pathlib import Path

STAGE5 = Path(__file__).resolve().parents[1] / "stage5_qubit_reuse"


def test_reuse_quick_suite_green():
    result = subprocess.run(
        [sys.executable, "run_stage5_tests.py", "--quick"],
        cwd=STAGE5, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
