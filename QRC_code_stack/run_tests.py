"""run_tests.py -- anchors-first driver for the whole stack.

Runs each stage's suite in ladder order and exits 0 iff every populated
stage is green (Part IX, R6: promotion gates, not vibes). Stages whose
directories hold only a PLAN.md are reported SKIP (planned) and do not
fail the run; a stage that exists but fails turns the run red.

Usage:
    python run_tests.py              all stages in order
    python run_tests.py --stage 5    one stage
    python run_tests.py --quick      stage suites in quick mode where supported

Status: Phase-1 scaffold. Stage suites were copied verbatim from their
source projects and still carry old-layout import paths; Phase 2 repairs
them, after which this driver's exit code becomes meaningful. Until then
it reports per-stage status honestly and exits non-zero.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# stage number -> (directory, [commands], pythonpath_rel, notes).
# commands None => plan-only. Each command runs in the stage directory;
# pythonpath_rel entries (relative to repo root) are prepended to PYTHONPATH.
STAGES: dict[int, tuple[str, list[list[str]] | None, list[str], str]] = {
    0: ("stage0_anchors", [[sys.executable, "-m", "pytest", "-q", "."]], [],
        "core anchor suite (37 tests; conftest.py supplies paths)"),
    1: ("stage1_numpy_core", [[sys.executable, "qrc_core.py"]], [],
        "reference validation suite (Fujii-Nakajima reservoir)"),
    2: ("stage2_circuits_exact",
        [[sys.executable, "qrc_qiskit.py"],
         [sys.executable, "cirq_qrc_minimal.py"]],
        ["stage1_numpy_core/enso_app", "stage1_numpy_core"],
        "circuit features == NumPy features at seed 7 (qiskit gate 1e-15)"
        " + cirq port; pennylane port run manually (slow: 14 opt iters)"),
    3: ("stage3_noisy_simple",
        [[sys.executable, "noise_models.py"],
         [sys.executable, "exp_tfim1d_noisy.py", "--check"]],
        ["stage5_qubit_reuse"],
        "noise anchors + degradation gates (full grid: run without --check)"),
    4: ("stage4_hamiltonian_battery",
        [[sys.executable, "exp_family_scan.py", "--check"]],
        ["stage5_qubit_reuse"],
        "RMT anchors 0.386/0.531 + disorder crossover gate"),
    5: ("stage5_qubit_reuse",
        [[sys.executable, "run_stage5_tests.py", "--quick"]], [],
        "reuse anchors + equivalence gate"),
    6: ("stage6_rfqrc",
        [[sys.executable, "mfe_model.py"],
         [sys.executable, "rfqrc_baselines.py"],
         [sys.executable, "exp_lorenz63_check.py", "--check"],
         [sys.executable, "exp_mfe_extremes.py", "--check"]],
        ["stage5_qubit_reuse"],
        "RF-QRC anchors + pilot gates (pre-registered study: --full only)"),
    7: ("stage7_integration",
        [[sys.executable, "exp_rfqrc_reuse.py", "--check"]],
        ["stage5_qubit_reuse", "stage6_rfqrc"],
        "batch-and-compress: TVD gate + width-vs-horizon structural claim"
        " (solar study still plan-level)"),
    8: ("stage8_product",
        [[sys.executable, "exp_product_demo.py", "--check"]],
        ["stage5_qubit_reuse", "stage6_rfqrc"],
        "six-layer walk-forward demo on SURROGATE data; Phase 8-9 "
        "acceptance gates (module self-tests live in stage 0)"),
}


def run_stage(k: int, quick: bool) -> str:
    import os

    dirname, cmds, pp_rel, note = STAGES[k]
    stage_dir = ROOT / dirname
    if cmds is None:
        return "SKIP"
    if not stage_dir.exists():
        return "MISSING"
    env = os.environ.copy()
    if pp_rel:
        extra = os.pathsep.join(str(ROOT / p) for p in pp_rel)
        env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
    for cmd in cmds:
        result = subprocess.run(cmd, cwd=stage_dir, env=env)
        if result.returncode != 0:
            return "FAIL"
    return "PASS"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    stages = [args.stage] if args.stage is not None else sorted(STAGES)
    statuses: dict[int, str] = {}
    for k in stages:
        print(f"\n=== stage {k}: {STAGES[k][0]} ({STAGES[k][3]}) ===")
        statuses[k] = run_stage(k, args.quick)
        print(f"stage {k}: {statuses[k]}")

    print("\n=== summary ===")
    for k in stages:
        print(f"  stage {k}: {statuses[k]:8s} {STAGES[k][0]}")
    bad = [k for k, s in statuses.items() if s in ("FAIL", "MISSING")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
