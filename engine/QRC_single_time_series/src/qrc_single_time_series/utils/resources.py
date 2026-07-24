"""resources.py -- record the environment for every run (spec s31)."""
from __future__ import annotations
import json, platform, subprocess, sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PKGS = ["numpy", "scipy", "pandas", "statsmodels", "matplotlib", "qiskit",
        "qiskit-aer", "pennylane", "pyyaml", "pytest", "torch"]


def package_versions() -> dict:
    out = {}
    for p in PKGS:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def git_commit(root: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root or Path.cwd()),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def run_environment() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": package_versions(), "git_commit": git_commit()}


if __name__ == "__main__":
    print(json.dumps(run_environment(), indent=2))
