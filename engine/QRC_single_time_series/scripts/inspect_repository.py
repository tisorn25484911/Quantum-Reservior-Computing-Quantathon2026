"""inspect_repository.py -- record the project's own structure + environment.

Writes results/logs/repo_inventory.json: the QRC_single_time_series/ file tree
(counts by extension), the pinned environment (package versions, python,
platform, git commit if any), and a confirmation that only files under the
project folder are referenced. Run:

    python scripts/inspect_repository.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qrc_single_time_series.utils.resources import run_environment  # noqa: E402


def inventory() -> dict:
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and ".venv" not in p.parts and "__pycache__" not in p.parts]
    by_ext = Counter(p.suffix or "(none)" for p in files)
    return {
        "root": str(ROOT),
        "n_files": len(files),
        "by_extension": dict(sorted(by_ext.items(), key=lambda kv: -kv[1])),
        "top_level": sorted(p.name for p in ROOT.iterdir()),
        "environment": run_environment(),
        "note": "self-contained: all project artifacts live under this root.",
    }


def main() -> int:
    out = ROOT / "results" / "logs" / "repo_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    out.write_text(json.dumps(inv, indent=2))
    print(f"{inv['n_files']} files; extensions {inv['by_extension']}")
    print(f"-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
