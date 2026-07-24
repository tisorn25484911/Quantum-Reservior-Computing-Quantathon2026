"""manifests.py -- read/verify data/manifests/<name>.json."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = ROOT / "data" / "manifests"
RAW = ROOT / "data" / "raw"


def load_manifest(name: str) -> dict:
    return json.loads((MANIFESTS / f"{name}.json").read_text())


def verify_checksum(name: str) -> bool:
    m = load_manifest(name)
    got = hashlib.sha256((RAW / m["filename"]).read_bytes()).hexdigest()
    return got == m["sha256"]
