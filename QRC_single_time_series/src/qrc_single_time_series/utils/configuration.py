"""configuration.py -- frozen config -> SHA-256 artifact key (G4)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import yaml


def config_hash(obj: Any) -> str:
    """SHA-256 over canonical JSON of a config dict/dataclass-dict."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())
