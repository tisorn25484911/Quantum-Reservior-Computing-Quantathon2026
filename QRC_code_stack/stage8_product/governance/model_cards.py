"""model_cards.py -- model cards carrying the pre-registered criteria
and their outcomes, INCLUDING NULLS (Phase 11 trust artifact).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

CARD_DIR = Path(__file__).resolve().parent / ".." / "data" / "model_cards"


def write_card(model_version: str, config: dict, gates_verdict: dict,
               battery_table: dict, data_sources: list[dict],
               claims_scope: str = "simulation",
               notes: list[str] | None = None) -> Path:
    """Emit the card as markdown + a JSON sidecar. Every number must have
    arrived from a run (caller's obligation, enforced upstream by the
    provenance rule)."""
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    surrogate = any(d.get("kind") == "surrogate" for d in data_sources)
    md = [f"# Model card: {model_version}", f"Generated: {ts}",
          f"Claims scope: **{claims_scope}**"
          + ("  |  DATA INCLUDES SURROGATE - so do all numbers below"
             if surrogate else ""),
          "", "## Configuration", "```json",
          json.dumps(config, indent=1, default=str), "```",
          "", "## Pre-registered gates and outcomes (nulls included)",
          "```json", json.dumps(gates_verdict, indent=1, default=str),
          "```", "", "## Baseline battery (non-removable)", "```json",
          json.dumps(battery_table, indent=1, default=str), "```",
          "", "## Data sources"]
    for d in data_sources:
        md.append(f"- {d.get('source')} [{d.get('kind')}] "
                  f"licence: {d.get('licence', 'see registry')}")
    if notes:
        md += ["", "## Notes"] + [f"- {n}" for n in notes]
    md += ["", "Banned claims regardless of outcome: quantum advantage, "
           "speed-up, hardware implications from simulation."]
    p = CARD_DIR / f"{model_version}.md"
    p.write_text("\n".join(md))
    (CARD_DIR / f"{model_version}.json").write_text(json.dumps(
        {"model_version": model_version, "generated": ts,
         "config": config, "gates": gates_verdict,
         "battery": battery_table, "sources": data_sources,
         "scope": claims_scope}, default=str, sort_keys=True))
    return p


if __name__ == "__main__":
    p = write_card("selftest-0.0", {"n_qubits": 4},
                   {"passed": True, "gates": []},
                   {"persistence": {"nmse": 1.0}},
                   [{"source": "solar_surrogate", "kind": "surrogate"}],
                   notes=["self-test card"])
    print(f"wrote {p}")
    assert p.exists() and "SURROGATE" in p.read_text()
    print("model card self-test PASS")
