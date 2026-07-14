"""Path setup for the anchor suite (Phases 2-3 import repair).

The original six test files were copied verbatim from the climate project
(Part IX: "moves in unchanged"); their internal sys.path.insert points at
the old layout and is a harmless no-op here. This conftest provides the
real paths for every stage the suite touches.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for rel in (
    "stage1_numpy_core/enso_app",
    "stage2_circuits_exact",
    "stage3_noisy_simple",
    "stage4_hamiltonian_battery",
    "stage5_qubit_reuse",
):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)
