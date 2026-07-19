"""config.py -- paths, provenance vocabulary, and import wiring.

The app sits beside two existing code trees rather than duplicating them:
``QRC_code_stack/stage1_numpy_core`` supplies the reference reservoir (the
project's semantic definition under repository law R2) and
``Quantathon_stack/DataBase_Analysis`` supplies dataset access. Both are plain
directories of modules rather than installed packages, so their paths go on
``sys.path`` here, once, at import time.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = APP_DIR.parent
PROJECT_ROOT = WEBAPP_DIR.parent

QRC_CORE_DIR = PROJECT_ROOT / "QRC_code_stack" / "stage1_numpy_core"
DB_ANALYSIS_DIR = PROJECT_ROOT / "Quantathon_stack" / "DataBase_Analysis"
EVALUATION_DIR = PROJECT_ROOT / "Quantathon_stack" / "Main_run_Evaluation"
DATA_DIR = PROJECT_ROOT / "Quantathon_stack" / "Data"

RESULTS_DIR = WEBAPP_DIR / "results"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

for _p in (QRC_CORE_DIR, DB_ANALYSIS_DIR, EVALUATION_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Project-wide seed (repository law R4: printed configuration, fixed seed).
SEED = 7

# ----------------------------------------------------------------------
# Provenance vocabulary
# ----------------------------------------------------------------------
# Every number this app shows is a simulation. The tier says what the *input*
# was, and the UI is required to display it next to any result, because the
# difference between "measured on a real record" and "measured on a synthetic
# stand-in" is the difference between a finding and a demo. The stage-8
# pipeline labels its artifacts the same way; this mirrors that vocabulary.
TIER_LABELS = {
    "real": "real record",
    "surrogate": "synthetic surrogate",
    "chaotic": "simulated chaotic system",
}
TIER_NOTE = {
    "real": "Observed data from a public archive. Forecast skill here is a "
            "genuine out-of-sample result on a real record.",
    "surrogate": "Synthetic stand-in generated for this project. Useful for "
                 "exercising the pipeline; NOT evidence about real-world "
                 "performance.",
    "chaotic": "Numerically integrated dynamical system with a known Lyapunov "
               "exponent. A controlled testbed, not an application result.",
}

# The claim discipline the whole app is written to enforce.
CLAIM_DISCLAIMER = (
    "All figures are classical simulation of a small quantum reservoir "
    "(exact density-matrix, no quantum hardware). At these system sizes there "
    "is no quantum speed-up to demonstrate; the defensible claim is parity "
    "with a size-matched classical baseline, which is reported alongside "
    "every result."
)
