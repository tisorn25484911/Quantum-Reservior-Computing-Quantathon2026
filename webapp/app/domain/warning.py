"""warning.py -- the combined marine-heatwave early-warning result for the webapp.

Reads the integrator harness's output (``Anomaly_Forecast/results``) and shapes
it for the /warning page: the Step-1 x Step-2 combination matrix, the supporting
numbers from each stage, and the three-path product view. No computation here --
this only surfaces results the scripts already wrote, so the page never disagrees
with the repository.
"""

from __future__ import annotations

import json

from ..config import ANOMALY_DIR

RESULTS = ANOMALY_DIR / "results"

# nicer labels for the raw registry keys
_FC = {"engine_qrc": "engine-QRC", "core_qrc": "core-QRC",
       "nvar": "NVAR", "persistence": "persistence"}
_DET = {"threshold": "threshold", "ensemble": "ensemble", "pca_t2": "PCA-T²"}
_TEAMMATE = {"engine_qrc", "pca_t2"}          # components from the sister repo


def _load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def matrix_rows():
    """Combination matrix, ranked by the actionable (<=7 d) window F1."""
    d = _load("combine_conclusion.json")
    if not d:
        return []
    rows = []
    for key, v in d.items():
        fc, det = key.split("|")
        rec = v.get("recall", [])
        rows.append({
            "forecaster": _FC.get(fc, fc),
            "detector": _DET.get(det, det),
            "f1_short": v.get("f1_1_7"),
            "f1_full": v.get("f1_1_14"),
            "rec3": rec[2] if len(rec) > 2 else None,
            "rec7": rec[6] if len(rec) > 6 else None,
            "rec14": rec[13] if len(rec) > 13 else None,
            "quantum": "qrc" in fc,
            "teammate": (fc in _TEAMMATE) or (det in _TEAMMATE),
        })
    rows.sort(key=lambda r: -(r["f1_short"] or 0))
    return rows


def summary():
    """Headline numbers for the page, pulled from each stage's own result file."""
    out = {"disclaimer": True}

    step1 = _load("engine_step1_got_sst_mhwi.json")
    if step1:
        m = step1.get("engine_qrc_mean", [])
        tax = step1.get("taxonomy", {})
        out["step1"] = {
            "mean_nmse": round(sum(m) / len(m), 3) if m else None,
            "esn": round(sum(step1["esn_mean"]) / len(step1["esn_mean"]), 3),
            "nvar": round(sum(step1["nvar"]) / len(step1["nvar"]), 3),
            "h_skill": tax.get("H_skill"),
            "beats_persistence": tax.get("H_beats_persistence"),
        }

    step2 = _load("step2_detector_mhw_h7.json")
    if step2:
        out["step2"] = {
            "pr_auc": round(step2.get("pr_auc", 0), 3),
            "base_rate": round(step2.get("base_rate", 0), 3),
            "lift": round(step2["pr_auc"] / step2["base_rate"], 1)
            if step2.get("base_rate") else None,
        }

    step3 = _load("step3_stochastic_xxz_hx.json")
    if step3:
        ens = step3.get("ensemble", {}).get("recall", [])
        mean = step3.get("mean_traj", {}).get("recall", [])
        out["step3"] = {
            "coverage": round(step3.get("mean_coverage", 0), 2),
            "ens_recall14": round(ens[13], 2) if len(ens) > 13 else None,
            "mean_recall14": round(mean[13], 2) if len(mean) > 13 else None,
        }

    rows = matrix_rows()
    if rows:
        best_q = next((r for r in rows if r["quantum"]), None)
        best_p = next((r for r in rows
                       if r["forecaster"] == "persistence"), None)
        out["best_quantum"] = best_q
        out["persistence"] = best_p
    return out


def paths():
    """The three-path warning product."""
    return [
        {"tag": "PRIMARY", "title": "Forecast → detect",
         "body": "engine-QRC forecast + calibrated ensemble threshold — the "
                 "days-ahead heatwave warning."},
        {"tag": "HEDGE", "title": "Precursor onset",
         "body": "ML detector on observed ENSO + rainfall + build-up. Warns "
                 "onset independent of the forecast — a backstop when the "
                 "forecast is weak."},
        {"tag": "RARE-EVENT", "title": "Compound co-exceedance",
         "body": "The teammate's PCA-T² subspace detector for rare "
                 "multi-driver ENSO events (ROC-AUC 0.96 at 4% base rate)."},
    ]
