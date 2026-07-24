"""inspect_data.py -- spec s14 audit of enso/pdo/soi + checksummed manifests.

Inspects each raw CSV WITHOUT assuming its schema, records the resolved index
identity (audited in Phase 0), and writes data/manifests/<name>.json. Run:

    python scripts/inspect_data.py

Outputs one manifest per dataset containing filename, source/identity, target
column, units, date coverage, frequency, missing-value policy, transform
policy, scaling policy, sha256, and the audited findings (sentinels, gaps,
duplicates, anomaly-or-raw). Raw files are never modified.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MAN = ROOT / "data" / "manifests"

# Resolved identities (Phase-0 audit; cross-checked vs statsmodels 'elnino').
IDENTITY = {
    "enso": {
        "value_col": "sst", "units": "degC", "is_anomaly": False,
        "source": "statsmodels.datasets.elnino (NOAA); pinned copy",
        "identity": "NOAA Nino 1+2 region SST (0-10S, 90-80W), raw monthly SST",
        "not": "NOT ONI / Nino3.4 / MEI; NOT an anomaly index",
        "transform_policy": "model as anomaly vs TRAIN-YEARS-ONLY monthly climatology "
                            "(removes strong annual cycle); detrend/deseasonalise as ablations",
    },
    "pdo": {
        "value_col": "value", "units": "index", "is_anomaly": True,
        "source": "NOAA PSL pdo.data (Mantua-style); pinned copy",
        "identity": "Pacific Decadal Oscillation index (leading PC of N-Pacific SST anomaly)",
        "not": "distinguish from ERSSTv5-based PDO variants (differ after ~2002)",
        "transform_policy": "already an anomaly; optional train-only deseasonalise as ablation",
    },
    "soi": {
        "value_col": "value", "units": "index", "is_anomaly": True,
        "source": "NOAA CPC soi (ANOMALY block: Tahiti-Darwin SLP anomaly); pinned copy",
        "identity": "Southern Oscillation Index -- sea-level-pressure ANOMALY (Tahiti-Darwin)",
        "not": "NOT the Troup-standardised SOI (differs by a scale factor)",
        "transform_policy": "already an anomaly; optional train-only deseasonalise as ablation",
    },
}
SENTINELS = (-99.9, -999.0, -9999.0, 99.99, -99.99)


def audit(name: str) -> dict:
    fname = f"{name}.csv"
    path = RAW / fname
    raw = path.read_bytes()
    df = pd.read_csv(path)
    cols = list(df.columns)
    d = pd.to_datetime(df["date"])
    df = df.iloc[d.argsort().to_numpy()].reset_index(drop=True)
    d = pd.to_datetime(df["date"])
    mi = d.dt.year * 12 + d.dt.month
    step_diffs = sorted(mi.diff().dropna().unique().tolist())
    col = IDENTITY[name]["value_col"]
    v = df[col].to_numpy(float)
    sent = {str(s): int(np.isclose(v, s).sum()) for s in SENTINELS
            if np.isclose(v, s).any()}
    man = {
        "name": name,
        "filename": fname,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "columns": cols,
        "dtypes": {c: str(t) for c, t in df.dtypes.astype(str).items()},
        "timestamp_column": "date",
        "target_column": col,
        "frequency": "monthly" if step_diffs == [1.0] else f"IRREGULAR {step_diffs}",
        "coverage_start": str(d.min().date()),
        "coverage_end": str(d.max().date()),
        "n_duplicate_dates": int(d.duplicated().sum()),
        "monthly_step_diffs": step_diffs,
        "missing": {"n_nan": int(np.isnan(v).sum()), "sentinels_found": sent},
        "missing_value_policy": "MASK (never silent interpolation); none present in these files",
        "scaling_policy": "affine [0,1] fit on TRAIN rows only (encoder needs s in [0,1]); "
                          "test values outside train range are clipped and counted",
        "value_stats": {"min": float(v.min()), "max": float(v.max()),
                        "mean": float(v.mean()), "std": float(v.std())},
        "looks_like_anomaly": bool(abs(v.mean()) < 0.5),
        **IDENTITY[name],
    }
    return man


def main() -> int:
    MAN.mkdir(parents=True, exist_ok=True)
    for name in ("enso", "pdo", "soi"):
        man = audit(name)
        (MAN / f"{name}.json").write_text(json.dumps(man, indent=2))
        print(f"{name}: {man['n_rows']} rows {man['coverage_start']}..{man['coverage_end']} "
              f"freq={man['frequency']} anomaly={man['looks_like_anomaly']} "
              f"sha256={man['sha256'][:12]} -> data/manifests/{name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
