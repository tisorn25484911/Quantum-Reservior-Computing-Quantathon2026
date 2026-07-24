"""engine_integration.py -- proof that the two repos are combined.

The sister repo (rigorous, test-anchored QRC engine) is vendored under
`engine/` via git subtree. This script is the integration anchor: it drives
**their** validated exact Fujii-Nakajima reservoir with **our** real
Gulf-of-Thailand data, through our loader, and produces a forecast. If this
runs, the applied pipeline can stand on the engine's validated core instead of
our lighter `qrc_core.py`.

Run from the repo root's venv:

    PYTHONPATH=engine/QRC_single_time_series/src:Quantathon_stack/DataBase_Analysis \\
        .venv/bin/python Quantathon_stack/Anomaly_Forecast/engine_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "engine" / "QRC_single_time_series" / "src"))
sys.path.insert(0, str(_REPO / "Quantathon_stack" / "DataBase_Analysis"))

from qrc_single_time_series.quantum.exact_qrc import ExactQRC      # noqa: E402
from qrc_single_time_series.models.readout import fit as ridge_fit  # noqa: E402
from dataloader import load                                        # noqa: E402


def nmse(t, p):
    return float(np.mean((t - p) ** 2) / np.var(t))


def main():
    # our real target: the Gulf-of-Thailand marine-heatwave intensity
    x = np.asarray(load("got_sst_mhwi").x, float)
    x = x[np.isfinite(x)][-4000:]
    train_end = 2800
    lo, hi = x[:train_end].min(), x[:train_end].max()      # train-span-only scaler
    u = np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    # their validated exact FN reservoir
    qrc = ExactQRC.from_config(dict(
        n_qubits=5, V_virtual_nodes=10, tau=2.0, J=1.0, h=0.5,
        reservoir_seed=7, observables="z_local"))
    feats = qrc.features(u)
    y = u[1:]
    tr, te = slice(100, train_end), slice(train_end, len(feats) - 1)
    readout = ridge_fit(feats[tr], y[tr])
    pred = readout.predict(feats[te])

    e = nmse(y[te], pred)
    print("=== engine integration anchor ===")
    print(f"engine: their exact FN reservoir (feature_dim={qrc.feature_dim})")
    print(f"data  : our got_sst_mhwi via our loader (N={len(x)})")
    print(f"1-step held-out test NMSE = {e:.4f}")
    assert e < 0.2, "engine forecast on our data is implausibly poor -- check wiring"
    print("PASS: our data pipeline drives the sister repo's validated engine.")


if __name__ == "__main__":
    main()
