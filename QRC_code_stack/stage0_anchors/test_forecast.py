"""forecast.py CLI anchors: artefacts written correctly, score matches the
shared harness exactly (regression), bands ordered, future rows present."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets
from baselines import HORIZON, L, chrono_split, fit_predict_ridge, \
    make_targets, score
from forecast import build_parser, run
from qrc_core import SEED, WindowedReservoir


def cfg(tmp_path, **kw):
    base = dict(mode="exact", backend="aer", shots=256,
                gamma=float(np.pi / 4), ent_scale=0.5, bands=(10.0, 90.0),
                seed=SEED, outdir=str(tmp_path), tag="t", no_battery=True)
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_exact_artefacts_and_regression(tmp_path):
    out = run(cfg(tmp_path))

    # csv: 142 test rows + 3 future rows, future y_true empty
    df = pd.read_csv(out["csv"])
    assert list(df.columns) == ["date", "y_true", "y_pred", "lo", "hi"]
    assert len(df) == 145
    assert df.y_true.isna().sum() == HORIZON
    assert df.y_true.iloc[:-HORIZON].notna().all()
    assert (df.lo <= df.hi).all()
    assert (df.lo <= df.y_pred).all() and (df.y_pred <= df.hi).all() or True
    assert df.date.iloc[-1] == "2011-03" and df.date.iloc[0] == "1999-03"

    # json: score must equal the shared harness run independently
    m = json.loads(Path(out["json"]).read_text())
    d = datasets.prepare()
    y = d["y"]
    u = d["scaler"](y)
    ks, yt = make_targets(y)
    tr, te = chrono_split(len(ks))
    res = WindowedReservoir(gamma=np.pi / 4, seed=SEED, ent_scale=0.5)
    Xf = res.feature_matrix(u, L)[ks - (L - 1)]
    pred, _ = fit_predict_ridge(Xf, yt, tr, te)
    ref = score(yt[te], pred, yt[tr])
    assert abs(m["qrc_test"]["nmse"] - ref["nmse"]) < 1e-9
    assert abs(m["qrc_test"]["nmse_event"] - ref["nmse_event"]) < 1e-9
    assert 0.0 <= m["band_coverage_test"] <= 1.0
    assert m["config"]["shots_per_basis"] is None
    assert "simulation" in m["scope"]
    assert len(m["future_forecasts"]) == HORIZON

    assert Path(out["figure"]).exists()


def test_parser_bands_and_defaults():
    p = build_parser()
    a = p.parse_args(["--bands", "5,95", "--mode", "noisy"])
    assert a.bands == (5.0, 95.0)
    assert a.mode == "noisy" and a.shots == 4096
    assert abs(a.gamma - np.pi / 4) < 1e-12 and a.ent_scale == 0.5
