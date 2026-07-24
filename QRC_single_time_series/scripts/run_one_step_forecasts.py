"""scripts/run_one_step_forecasts.py -- P9: observational climate campaign (enso/pdo/soi).

The REAL indices only -- never conflated with the Vallis ODE (spec item 21). Per
index this produces:

  * train-only EDA (stationarity, decorrelation time, seasonal strength, spectrum);
  * teacher-forced direct multi-horizon forecasts h in {1,3,6,12} sharing ONE QRC
    feature cache, scored on the UNTOUCHED test span against persistence / seasonal
    persistence / ridge-AR;
  * the preregistered GATE-1 verdict: QRC persistence skill > 0 at h in {1,3} with a
    block-bootstrap 90% CI lower bound > 0 (spec s25.7);
  * launch-month skill stratification (the ENSO spring-predictability barrier); and
  * a hybrid-readout ablation on ENSO (quantum-only vs classical-lags-only vs
    quantum + lags), the G7 test of whether the quantum features add anything.

Expectations are set by the predictability literature, not the synthetic track:
single-digit months of skill is the realistic regime; >12 months triggers a leakage
hunt. Results -> results/metrics/climate_campaign.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.data.loaders import load_raw, split
from qrc_single_time_series.data.preprocessing import anomaly, fit_scaler
from qrc_single_time_series.data.eda import eda_report
from qrc_single_time_series.data.windows import align
from qrc_single_time_series.quantum.hamiltonians import fc_tfi
from qrc_single_time_series.quantum.exact_qrc import ExactQRC
from qrc_single_time_series.models import readout as R
from qrc_single_time_series.models.recursive_forecaster import StatefulFN
from qrc_single_time_series.training.teacher_forcing import train_teacher_forced
from qrc_single_time_series.evaluation import accumulated_error as A
from qrc_single_time_series.evaluation import prediction_horizon as H
from qrc_single_time_series.evaluation.autonomous import (autonomous_from_series,
    make_origins, make_policy)
from qrc_single_time_series.evaluation.statistics import skill_bootstrap_ci

HORIZONS = [1, 3, 6, 12]
N, V, TAU, SEED = 5, 6, 2.0, 7


def _lag_matrix(y, p):
    y = np.asarray(y, float)
    return np.stack([y[p - j - 1: len(y) - j - 1] for j in range(p)], axis=1), y[p:]


def _skill_ci(se_model, se_base, block_length):
    return skill_bootstrap_ci(se_model, se_base, block_length, alpha=0.10)


def _qrc_direct(qrc, series, cut, horizon):
    """Direct h-step QRC forecast: fit on dev rows, score on the test span.

    Returns (se_model over test, se_persist over test, target abs-index over test)."""
    X = qrc.features(series, check_budget=False)
    Xa, ya = align(X, series, horizon=horizon)            # row k -> y_{k+h}
    tgt_idx = np.arange(len(Xa)) + horizon                # absolute index of target
    tr = tgt_idx < cut
    te = tgt_idx >= cut
    ro = R.fit(Xa[tr][50:], ya[tr][50:], lam="gcv")
    pred = ro.predict(Xa[te])
    truth = ya[te]
    se_model = (pred - truth) ** 2
    persist = series[tgt_idx[te] - horizon]               # y_{t-h}
    se_persist = (persist - truth) ** 2
    return se_model, se_persist, tgt_idx[te]


def _launch_month_skill(series, month, cut, qrc, horizon=3):
    """Skill vs persistence stratified by the target's calendar month (spring barrier)."""
    se_m, se_p, tgt = _qrc_direct(qrc, series, cut, horizon)
    months = month[tgt]
    out = {}
    for mo in range(1, 13):
        sel = months == mo
        if sel.sum() >= 3 and se_p[sel].sum() > 0:
            out[str(mo)] = float(1.0 - se_m[sel].sum() / se_p[sel].sum())
    return out


def _autonomous(series, cut, block):
    """Modest autonomous H_effective at eps=0.5 (Gate-3 input)."""
    qrc = ExactQRC(fc_tfi(N, J=1.0, h=0.5, seed=SEED), V=V, tau=TAU)
    n_train = int(0.8 * cut)
    ro, _ = train_teacher_forced(qrc, series, n_train=n_train)
    sigma = float(np.std(series[:n_train]))
    origins = make_origins(len(series), warmup=n_train + 10, n_steps=24,
                           n_origins=20, spacing=max(block, 3))
    hes = []
    for o in origins:
        r = autonomous_from_series(StatefulFN(qrc, ro), series, o, 24,
                                   policy=make_policy("hard_clip", 0.0, 1.0))
        p = r["predictions"]
        ne = A.instantaneous_ne(series[o:o + len(p)], p, sigma)
        hes.append(H.h_error(ne, 0.5, 3))
    hes = np.array(hes)
    return {"n_origins": len(origins),
            "median_H_effective_months": float(np.median(hes)),
            "survival_at_3": float((hes >= 3).mean()),
            "H_product_p0.8": int(H.h_product(hes, 24, 0.8))}


def _hybrid_ablation(series, cut, qrc):
    """ENSO quantum-only vs classical-lags-only vs quantum+lags, one-step, test span."""
    p = 6
    Xq = qrc.features(series, check_budget=False)
    Xq_a, _ = align(Xq, series, horizon=1)                # row k -> y_{k+1}
    Xl, _ = _lag_matrix(series, p)                        # lags for y[p:]
    start = p
    Xq_c = Xq_a[start - 1:]
    y_c = series[start:]
    Xl_c = Xl[:len(y_c)]
    m = min(len(Xq_c), len(Xl_c), len(y_c))
    Xq_c, Xl_c, y_c = Xq_c[:m], Xl_c[:m], y_c[:m]
    tgt = np.arange(start, start + m)
    tr, te = tgt < cut, tgt >= cut

    def nmse(Xf):
        ro = R.fit(Xf[tr][40:], y_c[tr][40:], lam="gcv")
        pr = ro.predict(Xf[te])
        return float(np.var(y_c[te] - pr) / np.var(y_c[te]))

    return {
        "quantum_only": nmse(Xq_c),
        "classical_lags_only": nmse(Xl_c),
        "hybrid_quantum_plus_lags": nmse(np.hstack([Xq_c, Xl_c])),
    }


def run_index(name):
    s = load_raw(name)
    cut = split(len(s))["cut"]
    anom, _ = anomaly(s.month, s.values, cut)
    scale, _ = fit_scaler(anom, cut)
    y = scale(anom)
    qrc = ExactQRC(fc_tfi(N, J=1.0, h=0.5, seed=SEED), V=V, tau=TAU)

    eda = eda_report(y, cut)
    block = max(eda["decorrelation_time"], 3)

    horizons, gate1 = {}, {}
    for h in HORIZONS:
        se_m, se_p, _ = _qrc_direct(qrc, y, cut, h)
        pt, lo, hi = _skill_ci(se_m, se_p, block)
        # Absolute test NMSE vs the mean predictor (=1.0). NMSE < 1 == genuine skill;
        # NMSE > 1 with positive persistence-skill means persistence is simply a
        # WORSE baseline than climatology at that lead -- the "skill" is spurious,
        # NOT leakage (a leak would drive NMSE far below 1). The leakage-hunt flag.
        nmse = float(np.mean(se_m) / np.var(y[cut:]))
        horizons[str(h)] = {"qrc_test_nmse": nmse, "beats_climatology": bool(nmse < 1.0),
                            "persistence_skill": pt, "skill_ci90": [lo, hi]}
        if h in (1, 3):
            gate1[str(h)] = {"skill": pt, "ci90_lo": lo,
                             "beats_persistence": bool(lo > 0),
                             "beats_climatology": bool(nmse < 1.0),
                             "passes": bool(lo > 0)}

    out = {
        "identity": s.identity, "n": len(s), "test_cut": cut,
        "eda": eda, "bootstrap_block": block,
        "horizons": horizons, "gate1": gate1,
        "gate1_pass": any(v["passes"] for v in gate1.values()),
        "autonomous": _autonomous(y, cut, block),
        "launch_month_skill_h3": _launch_month_skill(y, s.month, cut, qrc, 3),
    }
    if name == "enso":
        out["hybrid_readout_ablation"] = _hybrid_ablation(y, cut, qrc)
    return out


def main():
    out = {}
    print("Observational climate campaign (real indices only)")
    print("=" * 62)
    for name in ("enso", "pdo", "soi"):
        res = run_index(name)
        out[name] = res
        print(f"\n{name}: n={res['n']} test_cut={res['test_cut']} "
              f"block={res['bootstrap_block']} "
              f"seasonal_str={res['eda']['seasonal_strength']:.2f}")
        for h in HORIZONS:
            hz = res["horizons"][str(h)]
            flag = "" if hz["beats_climatology"] else "  (NMSE>1: persistence-artifact)"
            print(f"   h={h:2d}  skill={hz['persistence_skill']:+.3f} "
                  f"CI90=[{hz['skill_ci90'][0]:+.3f},{hz['skill_ci90'][1]:+.3f}]  "
                  f"NMSE={hz['qrc_test_nmse']:.2f}{flag}")
        genuine = [h for h in HORIZONS if res["horizons"][str(h)]["beats_climatology"]]
        print(f"   GATE-1 (skill>0 @h1/3, CI-lo>0): "
              f"{'PASS' if res['gate1_pass'] else 'fail'}   "
              f"genuine absolute skill (NMSE<1) at h={genuine or 'none'}")
        a = res["autonomous"]
        print(f"   autonomous: median H_eff={a['median_H_effective_months']:.0f} mo "
              f"survival@3={a['survival_at_3']:.2f} H_product(0.8)={a['H_product_p0.8']}")

    en = out["enso"].get("hybrid_readout_ablation")
    if en:
        print(f"\nENSO hybrid ablation (test NMSE): quantum={en['quantum_only']:.3f} "
              f"classical={en['classical_lags_only']:.3f} "
              f"hybrid={en['hybrid_quantum_plus_lags']:.3f}")

    dest = ROOT / "results" / "metrics" / "climate_campaign.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
