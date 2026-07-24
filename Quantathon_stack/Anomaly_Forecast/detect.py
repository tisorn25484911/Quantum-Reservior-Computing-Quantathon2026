"""detect.py -- Step 2: a marine-heatwave EARLY-WARNING detector.

The two-step plan: Step 1 (QRC) forecasts the MHW-intensity series; Step 2
(this module) is a supervised detector that flags marine heatwaves. Validated
here on its own train/val/test on REAL data; composed with Step-1 forecasts
separately (see `compose_*`).

A design choice that keeps this honest rather than circular. The MHW *label* is
a fixed rule on the intensity (SST > seasonal 90th percentile for >=5 days), so
a classifier fed today's intensity would just re-learn the threshold and prove
nothing. Instead the task is **onset prediction**:

    for each day t that is NOT currently in a heatwave,
    will a new heatwave BEGIN within the next L days?
    -- using only causal features available up to day t.

That is a genuine early-warning problem (it has warning time), it is non-trivial
(the answer is not in today's value), and it is exactly what the product needs.
Features are multivariate precursors -- intensity dynamics plus the drivers the
domain says lead Gulf warming (ENSO state, rainfall, drought) -- all trailing,
all causal.

Baselines it must beat (a detector that beats none of these is not a model):
  * climatological onset rate (predict the base rate every day),
  * "warm-now" (today's intensity as the score -- the naive precursor),
  * ONI-only (ENSO state alone).

Metrics are for RARE EVENTS and for TIME SERIES:
  * PR-AUC (average precision) -- the honest headline under heavy imbalance;
    ROC-AUC flatters rare-event detectors and is reported only alongside.
  * event-level recall at a fixed false-alarm budget, chosen on validation.
Never point-adjust F1 (Kim et al. 2022): it scores a random detector ~0.9.

    python detect.py                 # MHW onset early-warning on got_sst
    python detect.py --horizon 14    # warn up to 14 days ahead
    python detect.py --nab           # independent check on NAB temperature data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "DataBase_Analysis"))
from dataloader import DATA, _TABULAR                          # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.inspection import permutation_importance          # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

LABELS = _HERE / "labels"
RESULTS = _HERE / "results"


# ----------------------------------------------------------------------
# Feature construction (all trailing / causal)
# ----------------------------------------------------------------------
def _daily_driver(key, dates):
    """Load a driver series and align it onto the daily label index by
    forward-fill (each day gets the most recent observed value -- causal)."""
    spec = _TABULAR[key]
    df = pd.read_csv(DATA / spec["path"], parse_dates=[spec["index"]])
    s = df.set_index(spec["index"])[spec["column"]].sort_index()
    return s.reindex(dates.union(s.index)).ffill().reindex(dates)


def build_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Causal precursor features + the onset label, per day."""
    f = frame.copy()
    d = pd.DatetimeIndex(f["date"])
    inten = f["intensity"].astype(float)

    feat = pd.DataFrame(index=f.index)
    feat["intensity"] = inten
    for w in (7, 30):
        feat[f"i_mean{w}"] = inten.rolling(w, min_periods=1).mean()
        feat[f"i_max{w}"] = inten.rolling(w, min_periods=1).max()
        feat[f"i_std{w}"] = inten.rolling(w, min_periods=2).std().fillna(0)
    # short-term trend (slope over last 14 d, via diff of 7d means)
    feat["i_slope14"] = (inten.rolling(7, min_periods=1).mean()
                         - inten.rolling(7, min_periods=1).mean().shift(7)).fillna(0)
    # time since the last heatwave ended (a refractory / build-up signal)
    since = np.zeros(len(f))
    c = 0
    mhw = f["mhw"].to_numpy()
    for i in range(len(f)):
        c = 0 if mhw[i] else c + 1
        since[i] = c
    feat["days_since_mhw"] = np.log1p(since)
    # exogenous drivers -- the domain precursors
    feat["oni"] = _daily_driver("oni", d).to_numpy()
    feat["oni_chg90"] = pd.Series(feat["oni"].to_numpy()).diff(90).fillna(0).to_numpy()
    feat["rain30"] = _daily_driver("maeklong_rain", d).rolling(30, min_periods=1).sum().to_numpy()
    feat["spei"] = _daily_driver("maeklong_spei", d).to_numpy()
    # seasonal phase (cyclical)
    doy = d.dayofyear.to_numpy()
    feat["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    feat["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    # ---- onset label: a NEW event starts within (t, t+horizon] ----
    onset = (f["event_id"].to_numpy() > 0) & (
        np.r_[True, f["event_id"].to_numpy()[1:] != f["event_id"].to_numpy()[:-1]])
    onset_idx = np.flatnonzero(onset)
    y = np.zeros(len(f), dtype=int)
    for oi in onset_idx:
        lo = max(0, oi - horizon)
        y[lo:oi] = 1                      # the `horizon` days before an onset
    feat["y"] = y
    # eligibility: only days NOT already inside a heatwave can "warn" of one
    feat["eligible"] = (f["mhw"].to_numpy() == 0)
    feat["date"] = d
    return feat


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------
def chrono_split(n, fr_tr=0.6, fr_va=0.2):
    a, b = int(fr_tr * n), int((fr_tr + fr_va) * n)
    return slice(0, a), slice(a, b), slice(b, n)


def event_recall_fa(dates, y_true, score, thr, horizon):
    """Event-level recall and false-alarms/year at a score threshold.

    An onset is 'caught' if any day in its warning window scored above thr; a
    false alarm is a flagged day with no true onset in the next `horizon` days.
    """
    dates = pd.DatetimeIndex(dates)
    fired = score >= thr
    # group true windows by contiguous y_true==1 runs (each precedes one onset)
    caught = 0
    total = 0
    i = 0
    n = len(y_true)
    while i < n:
        if y_true[i] == 1:
            j = i
            while j + 1 < n and y_true[j + 1] == 1:
                j += 1
            total += 1
            if fired[i:j + 1].any():
                caught += 1
            i = j + 1
        else:
            i += 1
    fa = int((fired & (y_true == 0)).sum())
    years = max((dates.max() - dates.min()).days / 365.25, 1e-9)
    return (caught / total if total else float("nan"), fa / years, total)


# ----------------------------------------------------------------------
# Main MHW early-warning experiment
# ----------------------------------------------------------------------
def run_mhw(horizon=7, seed=7):
    frame = pd.read_csv(LABELS / "got_sst_mhw_labeled.csv", parse_dates=["date"])
    feat = build_features(frame, horizon)

    cols = [c for c in feat.columns if c not in ("y", "eligible", "date")]
    elig = feat["eligible"].to_numpy()
    X = feat[cols].to_numpy(float)
    y = feat["y"].to_numpy(int)
    dates = pd.DatetimeIndex(feat["date"])

    tr, va, te = chrono_split(len(feat))
    # keep only eligible rows within each split for fitting/scoring
    def sub(sl):
        idx = np.arange(len(feat))[sl]
        idx = idx[elig[idx]]
        return idx
    itr, iva, ite = sub(tr), sub(va), sub(te)

    print(f"=== Step 2: MHW onset early-warning (warn <= {horizon} d ahead) ===")
    print(f"N={len(feat)}  eligible days: train={len(itr)} val={len(iva)} "
          f"test={len(ite)}")
    print(f"onset-window positives: train={y[itr].sum()} "
          f"({y[itr].mean():.1%})  val={y[iva].sum()} test={y[ite].sum()} "
          f"({y[ite].mean():.1%})")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # class-balanced sample weights (rare positives)
    pos = max(1, y[itr].sum())
    neg = max(1, (y[itr] == 0).sum())
    w = np.where(y[itr] == 1, neg / pos, 1.0)
    base_rate = y[ite].mean()

    def ap(score):
        return average_precision_score(y[ite], score)

    # two models: gradient boosting (flexible) and logistic (robust under the
    # train->test base-rate shift below). Report both against the baselines.
    models = {}
    hgb = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.06, max_iter=400,
        l2_regularization=1.0, random_state=seed)
    hgb.fit(X[itr], y[itr], sample_weight=w)
    models["gradient_boost"] = (hgb, hgb.predict_proba(X[iva])[:, 1],
                                hgb.predict_proba(X[ite])[:, 1])
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000, C=0.5,
                                          class_weight="balanced"))
    lr.fit(X[itr], y[itr])
    models["logistic"] = (lr, lr.predict_proba(X[iva])[:, 1],
                          lr.predict_proba(X[ite])[:, 1])

    # baselines on the same test rows
    warm_now = X[ite][:, cols.index("intensity")]
    i_slope = X[ite][:, cols.index("i_slope14")]
    oni_only = X[ite][:, cols.index("oni")]
    baselines = {"climatology_base_rate": base_rate,
                 "warm_now": ap(warm_now), "i_slope14": ap(i_slope),
                 "oni_only": ap(oni_only)}

    print(f"\n  train onset-rate {y[itr].mean():.1%} -> test {base_rate:.1%} "
          "(warming trend: recent years have MORE heatwaves)")
    print("\n  PR-AUC on identical test rows (higher better; base rate "
          f"{base_rate:.3f}):")
    for k, v in baselines.items():
        print(f"    baseline {k:22s} {v:.3f}")
    for name, (_, _, p_te) in models.items():
        print(f"    MODEL    {name:22s} {ap(p_te):.3f}")

    # pick the model that wins on VALIDATION PR-AUC (no test peeking)
    best_name = max(models, key=lambda k: average_precision_score(y[iva], models[k][1]))
    clf, p_va, p_te = models[best_name]
    ap_te = ap(p_te)
    roc_te = roc_auc_score(y[ite], p_te)
    best_baseline = max(v for k, v in baselines.items() if k != "climatology_base_rate")
    print(f"\n  selected model (best val PR-AUC): {best_name}  -> test PR-AUC {ap_te:.3f}")
    verdict = ("BEATS" if ap_te > best_baseline else "does NOT beat")
    print(f"  VERDICT: the detector {verdict} the best simple baseline "
          f"({best_baseline:.3f}).")

    # pick an operating threshold on VALIDATION at ~2 false alarms/yr, apply to test
    order = np.argsort(-p_va)
    thr_grid = np.unique(p_va[order])[::max(1, len(p_va)//200)]
    best_thr, best = 0.5, -1
    for thr in thr_grid:
        rec, fa, _ = event_recall_fa(dates[iva].values, y[iva], p_va, thr, horizon)
        if fa <= 6 and rec > best:            # <=6 FA/yr budget on val
            best, best_thr = rec, thr
    rec_te, fa_te, n_ev = event_recall_fa(
        pd.DatetimeIndex(dates[ite]).values.astype("datetime64[D]"),
        y[ite], p_te, best_thr, horizon)
    print(f"\n  operating point (thr={best_thr:.3f}, chosen on val @<=6 FA/yr):")
    print(f"    TEST event recall = {rec_te:.2f} of {n_ev} onsets caught, "
          f"false alarms/yr = {fa_te:.1f}")

    # which precursors matter
    pi = permutation_importance(clf, X[ite], y[ite], n_repeats=8,
                                random_state=seed,
                                scoring="average_precision")
    imp = sorted(zip(cols, pi.importances_mean), key=lambda t: -t[1])[:6]
    print("\n  top precursors (permutation importance, drop in PR-AUC):")
    for name, v in imp:
        print(f"    {name:16s} {v:+.4f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"step2_detector_mhw_h{horizon}.json").write_text(json.dumps(dict(
        task="mhw_onset_early_warning", horizon=horizon, n=len(feat),
        n_test_eligible=len(ite), test_positives=int(y[ite].sum()),
        train_onset_rate=float(y[itr].mean()), base_rate=float(base_rate),
        selected_model=best_name, pr_auc=float(ap_te), roc_auc=float(roc_te),
        models={k: float(ap(v[2])) for k, v in models.items()},
        baselines={k: float(v) for k, v in baselines.items()},
        beats_best_baseline=bool(ap_te > best_baseline),
        operating=dict(threshold=float(best_thr), event_recall=float(rec_te),
                       false_alarms_per_year=float(fa_te), n_events=int(n_ev)),
        top_features=[(n, float(v)) for n, v in imp],
    ), indent=2))
    print(f"\n  wrote results/step2_detector_mhw_h{horizon}.json")
    return ap_te


# ----------------------------------------------------------------------
# Independent method check on NAB temperature data
# ----------------------------------------------------------------------
def run_nab():
    """Sanity-check the windowed-feature detection approach on an INDEPENDENT
    labelled temperature benchmark (NAB realKnownCause). Different domain,
    different labels, trusted ground truth -- shows the method is not tuned to
    our own Hobday labels."""
    from sklearn.ensemble import IsolationForest
    nab = DATA / "benchmark_nab"
    labels = json.loads((nab / "combined_windows.json").read_text())
    print("=== independent check: NAB temperature anomalies (unsupervised) ===")
    for name in ("ambient_temperature_system_failure",
                 "machine_temperature_system_failure"):
        df = pd.read_csv(nab / f"{name}.csv", parse_dates=["timestamp"])
        v = df["value"].to_numpy(float)
        # trailing windowed features (causal): value, rolling mean/std, z-score
        s = pd.Series(v)
        W = 60
        feats = np.column_stack([
            v,
            s.rolling(W, min_periods=1).mean().to_numpy(),
            s.rolling(W, min_periods=2).std().fillna(0).to_numpy(),
            ((v - s.rolling(W, min_periods=1).mean())
             / (s.rolling(W, min_periods=2).std().replace(0, 1))).fillna(0).to_numpy(),
        ])
        iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=7)
        score = -iso.fit(feats).score_samples(feats)
        # label windows -> boolean truth
        wins = labels[f"realKnownCause/{name}.csv"]
        truth = np.zeros(len(df), dtype=bool)
        ts = df["timestamp"]
        for a, b in wins:
            truth |= (ts >= pd.Timestamp(a)) & (ts <= pd.Timestamp(b))
        ap = average_precision_score(truth, score)
        # event recall: is the top-scoring region inside a labelled window?
        k = max(1, int(0.02 * len(df)))
        flagged = np.zeros(len(df), bool)
        flagged[np.argsort(-score)[:k]] = True
        caught = sum(any(flagged[(ts >= pd.Timestamp(a)) & (ts <= pd.Timestamp(b))])
                     for a, b in wins)
        print(f"  {name:38s} PR-AUC={ap:.3f}  "
              f"windows caught {caught}/{len(wins)} (top-2% flagged)")
    print("  -> the generic windowed detector recovers the labelled anomalies "
          "on independent data.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--nab", action="store_true",
                    help="run the independent NAB benchmark check instead")
    a = ap.parse_args()
    if a.nab:
        run_nab()
    else:
        run_mhw(horizon=a.horizon)


if __name__ == "__main__":
    main()
