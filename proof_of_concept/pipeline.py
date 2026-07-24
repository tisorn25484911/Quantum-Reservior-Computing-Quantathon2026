"""pipeline.py -- forecast-then-detect harness for the QRC proof of concept.

The reservoir does the honest thing it is good at -- temporal forecasting --
and the anomaly/compound logic sits DOWNSTREAM on its features. Two heads:

  1. a ridge POINT forecast  (for NMSE and the naive "threshold the level"
     detector -- which under-shoots extremes; that damping is a real failure
     mode we measure, not hide), and
  2. a logistic EXCEEDANCE head  that emits P(event) / P(compound event)
     directly, optimising the rare-event objective we actually care about.

The multivariate section pits three feature sources through the *same*
exceedance head on the *same* compound labels:

    joint  -- one MultiChannelReservoir (cross-channel quantum features)
    bank   -- one WindowedReservoir per driver, features concatenated
              (per-channel only; no genuine cross-channel term)
    esn    -- a size-matched classical multivariate echo-state network

so any compound-detection edge of ``joint`` over ``bank`` is attributable to
inter-driver dependence the shared entangler builds and the bank cannot.
That is the falsifiable claim; this module reports it either way, including a
clean null.

Alignment convention (shared by every model): sample k uses inputs
``series[k-L+1 .. k]`` and target/label at month ``k+H``. Chronological split
only; every fitted statistic and operating point comes from the train span.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

from data_sources import HORIZON, L, TRAIN_FRAC
from QRC_theoretical import (MultiChannelReservoir, WindowedReservoir, add_bias,
                             entangler_scrambling, linear_memory_capacity,
                             nmse, ridge_gcv)

LAMS = np.logspace(-8, 4, 25)
INNER_TRAIN_FRAC = 0.8      # inner split of the train span for operating points


# ============================================================ harness
def make_samples(T, L=L, H=HORIZON, train_frac=TRAIN_FRAC):
    """Sample indices and a chronological train/test split.

    ks   -- month index k of each sample (target month is k+H)
    tr,te-- positions INTO ks (not month indices)
    itr,iv- inner train / inner-val positions (train span only), for
            leak-free operating-point and band selection.
    """
    ks = np.arange(L - 1, T - H)
    cut = int(len(ks) * train_frac)
    tr, te = np.arange(cut), np.arange(cut, len(ks))
    icut = int(len(tr) * INNER_TRAIN_FRAC)
    return dict(ks=ks, tr=tr, te=te, itr=tr[:icut], iv=tr[icut:], H=H)


def align(Xall, ks, L=L):
    """Rows of a full feature matrix (k = L-1 .. T-1) aligned to samples ks."""
    return Xall[ks - (L - 1)]


# ============================================================ point forecast
class RidgeModel:
    """Standardise by the fit rows, GCV-ridge read-out; predicts any rows."""

    def fit(self, X, y):
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.w, self.lam = ridge_gcv(add_bias((X - self.mu) / self.sd),
                                     y, LAMS)
        return self

    def predict(self, X):
        return add_bias((X - self.mu) / self.sd) @ self.w


def forecast_scores(y_true, y_pred, y_train, thr):
    """NMSE overall + conditioned on the extreme (|anomaly| > thr) regime,
    where forecast skill is what actually matters and decays fastest."""
    m = y_train.mean()
    ev = np.abs(y_true) > thr
    out = {"nmse": float(nmse(y_true, y_pred, m)),
           "n": int(len(y_true)), "n_event": int(ev.sum())}
    out["nmse_event"] = (float(nmse(y_true[ev], y_pred[ev], m))
                         if ev.any() else None)
    return out


def forecast_univariate(Xall, y, thr, samples=None):
    """Point-forecast a univariate anomaly series from reservoir features.
    Returns scores, the test predictions, and the aligned split."""
    s = samples or make_samples(len(y))
    ks, tr, te = s["ks"], s["tr"], s["te"]
    yt = y[ks + s["H"]]
    Xs = align(Xall, ks)
    model = RidgeModel().fit(Xs[tr], yt[tr])
    pred = model.predict(Xs[te])
    sc = forecast_scores(yt[te], pred, yt[tr], thr)
    sc["lambda"] = float(model.lam)
    return dict(scores=sc, pred=pred, y_test=yt[te], ks=ks, tr=tr, te=te,
                model=model, Xs=Xs, yt=yt)


# ============================================================ finite-shot sampling
# Everything above used EXACT features (the S -> infinity statevector limit). On
# real hardware each <Z_i>, <X_i>, <Z_iZ_j> is estimated from S measurement
# shots. WindowedReservoir.sampled_features draws that estimate by multinomial
# sampling of the exact basis probabilities -- statistically identical to the
# noiseless Qiskit AerSimulator sampled path. Caching the exact per-window basis
# probabilities lets a whole shot-budget sweep reuse ONE reservoir drive.
def probs_cache(res, u, L=L):
    """Exact per-window (p_z, p_x) basis probabilities, one entry per sample
    window k = L-1 .. T-1 (so a shot sweep never re-evolves the reservoir)."""
    return [res.basis_probs(u[k - L + 1:k + 1]) for k in range(L - 1, len(u))]


def sampled_feature_matrix(res, u, shots, rng, L=L, cache=None):
    """Finite-shot feature matrix: each window's features estimated from
    ``shots`` shots per basis. Same shape/rows as res.feature_matrix(u, L)."""
    if cache is None:
        cache = probs_cache(res, u, L)
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    Xf = np.zeros((len(u) - L + 1, res.n_features()))
    for i, k in enumerate(range(L - 1, len(u))):
        Xf[i] = res.sampled_features(u[k - L + 1:k + 1], shots, rng, probs=cache[i])
    return Xf


def forecast_nmse_vs_shots(res, y, u, thr, shot_grid, n_seeds=6, cache=None, base_seed=0):
    """Mean +/- std test NMSE over ``n_seeds`` shot realisations for each S in
    ``shot_grid``, plus the exact (S -> infinity) ceiling. Reuses one drive."""
    if cache is None:
        cache = probs_cache(res, u)
    shot_grid = np.asarray(shot_grid)
    mean = np.zeros(len(shot_grid))
    std = np.zeros(len(shot_grid))
    for i, S in enumerate(shot_grid):
        vals = []
        for s in range(n_seeds):
            Xs = sampled_feature_matrix(res, u, int(S),
                                        base_seed + 1000 * i + s, cache=cache)
            vals.append(forecast_univariate(Xs, y, thr)["scores"]["nmse"])
        mean[i], std[i] = np.mean(vals), np.std(vals)
    exact = forecast_univariate(res.feature_matrix(u, L), y, thr)["scores"]["nmse"]
    return mean, std, exact


def forecast_cv_strict(month, raw, make_features, horizons=(HORIZON,),
                       n_folds=6, tf0=0.4):
    """STRICT rolling-origin forecast CV: every fitted statistic per fold.

    A single 80/20 split is fragile for these non-stationary anomalies (SST
    H=3 swings ~0.67 -> ~1.29 with the cut), so skill is pooled across
    expanding-window folds. Audit fix (2026-07): an earlier version reused the
    80%-span climatology/[0,1]-scaler for every fold, so folds testing months
    before the 80% boundary had those months inside their own deseasonalisation
    statistics -- a leak worth ~+0.04 NMSE on SST. Here EVERYTHING is re-fitted
    per fold on that fold's train months only: monthly climatology, scaler, the
    reservoir feature matrix (via ``make_features(u)``), and the GCV ridge.

    Fold boundaries are calendar months (np.linspace(tf0*T, T)), shared by all
    horizons, so the per-fold transforms/features are computed once and reused
    across ``horizons``. Pooled NMSE = sum-fold SSE(model) / sum-fold SSE(train
    -mean); persistence pooled the same way. A fold only predicts months
    strictly after every statistic that produced its model."""
    from data_sources import deseasonalize, fit_scaler
    T = len(raw)
    M = np.linspace(int(tf0 * T), T, n_folds + 1).astype(int)   # month bounds
    folds = []
    for f in range(n_folds):
        yf, _ = deseasonalize(month, raw, M[f])    # climatology: months [0,M_f)
        scaler, _ = fit_scaler(yf, M[f])
        folds.append((yf, make_features(scaler(yf))))
    out = {}
    for H in horizons:
        ks = np.arange(L - 1, T - H)
        sse_m = sse_0 = sse_p = 0.0
        n_te = 0
        for f, (yf, Xf) in enumerate(folds):
            tr = np.where(ks + H < M[f])[0]
            te = np.where((ks + H >= M[f]) & (ks + H < M[f + 1]))[0]
            if len(te) == 0 or len(tr) < 50:
                continue
            Xs, yt = align(Xf, ks), yf[ks + H]
            model = RidgeModel().fit(Xs[tr], yt[tr])
            pred = model.predict(Xs[te])
            mu = yt[tr].mean()
            sse_m += float(np.sum((yt[te] - pred) ** 2))
            sse_0 += float(np.sum((yt[te] - mu) ** 2))
            sse_p += float(np.sum((yt[te] - yf[ks[te]]) ** 2))
            n_te += len(te)
        out[H] = dict(qrc=sse_m / sse_0, persistence=sse_p / sse_0,
                      mean=1.0, n=n_te)
    return out


def point_forecast_vs_horizon(month, raw, make_features, horizons):
    """QRC vs persistence pooled-CV NMSE as a function of the forecast lead H,
    on the strict per-fold engine (one set of fold transforms/features is
    shared by every horizon -- see forecast_cv_strict)."""
    cv = forecast_cv_strict(month, raw, make_features, horizons=tuple(horizons))
    return dict(horizons=list(horizons),
                qrc=np.array([cv[H]["qrc"] for H in horizons]),
                persistence=np.array([cv[H]["persistence"] for H in horizons]))


# ============================================================ classical battery
class ESN:
    """Leaky echo-state network; N units == N read-out features.
    ``Win`` is (N, C) so the same class serves uni- and multivariate inputs."""

    def __init__(self, n_units=20, n_in=1, rho=0.9, leak=0.3, in_scale=1.0,
                 seed=7):
        rng = np.random.default_rng(seed)
        Wr = rng.normal(size=(n_units, n_units))
        Wr *= rho / np.max(np.abs(np.linalg.eigvals(Wr)))
        self.Wr = Wr
        self.Win = rng.uniform(-in_scale, in_scale, size=(n_units, n_in))
        self.b = rng.uniform(-0.1, 0.1, size=n_units)
        self.leak = leak

    def states(self, series):
        series = np.asarray(series, dtype=float)
        if series.ndim == 1:                       # (T,) -> (T, 1)
            series = series[:, None]
        assert series.shape[1] == self.Win.shape[1], "input width != Win width"
        x = np.zeros(self.Wr.shape[0])
        out = np.zeros((len(series), len(x)))
        for t in range(len(series)):
            x = (1 - self.leak) * x + self.leak * np.tanh(
                self.Wr @ x + self.Win @ series[t] + self.b)
            out[t] = x
        return out


def _haar_reservoir(seed):
    """WindowedReservoir with W replaced by a Haar-random unitary: same
    observables and read-out, so it isolates 'tuned dynamics' from 'quantum'."""
    r = WindowedReservoir(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    d = 2 ** r.n
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    r.W = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))
    return r


def classical_battery(y, u_scaled, thr, seed=7, samples=None):
    """mean, persistence, linear_lags, esn, haar -- the bar the QRC must clear.

    Baseline inputs (audit fix, 2026-07): the classical baselines consume the
    RAW anomaly series ``y`` -- only ``haar`` needs ``u_scaled`` because its
    quantum encoding requires bounded [0,1] input. The ESN state matrix has one
    row per TIME STEP, so it is indexed at ``ks`` directly; ``align()`` is only
    for windowed feature matrices whose row 0 is k = L-1. (A previous version
    passed the ESN through ``align()``, silently feeding it a state L-1 = 23
    months stale, and fed ESN/linear_lags the clipped [0,1] series -- both
    handicapped the classical side. This configuration reproduces the validated
    reference battery in QRC_Climate_Prediction_complete/baselines.py.)
    """
    s = samples or make_samples(len(y))
    ks, tr, te = s["ks"], s["tr"], s["te"]
    yt = y[ks + s["H"]]
    m = yt[tr].mean()

    def ridge_pred(Xs):                           # Xs already sample-aligned
        model = RidgeModel().fit(Xs[tr], yt[tr])
        return model.predict(Xs[te])

    lag = np.stack([y[k - L + 1:k + 1] for k in ks])          # raw-anomaly lags
    preds = {
        "mean": np.full(len(te), m),
        "persistence": y[ks[te]],
        "linear_lags": ridge_pred(lag),
        "esn": ridge_pred(ESN(n_in=1, seed=seed).states(y)[ks]),
        "haar": ridge_pred(align(_haar_reservoir(seed).feature_matrix(u_scaled, L), ks)),
    }
    return {name: forecast_scores(yt[te], p, yt[tr], thr)
            for name, p in preds.items()}


# ============================================================ exceedance head
def _operating_threshold(p_val, y_val):
    """Probability cut-off that maximises F1 on the inner-validation slice
    (leak-free: the test span never sees it). Falls back to 0.5 if degenerate."""
    if y_val.sum() == 0 or y_val.sum() == len(y_val):
        return 0.5
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def exceedance_head(X, labels, samples, seed=7, C_reg=1.0):
    """Logistic P(event) head on reservoir features.

    Fit on inner-train, pick the operating point on inner-val, refit on the
    full train span, evaluate on test. Returns threshold-free scores (average
    precision, ROC-AUC) plus precision/recall/F1 at the chosen operating point.
    """
    ks, tr, te = samples["ks"], samples["tr"], samples["te"]
    itr, iv = samples["itr"], samples["iv"]
    Xs = align(X, ks)
    yb = labels[ks + samples["H"]].astype(int)

    def _std_fit(rows):
        mu, sd = Xs[rows].mean(0), Xs[rows].std(0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(class_weight="balanced", C=C_reg,
                                 max_iter=2000, random_state=seed)
        clf.fit((Xs[rows] - mu) / sd, yb[rows])
        return clf, mu, sd

    clf_i, mu_i, sd_i = _std_fit(itr)
    p_iv = clf_i.predict_proba((Xs[iv] - mu_i) / sd_i)[:, 1]
    op = _operating_threshold(p_iv, yb[iv])

    clf, mu, sd = _std_fit(tr)
    p_te = clf.predict_proba((Xs[te] - mu) / sd)[:, 1]
    yt = yb[te]
    pred = (p_te >= op).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(
        ap=float(average_precision_score(yt, p_te)) if yt.any() else None,
        auc=float(roc_auc_score(yt, p_te)) if 0 < yt.sum() < len(yt) else None,
        precision=prec, recall=rec, f1=f1,
        base_rate=float(yt.mean()), n_event=int(yt.sum()), n=int(len(yt)),
        op_threshold=op, proba=p_te, labels=yt,
    )


# ============================================================ compound experiment
def _feature_sources(comp, gamma, ent_scale, seed):
    """Build the three multivariate feature matrices (full, k=L-1..T-1)."""
    U, names = comp["U"], comp["names"]
    C = len(names)
    joint = MultiChannelReservoir(gamma=gamma, seed=seed, ent_scale=ent_scale,
                                  n_channels=C).feature_matrix(U, L)
    banks = [WindowedReservoir(gamma=gamma, seed=seed + 100 * (c + 1),
                               ent_scale=ent_scale).feature_matrix(U[:, c], L)
             for c in range(C)]
    bank = np.hstack(banks)
    esn = ESN(n_units=20 * C, n_in=C, seed=seed).states(U)[L - 1:]
    return {"joint": joint, "bank": bank, "esn_multi": esn}


def compound_experiment(comp, gamma=np.pi / 4, ent_scale=0.5, seed=7):
    """joint vs bank vs classical ESN through the same exceedance head, on the
    compound co-exceedance label. Also runs the naive point-forecast-then-
    threshold detector on the joint features to expose the mean-regression
    failure mode. Returns a dict ready for tabulation/plotting."""
    samples = make_samples(comp["T"])
    labels = comp["compound"].astype(int)
    feats = _feature_sources(comp, gamma, ent_scale, seed)

    heads = {name: exceedance_head(X, labels, samples, seed=seed)
             for name, X in feats.items()}
    naive = _forecast_then_threshold(comp, feats["joint"], samples)
    return dict(samples=samples, feats=feats, heads=heads, naive=naive,
                base_rate=float(labels[samples["ks"] + HORIZON][samples["te"]].mean()))


def _forecast_then_threshold(comp, joint_feats, samples):
    """Detector (a): ridge-forecast each driver's anomaly, then flag a compound
    event when the forecast LEVELS jointly exceed. Point forecasts regress to
    the mean, so this systematically under-detects the tail -- we report its
    recall next to the probabilistic head's to make the gap explicit."""
    ks, tr, te = samples["ks"], samples["tr"], samples["te"]
    H = samples["H"]
    Xs = align(joint_feats, ks)
    Y, thr, signs = comp["Y"], comp["thresholds"], comp["signs"]
    exceed_pred = np.ones((len(te), len(comp["names"])), dtype=bool)
    for c in range(len(comp["names"])):
        yt = Y[ks + H, c]
        model = RidgeModel().fit(Xs[tr], yt[tr])
        fc = model.predict(Xs[te])
        exceed_pred[:, c] = (fc * signs[c]) > thr[c]
    pred = exceed_pred.all(axis=1)
    truth = comp["compound"][ks[te] + H]
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=prec, recall=rec,
                f1=(2 * prec * rec / (prec + rec) if prec + rec else 0.0),
                n_event=int(truth.sum()), n=int(len(truth)))


def _pooled_cv_head(Xs, yb, seed=7, n_folds=6):
    """Rolling-origin (expanding-window) evaluation of one logistic exceedance
    head. Out-of-sample probabilities are POOLED across folds so average
    precision / ROC-AUC are computed over ALL events in the record, not the
    handful in a single test tail. Per-fold F1 (at a leak-free operating point
    chosen on each fold's inner-val) is averaged.

    Xs, yb are already sample-aligned (row k is the window ending at month
    ks[k], label at ks[k]+H). A fold only ever predicts strictly future rows.
    """
    n = len(yb)
    bounds = np.linspace(int(0.4 * n), n, n_folds + 1).astype(int)
    pooled_p, pooled_y, f1s = [], [], []
    for f in range(n_folds):
        tr, te = np.arange(0, bounds[f]), np.arange(bounds[f], bounds[f + 1])
        if len(te) == 0 or yb[tr].sum() == 0:
            continue
        icut = int(len(tr) * INNER_TRAIN_FRAC)
        itr, iv = tr[:icut], tr[icut:]
        clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                                 random_state=seed)

        def _fit_predict(fit_rows, pred_rows):
            mu, sd = Xs[fit_rows].mean(0), Xs[fit_rows].std(0)
            sd[sd == 0] = 1.0
            clf.fit((Xs[fit_rows] - mu) / sd, yb[fit_rows])
            return clf.predict_proba((Xs[pred_rows] - mu) / sd)[:, 1]

        op = _operating_threshold(_fit_predict(itr, iv), yb[iv])
        p = _fit_predict(tr, te)
        pooled_p.append(p)
        pooled_y.append(yb[te])
        pred = (p >= op).astype(int)
        tp = ((pred == 1) & (yb[te] == 1)).sum()
        denom = (2 * tp + ((pred == 1) & (yb[te] == 0)).sum()
                 + ((pred == 0) & (yb[te] == 1)).sum())
        f1s.append(2 * tp / denom if denom else 0.0)
    py, pp = np.concatenate(pooled_y), np.concatenate(pooled_p)
    return dict(
        ap=float(average_precision_score(py, pp)),
        auc=float(roc_auc_score(py, pp)) if 0 < py.sum() < len(py) else None,
        f1_mean=float(np.mean(f1s)), f1_folds=[round(x, 3) for x in f1s],
        n_event=int(py.sum()), n=int(len(py)), base_rate=float(py.mean()))


def compound_cv(comp, gamma=np.pi / 4, ent_scale=0.5, seed=7, n_folds=6):
    """joint vs bank vs classical ESN on the compound label, each through the
    same pooled rolling-origin exceedance head (see _pooled_cv_head)."""
    T, H = comp["T"], HORIZON
    feats = _feature_sources(comp, gamma, ent_scale, seed)
    ks = np.arange(L - 1, T - H)
    yb = comp["compound"].astype(int)[ks + H]
    return {name: _pooled_cv_head(align(X, ks), yb, seed=seed, n_folds=n_folds)
            for name, X in feats.items()}


def skill_vs_horizon(comp, horizons=(1, 3, 6, 9, 12), gamma=np.pi / 4,
                     ent_scale=0.5, seed=7, n_folds=6):
    """Pooled-CV average precision / AUC of the joint-reservoir exceedance head
    vs lead time. Features do not depend on H (only the label shifts), so the
    same feature matrix is reused; this shows the horizon at which compound
    skill collapses toward the base rate."""
    U, C = comp["U"], len(comp["names"])
    joint = MultiChannelReservoir(gamma=gamma, seed=seed, ent_scale=ent_scale,
                                  n_channels=C).feature_matrix(U, L)
    labels = comp["compound"].astype(int)
    out = {}
    for H in horizons:
        ks = np.arange(L - 1, comp["T"] - H)
        out[H] = _pooled_cv_head(align(joint, ks), labels[ks + H],
                                 seed=seed, n_folds=n_folds)
    return out


# ============================================================ operating-point sweep
def _joint_features(comp, gamma, ent_scale, seed):
    return MultiChannelReservoir(gamma=gamma, seed=seed, ent_scale=ent_scale,
                                 n_channels=len(comp["names"])).feature_matrix(comp["U"], L)


def _bank_features(comp, gamma, ent_scale, seed):
    return np.hstack([
        WindowedReservoir(gamma=gamma, seed=seed + 100 * (c + 1),
                          ent_scale=ent_scale).feature_matrix(comp["U"][:, c], L)
        for c in range(len(comp["names"]))])


def operating_sweep(comp, gammas, ent_scales, model="joint", seed=7, n_folds=6):
    """Rolling-CV compound-detection AP/AUC over a (gamma, ent_scale) grid, for
    the joint reservoir or the univariate bank, on the EXACT matrix reservoir.

    This is the operating-point sweep the single-point study left open. Reading
    the grid MAX is an optimistic upper bound (the point is chosen on the same
    CV), which is exactly the right test of robustness: if the joint reservoir
    cannot beat the classical alternatives even at its best grid cell, the null
    is robust; if it can, that cell is a lead for nested validation."""
    build = {"joint": _joint_features, "bank": _bank_features}[model]
    ks = np.arange(L - 1, comp["T"] - HORIZON)
    yb = comp["compound"].astype(int)[ks + HORIZON]
    ap = np.zeros((len(gammas), len(ent_scales)))
    auc = np.zeros_like(ap)
    for gi, g in enumerate(gammas):
        for sj, s in enumerate(ent_scales):
            h = _pooled_cv_head(align(build(comp, g, s, seed), ks), yb,
                                seed=seed, n_folds=n_folds)
            ap[gi, sj], auc[gi, sj] = h["ap"], (h["auc"] or np.nan)
    best = np.unravel_index(np.nanargmax(ap), ap.shape)
    return dict(gammas=np.asarray(gammas), ent_scales=np.asarray(ent_scales),
                ap=ap, auc=auc, model=model,
                best=dict(gamma=float(gammas[best[0]]), ent_scale=float(ent_scales[best[1]]),
                          ap=float(ap[best]), auc=float(auc[best])))


def memory_capacity_grid(comp, gammas, ent_scales, seed=7, T=1200):
    """Theoretical panel: linear memory capacity of the JOINT reservoir over the
    same (gamma, ent_scale) grid, computed DIRECTLY from the reservoir matrices
    on i.i.d. input (no task data, no fitted detector) -- see
    QRC_theoretical.linear_memory_capacity. Also returns the pure-matrix
    entangler scrambling for each ent_scale."""
    C = len(comp["names"])
    mc = np.zeros((len(gammas), len(ent_scales)))
    for gi, g in enumerate(gammas):
        for sj, s in enumerate(ent_scales):
            res = MultiChannelReservoir(gamma=g, seed=seed, ent_scale=s, n_channels=C)
            mc[gi, sj], _ = linear_memory_capacity(res, L, T=T, seed=0)
    scrambling = np.array([
        entangler_scrambling(MultiChannelReservoir(
            gamma=gammas[0], seed=seed, ent_scale=s, n_channels=C).W)
        for s in ent_scales])
    return dict(gammas=np.asarray(gammas), ent_scales=np.asarray(ent_scales),
                mc=mc, scrambling=scrambling)


if __name__ == "__main__":
    import data_sources as ds

    print("== univariate SST forecast (exact numpy reservoir) ==")
    d = ds.prepare_univariate("sst", verbose=False)
    res = WindowedReservoir(gamma=np.pi / 4, seed=7, ent_scale=0.5)
    Xall = res.feature_matrix(d["u"], L)
    fc = forecast_univariate(Xall, d["y"], d["event_threshold"])
    print("  QRC point forecast:", {k: round(v, 3) if isinstance(v, float) else v
                                     for k, v in fc["scores"].items()})
    bat = classical_battery(d["y"], d["u"], d["event_threshold"])
    for name, sc in bat.items():
        print(f"  {name:12s} NMSE={sc['nmse']:.3f}")

    print("\n== compound SST+SOI detection (joint vs bank vs ESN) ==")
    comp = ds.prepare_compound(("sst", "soi"), verbose=False)
    exp = compound_experiment(comp)
    print(f"  test base rate = {exp['base_rate']:.3f}")
    for name, h in exp["heads"].items():
        print(f"  {name:10s} AP={h['ap']:.3f} AUC={h['auc']:.3f} "
              f"F1={h['f1']:.3f} recall={h['recall']:.3f}")
    print(f"  naive point-then-threshold: recall={exp['naive']['recall']:.3f} "
          f"F1={exp['naive']['f1']:.3f}  <- mean-regression failure mode")

    print("\n== joint operating-point sweep (2x2 smoke test) ==")
    gammas, ent_scales = [np.pi / 8, np.pi / 4], [0.5, 1.0]
    sw = operating_sweep(comp, gammas, ent_scales, model="joint")
    print(f"  best joint over grid: gamma={sw['best']['gamma']:.3f} "
          f"ent_scale={sw['best']['ent_scale']} AP={sw['best']['ap']:.3f}")
    mcg = memory_capacity_grid(comp, gammas, ent_scales, T=600)
    print(f"  memory-capacity grid (theoretical):\n{np.round(mcg['mc'], 2)}")
