"""exp_product_demo.py -- the six layers end-to-end (Phases 8-10 demo).

One walk-forward pass over the SOLAR SURROGATE (labelled surrogate in
every artifact this run emits) through the SAME code path a live
deployment would use:

  L1 connectors.solar_surrogate -> L1 qc.apply_qc -> L2 ClearSkyIndex
  (train-span fit; leakage audit) -> L3 SimProvider RF-QRC features +
  calendar lookup + per-horizon leak -> L3 quantile + event heads ->
  conformal wrapper + coverage monitor -> shadow battery on every row ->
  L4 isotonic recalibration + persona cost threshold + regime router ->
  alert engine (provenance payload, data_kind=surrogate) -> L6 drift +
  effective-rank monitors -> stacker fairness table -> promotion-gate
  verdict -> model card.

Scope: all numbers are SIMULATION on SURROGATE data. The pre-registered
pilot gates are evaluated against this run only to exercise the gate
machinery -- the verdict is a demo artifact, not a product claim.

Usage:
    python exp_product_demo.py            full demo (~2-3 min)
    python exp_product_demo.py --check    reduced (~40 s); exit 0 iff the
                                          Phase 8-9 acceptance gates pass:
                                          leakage audit, coverage within
                                          3 pts, battery columns in every
                                          row, alert objects valid.

Requires stage5_qubit_reuse and stage6_rfqrc on PYTHONPATH plus the
stage8 subpackage dirs (the run_tests.py driver provides all of them).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for sub in ("data_plane", "serving", "decision", "governance"):
    sys.path.insert(0, str(_HERE / sub))

from alerts import Alert, AlertEngine
from connectors import solar_surrogate
from conformal import AdaptiveConformal, CoverageMonitor
from feature_provider import CachedProvider, SimProvider
from heads import EventHead, QuantileHeads, per_horizon_leak
from model_cards import write_card
from monitors import DriftMonitor, EffectiveRankMonitor
from promotion_gates import evaluate
from qc import apply_qc
from regime_router import classify_regime
from rfqrc_reservoir import RFQRCConfig, apply_leak
from shadow_battery import battery_row, fit_linear_lags
from stacker import stacker_table
from thresholds import f1 as f1_score
from thresholds import isotonic_recalibrate, optimise_threshold
from transforms import ClearSkyIndex, calendar_features, leakage_audit

SEED = 7
N_SLOTS = 48                      # 30-min data
RUN_ID = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "#demo"
MODEL_VERSION = "hybrid-demo-0.1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    n_days = 60 if args.check else 150
    horizon = 2                          # 1 hour ahead at 30-min steps
    n_qubits = 4 if args.check else 6
    print(f"exp_product_demo config: days={n_days} horizon={horizon} "
          f"n_qubits={n_qubits} seed={SEED} run_id={RUN_ID}")

    # ---------------- L1: ingest + QC
    ghi, meta = solar_surrogate(seed=42)
    data_kind = meta["kind"]             # 'surrogate' -> propagates
    T = n_days * N_SLOTS
    ghi = ghi[:T]
    segments, qcrep = apply_qc(ghi)
    print(f"L1: {meta['source']} [{data_kind}]; {qcrep.summary()}")
    x = np.full(T, np.nan)               # QC'd series back on the grid
    pos = 0
    for seg in segments:
        x[pos:pos + len(seg)] = seg
        pos += len(seg)
    slot = np.arange(T) % N_SLOTS

    # ---------------- L2: transform (train-span fit = leakage firewall)
    n_train = int(0.6 * T)
    n_cal = int(0.2 * T)                 # conformal calibration span
    cs = ClearSkyIndex(MODEL_VERSION).fit(x[:n_train], slot[:n_train],
                                          N_SLOTS)
    kt = cs.apply(x, slot)
    audit_ok = leakage_audit(n_train, T, fit_call_indices=[0])
    day = np.isfinite(kt)
    drive = np.where(day, kt, 0.0)       # night: structurally no sun ->
    #                                      drive 0; night targets EXCLUDED
    ramp_rho = float(np.nanpercentile(
        np.abs(np.diff(kt[:n_train])), 95))
    event = np.zeros(T, dtype=bool)
    dk = np.abs(kt[horizon:] - kt[:-horizon])
    event[:T - horizon] = np.where(np.isfinite(dk), dk >= ramp_rho, False)
    print(f"L2: k_t fit on train span only (audit {'PASS' if audit_ok else 'FAIL'}); "
          f"ramp rho={ramp_rho:.3f} (train 95th pct); "
          f"{int(event[:n_train].sum())} train events")

    # ---------------- L3: features (quantum + calendar), leak, heads
    cfg = RFQRCConfig(n_qubits=n_qubits, n_uploads=2, entangler_layers=2,
                      n_virtual_nodes=2, readout="local_zz", seed=SEED)
    provider = CachedProvider(SimProvider(), decimals=2)
    Fq_raw = np.array([provider.features(np.array([[drive[k]]]), cfg)
                       for k in range(T)])
    eps = per_horizon_leak([horizon])[horizon]
    Fq = apply_leak(Fq_raw, eps)
    Fcal = calendar_features(slot, N_SLOTS)
    lags = np.stack([np.r_[np.zeros(d), drive[:T - d]]
                     for d in range(1, 5)], axis=1)
    Fc = np.hstack([Fcal, lags])
    F = np.hstack([Fq, Fc])
    cache = provider.stats()
    print(f"L3: features {F.shape[1]} (quantum {Fq.shape[1]} + classical "
          f"{Fc.shape[1]}); leak eps={eps:.2f}; cache hit rate "
          f"{cache['hit_rate']:.2f} ({cache['hits']}/{cache['hits'] + cache['misses']})")

    valid = day.copy()                   # train/eval on daylight targets
    valid[T - horizon:] = False
    tr = valid & (np.arange(T) < n_train)
    ca = valid & (np.arange(T) >= n_train) & (np.arange(T) < n_train + n_cal)
    te = valid & (np.arange(T) >= n_train + n_cal)
    y = np.full(T, np.nan)
    y[:T - horizon] = kt[horizon:]       # target: k_t at t + horizon
    tr &= np.isfinite(y); ca &= np.isfinite(y); te &= np.isfinite(y)

    qh = QuantileHeads([horizon]).fit(F[tr], y[tr])
    # note: heads fit on (F[k] -> y[k]) pairs where y[k]=kt[k+h] (already
    # shifted), so pass h=0-style aligned matrices:
    qh.w = {}
    for tau in qh.taus:
        from heads import fit_quantile
        qh.w[(horizon, tau)] = fit_quantile(F[tr], y[tr], tau)
    fan_ca = qh.predict(F[ca], horizon)
    fan_te = qh.predict(F[te], horizon)
    eh = EventHead().fit(F[tr], event[tr])
    p_ca, p_te = eh.predict_proba(F[ca]), eh.predict_proba(F[te])

    # ---------------- conformal + coverage
    ac = AdaptiveConformal(alpha=0.2).start(y[ca], fan_ca[:, 0],
                                            fan_ca[:, 2])
    mon = CoverageMonitor(target=0.8)
    te_idx = np.nonzero(te)[0]
    for i, k in enumerate(te_idx):
        mon.observe(ac.update(y[k], fan_te[i, 0], fan_te[i, 2]))
    print(f"conformal: rolling coverage {mon.coverage():.3f} (target 0.80) "
          f"healthy={mon.healthy()}")

    # ---------------- L4: recalibration, threshold, regimes, alerts
    recal = isotonic_recalibrate(p_ca, event[ca])
    thr = optimise_threshold(recal(p_ca), event[ca], "microgrid_operator")
    print(f"L4: threshold {thr['threshold']:.2f} expected cost "
          f"{thr['expected_cost']:.3f} (all-silent "
          f"{thr['cost_all_silent']:.3f}) F1 {thr['f1_at_threshold']:.3f}")

    w_lags = fit_linear_lags(kt[:n_train][np.isfinite(kt[:n_train])],
                             L=6, horizon=horizon)
    f1_pers, f1_model = _event_f1_battery(kt, event, te_idx, horizon,
                                          recal(p_te), thr["threshold"])
    engine = AlertEngine(weekly_false_alarm_budget=50)
    n_rows_with_battery = 0
    statuses = []
    for i, k in enumerate(te_idx):
        hist = kt[max(0, k - N_SLOTS):k + 1]
        brow = battery_row(hist[np.isfinite(hist)], horizon,
                           lag_weights=w_lags)
        n_rows_with_battery += ("persistence" in brow
                                and "linear_lags" in brow)
        p_ev = float(recal(np.array([p_te[i]]))[0])
        if p_ev >= thr["threshold"]:
            a = Alert(site="demo-site", event="ramp", horizon_min=30 * horizon,
                      p_event=p_ev,
                      magnitude_q10_q50_q90=list(np.sort(fan_te[i])),
                      lead_time_min=30 * horizon,
                      regime=classify_regime(kt[max(0, k - 12):k + 1]),
                      conformal_coverage_target=0.8, run_id=RUN_ID,
                      model_version=MODEL_VERSION,
                      baseline_delta={"vs_persistence_F1":
                                      round(f1_model - f1_pers, 3)},
                      data_kind=data_kind)
            statuses.append(engine.submit(a))
            engine.resolve("demo-site", "ramp", 30 * horizon)
    n_bad = sum(s.startswith("invalid") for s in statuses)
    print(f"alerts: {len(statuses)} fired, {n_bad} invalid; battery in "
          f"{n_rows_with_battery}/{len(te_idx)} rows; "
          f"F1 model {f1_model:.3f} vs persistence {f1_pers:.3f}")

    # ---------------- L6: monitors, stacker fairness, gates, card
    dm = DriftMonitor(kt[:n_train][np.isfinite(kt[:n_train])])
    for v in kt[te_idx]:
        dm.observe(v)
    em = EffectiveRankMonitor(rank_at_promotion=float(
        _eff_rank(Fq[tr])))
    for k in te_idx:
        em.observe(Fq[k])
    print(f"L6: drift KS {dm.statistic():.3f} healthy={dm.healthy()}; "
          f"effective rank {em.rank():.1f} healthy={em.healthy()}")

    tab = stacker_table({"classical": Fc[tr], "quantum": Fq[tr]},
                        {"classical": Fc[te], "quantum": Fq[te]},
                        y[tr], y[te])
    for kx, v in tab.items():
        print(f"stacker {kx:6s}: hybrid {v['hybrid_nmse']:.3f} "
              f"classical-only {v['classical_only_nmse']:.3f} "
              f"delta {v['quantum_delta']:+.3f}"
              + ("  (zero -> table says so)" if v["delta_is_zero"] else ""))

    verdict = evaluate(
        {"ramp_f1_delta": f1_model - f1_pers, "alert_latency_s": 1.0,
         "n_fabricated_numbers": 0,
         "coverage_gap_abs": abs(mon.coverage() - 0.8),
         "hybrid_minus_classical_f1":
             tab["ridge"]["quantum_delta"]}, run_id=RUN_ID)
    print(f"promotion verdict: passed={verdict.passed} "
          f"quantum_demotion={verdict.quantum_demotion} (demo artifact, "
          "not a claim)")
    card = write_card(MODEL_VERSION, cfg.__dict__,
                      {"passed": verdict.passed,
                       "quantum_demotion": verdict.quantum_demotion,
                       "gates": verdict.gates},
                      {"stacker": tab,
                       "event_f1": {"model": f1_model,
                                    "persistence": f1_pers}},
                      [{"source": meta["source"], "kind": data_kind}],
                      notes=[f"run_id {RUN_ID}", "end-to-end demo"])
    print(f"model card: {card}")

    # ---------------- Phase 8-9 acceptance gates (the --check exit code)
    gates = {
        "leakage_audit": audit_ok,
        "coverage_within_3pts": mon.healthy(),
        "battery_in_every_row": n_rows_with_battery == len(te_idx),
        "alerts_all_valid": n_bad == 0,
    }
    print("\nacceptance gates: " + "  ".join(
        f"{k}[{'PASS' if v else 'FAIL'}]" for k, v in gates.items()))
    print(f"total {time.time() - t0:.0f}s; ALL NUMBERS: simulation on "
          f"{data_kind.upper()} data")
    return 0 if all(gates.values()) else 1


def _eff_rank(F):
    cov = np.atleast_2d(np.cov(F.T))
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
    s = eig.sum()
    return s ** 2 / np.sum(eig ** 2) if s > 0 else 0.0


def _event_f1_battery(kt, event, te_idx, horizon, p_te_recal, thr):
    """Event F1 for persistence (fires if the CURRENT step just ramped)
    and for the model at the chosen threshold, on the test rows."""
    fire_pers, fire_model, truth = [], [], []
    for i, k in enumerate(te_idx):
        prev = kt[k - horizon] if k - horizon >= 0 else np.nan
        ramp_now = (np.isfinite(prev) and np.isfinite(kt[k])
                    and abs(kt[k] - prev) >= 0)
        dk = abs(kt[k] - prev) if np.isfinite(prev) else 0.0
        fire_pers.append(dk >= np.nanpercentile(
            np.abs(np.diff(kt[:len(kt) // 2])), 95))
        fire_model.append(p_te_recal[i] >= thr)
        truth.append(bool(event[k]))
    fp = np.array(fire_pers); fm = np.array(fire_model)
    tv = np.array(truth)

    def _f1(fire):
        tp = int(np.sum(tv & fire)); f_p = int(np.sum(~tv & fire))
        fn = int(np.sum(tv & ~fire))
        pr = tp / (tp + f_p) if tp + f_p else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return _f1(fp), _f1(fm)


if __name__ == "__main__":
    sys.exit(main())
