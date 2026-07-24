"""scripts/estimate_sequential_latency.py -- P11: sequential-QPU feasibility.

Builds the per-step latency ledger for the closed-loop rewind architecture from
MEASURED local components (wall-clocked on Aer here) + ESTIMATED remote components
(queue/network/execute, read from the frozen assumptions file beside the backend
snapshot -- never live). Compares strategies A-G, totals the SEQUENTIAL step counts
1/12/H_eff/100/1000 and one rolling-origin backtest, fits the classical surrogate
(strategy F / Gate-7 comparator), and writes the accuracy-vs-latency Pareto.

Batched open-loop throughput is NOT used as closed-loop evidence (acceptance item 24):
every total is n x (one QPU round trip). Results -> results/metrics/sequential_latency.json
and the Pareto figure -> results/figures/latency_pareto.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.hardware import latency_model as LM
from qrc_single_time_series.hardware import sequential_execution as SEQ

SHOTS = 1024                      # representative preregistered shot budget (P10 grid)
N, T_W, TAU = 5, 10, 1.0         # rewind config matching the P10 shot study


def _h_eff():
    """H_effective (rounded) + backtest step count from the stored climate campaign."""
    p = ROOT / "results" / "metrics" / "climate_campaign.json"
    if not p.exists():
        return 4, 20, 4.5, 20      # documented fallback if the campaign isn't built
    d = json.loads(p.read_text())["enso"]["autonomous"]
    h = float(d["median_H_effective_months"])
    n_origins = int(d["n_origins"])
    return int(round(h)), n_origins, h, n_origins


def main():
    asm = LM.LatencyAssumptions.load()
    print("Sequential-QPU feasibility (rewind architecture)")
    print("=" * 62)
    print(f"backend (ASSUMED class): {asm.backend_name}  shots={SHOTS}")
    print(f"remote timing is ESTIMATED from {Path('results/backend_snapshots/latency_assumptions.yaml')}")
    print("  queue=%.0fs  net=%.2f+%.2fs  execute(1024)=%.3fs  (all ASSUMPTIONS)"
          % (asm.queue_s, asm.network_send_s, asm.result_return_s, asm.execute_s(SHOTS)))

    print("\nWall-clocking local components (this machine, on Aer) ...")
    measured = LM.measure_local_components(N=N, t_w=T_W, tau=TAU, shots=SHOTS, repeats=7)
    md = measured
    print("  construct=%.3fms bind=%.3fms transpile=%.3fms readout=%.3fms "
          "local_exec=%.3fms buffer=%.4fms depth=%d"
          % (md.construct_s*1e3, md.bind_s*1e3, md.transpile_s*1e3, md.readout_s*1e3,
             md.local_execute_s*1e3, md.buffer_s*1e3, md.meta["circuit_depth"]))

    print("\nFitting classical surrogate of the feature map (strategy F) ...")
    surrogate = SEQ.fit_surrogate(N=4, t_w=6, tau=TAU)
    print("  QRC one-step NMSE=%.4f   surrogate NMSE=%.4f   feat_rmse=%.4f   "
          "predict=%.1fus" % (surrogate.qrc_test_nmse, surrogate.surrogate_test_nmse,
                              surrogate.feature_rmse, surrogate.predict_s*1e6))

    h_eff, n_origins, h_eff_raw, backtest_origins = _h_eff()
    backtest_steps = backtest_origins * h_eff
    step_counts = {"1": 1, "12": 12, "H_eff": h_eff, "100": 100, "1000": 1000}

    ledgers = SEQ.compare_strategies(measured, asm, SHOTS, surrogate.predict_s)
    accuracy = SEQ.accuracy_by_strategy(surrogate.qrc_test_nmse,
                                        surrogate.surrogate_test_nmse)
    totals = SEQ.sequential_totals(ledgers, step_counts, backtest_steps)
    pareto = SEQ.pareto_points(ledgers, accuracy)
    verdicts = SEQ.deployment_verdicts(totals, horizon_label="H_eff")

    # ---- per-step ledger + totals table -----------------------------------
    print(f"\nPer-step + sequential totals (H_eff={h_eff} from ENSO median "
          f"{h_eff_raw}; backtest={backtest_origins}x{h_eff}={backtest_steps} steps)")
    print("-" * 62)
    print("%-38s %10s %10s %10s" % ("strategy", "per_step", "12-step", "backtest"))
    for k, led in ledgers.items():
        t = totals[k]["totals_s"]
        print("%-38s %9.4fs %9.1fs %9.1fs"
              % (k, led["total_s"], t["12"], t["backtest"]))

    print("\nAccuracy-vs-latency Pareto (one-step NMSE; * = Pareto-optimal):")
    for p in sorted(pareto, key=lambda q: q["per_step_s"]):
        star = "*" if p.get("pareto_optimal") else " "
        cl = "" if p["closed_loop_capable"] else "  [NOT closed-loop capable]"
        print("  %s %-38s lat=%9.4fs  nmse=%.3f%s"
              % (star, p["strategy"], p["per_step_s"], p["nmse"], cl))

    print("\nPer-mode verdicts (s26.4 practicality, at H_eff horizon):")
    for mode in ("research_sim", "delayed_batch", "operational"):
        v = verdicts[mode]
        print(f"  {mode}: {v['verdict']}")

    fig_path = _pareto_figure(pareto, ROOT)

    out = {
        "config": {"N": N, "t_w": T_W, "tau": TAU, "shots": SHOTS,
                   "h_eff": h_eff, "h_eff_raw": h_eff_raw,
                   "backtest_origins": backtest_origins, "backtest_steps": backtest_steps,
                   "step_counts": step_counts},
        "assumptions": {
            "backend_name": asm.backend_name, "per_shot_execution_s": asm.per_shot_execution_s,
            "sampler_overhead_s": asm.sampler_overhead_s, "queue_s": asm.queue_s,
            "network_send_s": asm.network_send_s, "result_return_s": asm.result_return_s,
            "refresh_period_steps": asm.refresh_period_steps, "sources": list(asm.sources),
            "note": "remote components are ASSUMPTIONS, not live-backend measurements."},
        "measured_local_s": {
            "construct": md.construct_s, "bind": md.bind_s, "transpile": md.transpile_s,
            "readout": md.readout_s, "local_execute": md.local_execute_s,
            "buffer": md.buffer_s, "meta": md.meta},
        "surrogate": {
            "qrc_test_nmse": surrogate.qrc_test_nmse,
            "surrogate_test_nmse": surrogate.surrogate_test_nmse,
            "feature_rmse": surrogate.feature_rmse, "predict_s": surrogate.predict_s,
            "meta": surrogate.meta},
        "ledgers": ledgers,
        "totals": totals,
        "accuracy_onestep_nmse": accuracy,
        "pareto": pareto,
        "verdicts": verdicts,
        "structural_conclusion": (
            "The rewind loop is inherently sequential: circuit t+1 depends on result t, "
            "so a k-step forecast is k QPU round trips, each dominated by the queue "
            "assumption. On the accuracy-vs-latency plane the local simulator (D) and "
            "classical surrogate (F) dominate every remote-QPU strategy at this problem "
            "size. For a MONTHLY index the horizon latency (minutes) is far inside the "
            "decision deadline, so the binding constraint is QPU access/cost, not "
            "per-forecast latency."),
        "figure": str(fig_path.relative_to(ROOT)) if fig_path else None,
    }
    dest = ROOT / "results" / "metrics" / "sequential_latency.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {dest.relative_to(ROOT)}")
    if fig_path:
        print(f"-> {fig_path.relative_to(ROOT)}")


def _pareto_figure(pareto, root):
    """Accuracy-vs-latency scatter (spec fig. 40); skips cleanly if mpl is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                              # pragma: no cover
        print(f"(figure skipped: {exc})")
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for p in pareto:
        cl = p["closed_loop_capable"]
        opt = p.get("pareto_optimal")
        color = "tab:green" if opt else ("tab:blue" if cl else "tab:red")
        kw = dict(marker="o", s=90 if opt else 55, color=color, zorder=3)
        if cl:                                            # filled marker: outline it
            kw.update(edgecolor="k" if opt else "none", linewidth=1.2 if opt else 0.0)
        else:                                             # unfilled x for E (no outline)
            kw["marker"] = "x"
        ax.scatter(p["per_step_s"], p["nmse"], **kw)
        ax.annotate(p["strategy"].split("_")[0], (p["per_step_s"], p["nmse"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("per-step latency (s, log)  --  remote parts ESTIMATED")
    ax.set_ylabel("one-step held-out NMSE (lower = better)")
    ax.set_title("Sequential-QPU accuracy vs latency (rewind, ENSO)\n"
                 "green = Pareto-optimal; x = not closed-loop capable")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    dest = root / "results" / "figures" / "latency_pareto.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


if __name__ == "__main__":
    main()
