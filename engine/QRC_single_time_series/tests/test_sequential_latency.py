"""P11: sequential-QPU latency ledger, strategy A-G comparison, surrogate + Pareto."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrc_single_time_series.hardware import latency_model as LM      # noqa: E402
from qrc_single_time_series.hardware import sequential_execution as SEQ  # noqa: E402


# a cheap synthetic MeasuredLocal so the pure-logic tests never touch Aer
def _fake_measured():
    return LM.MeasuredLocal(
        construct_s=5e-4, bind_s=5e-5, transpile_s=4e-3, readout_s=8e-4,
        local_execute_s=4e-3, buffer_s=1e-6, meta={"N": 5, "t_w": 10})


def _asm():
    return LM.LatencyAssumptions.load()


# ------------------------------------------------------------- assumptions
def test_assumptions_load_and_execute_formula():
    a = _asm()
    assert a.n_qubits >= 1 and a.queue_s > 0
    # execute = session overhead + shots x per-shot anchor
    assert a.execute_s(1000) == pytest.approx(
        a.sampler_overhead_s + 1000 * a.per_shot_execution_s)
    assert a.execute_s(2000) > a.execute_s(1000)


# ------------------------------------------------------------- ledger shape
def test_every_strategy_produces_a_total():
    m, a = _fake_measured(), _asm()
    for k in LM.STRATEGIES:
        led = LM.step_ledger(k, m, a, 1024, surrogate_predict_s=5e-5)
        assert led["total_s"] > 0
        assert set(("strategy", "total_s", "remote", "closed_loop_capable")) <= led.keys()


def test_remote_strategies_carry_the_estimated_queue():
    m, a = _fake_measured(), _asm()
    ledA = LM.step_ledger("A_fresh_circuits", m, a, 1024)
    assert ledA["remote"] and ledA["estimated"]
    assert ledA["components_s"]["queue"] == a.queue_s
    ledD = LM.step_ledger("D_local_noisy_sim", m, a, 1024)
    assert not ledD["remote"] and "queue" not in ledD["components_s"]


def test_local_strategies_orders_of_magnitude_faster_than_remote():
    m, a = _fake_measured(), _asm()
    remote = LM.step_ledger("B_pretranspiled_templates", m, a, 1024)["total_s"]
    local = LM.step_ledger("D_local_noisy_sim", m, a, 1024)["total_s"]
    assert local < remote / 100                          # queue dominates the remote path


def test_fresh_circuits_never_cheaper_than_pretranspiled():
    m, a = _fake_measured(), _asm()
    A = LM.step_ledger("A_fresh_circuits", m, a, 1024)["total_s"]
    B = LM.step_ledger("B_pretranspiled_templates", m, a, 1024)["total_s"]
    assert A >= B                                         # A pays construct+transpile too


def test_offline_features_flagged_not_closed_loop():
    m, a = _fake_measured(), _asm()
    ledE = LM.step_ledger("E_offline_features_classical_deploy", m, a, 1024)
    assert ledE["closed_loop_capable"] is False and ledE["remote"] is False


def test_dynamic_circuits_flag_requires_snapshot_support():
    m, a = _fake_measured(), _asm()
    assert LM.step_ledger("C_dynamic_circuits", m, a, 1024)["requires_dynamic"] is True


def test_periodic_refresh_between_surrogate_and_full_roundtrip():
    m, a = _fake_measured(), _asm()
    B = LM.step_ledger("B_pretranspiled_templates", m, a, 1024)["total_s"]
    F = LM.step_ledger("F_classical_surrogate", m, a, 1024, surrogate_predict_s=5e-5)["total_s"]
    G = LM.step_ledger("G_periodic_refresh", m, a, 1024, surrogate_predict_s=5e-5)["total_s"]
    assert F < G < B                                     # amortised over R steps


# --------------------------------------------------- sequential totals (item 24)
def test_total_time_is_linear_in_sequential_steps():
    assert LM.total_time(2.0, 50) == pytest.approx(100.0)
    m, a = _fake_measured(), _asm()
    led = SEQ.compare_strategies(m, a, 1024, 5e-5)
    tot = SEQ.sequential_totals(led, {"1": 1, "12": 12}, backtest_steps=80)
    b = tot["B_pretranspiled_templates"]
    # closed loop = n round trips: 12 steps cost exactly 12x one step
    assert b["totals_s"]["12"] == pytest.approx(12 * b["per_step_s"])
    assert b["totals_s"]["backtest"] == pytest.approx(80 * b["per_step_s"])


# ------------------------------------------------------------- Pareto frontier
def test_pareto_local_dominates_and_offline_excluded():
    m, a = _fake_measured(), _asm()
    led = SEQ.compare_strategies(m, a, 1024, 5e-5)
    acc = SEQ.accuracy_by_strategy(0.36, 0.40)           # surrogate slightly worse
    pts = {p["strategy"]: p for p in SEQ.pareto_points(led, acc)}
    # local sim (D) has QRC accuracy at ms latency -> optimal; remote QPU dominated
    assert pts["D_local_noisy_sim"]["pareto_optimal"] is True
    assert pts["A_fresh_circuits"]["pareto_optimal"] is False
    # F is optimal (cheapest, only it reaches its own accuracy); E never on the frontier
    assert pts["F_classical_surrogate"]["pareto_optimal"] is True
    assert pts["E_offline_features_classical_deploy"]["pareto_optimal"] is False


# --------------------------------------------------------- s26.4 verdicts
def test_verdicts_monthly_feasible_intraday_not_for_remote():
    m, a = _fake_measured(), _asm()
    led = SEQ.compare_strategies(m, a, 1024, 5e-5)
    tot = SEQ.sequential_totals(led, {"1": 1, "12": 12, "H_eff": 4}, 80)
    v = SEQ.deployment_verdicts(tot, horizon_label="H_eff")
    op = v["operational"]["by_use_case"]
    assert op["monthly_index"]["feasible"] is True       # minutes << one month
    assert op["intraday_trading"]["feasible"] is False   # queue alone busts a 60s loop
    assert v["research_sim"]["by_use_case"]["intraday_trading"]["feasible"] is True


# ----------------------------------------------------- circuit builder shapes
def test_rewind_circuit_measure_variant_is_transpilable_shaped():
    from qrc_single_time_series.quantum.hamiltonians import nn_tfi
    from qrc_single_time_series.quantum.schedule import build_block_schedule
    res = nn_tfi(4, J=1.0, h=0.5, seed=7)
    sch = build_block_schedule(res, block_time=1.0, kappa=1, order=1)
    qc_dm = LM.build_rewind_step_circuit(4, 5, sch, readout="save_dm")
    qc_hw = LM.build_rewind_step_circuit(4, 5, sch, readout="measure")
    assert qc_dm.num_qubits == 4 and qc_hw.num_clbits == 4
    # the measure variant has no Aer save instructions (so it can transpile to a basis)
    names = {ins.operation.name for ins in qc_hw.data}
    assert "save_density_matrix" not in names and "measure" in names


# ---------------------------------------------------- slow integration (real)
@pytest.mark.parametrize("N,t_w", [(4, 4)])
def test_measure_and_surrogate_integration(N, t_w):
    m = LM.measure_local_components(N=N, t_w=t_w, repeats=1)
    assert m.transpile_s > 0 and m.local_execute_s > 0 and m.meta["circuit_depth"] > 0
    # tiny custom series keeps the MLP fit fast; surrogate NMSE should be finite/positive
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.standard_normal(400)) + 50.0
    sf = SEQ.fit_surrogate(series=series, N=N, t_w=t_w, hidden=(16,), predict_repeats=20)
    assert np.isfinite(sf.qrc_test_nmse) and sf.surrogate_test_nmse > 0
    assert sf.predict_s > 0 and sf.feature_rmse >= 0
