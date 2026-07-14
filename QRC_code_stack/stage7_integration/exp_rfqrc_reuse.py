"""exp_rfqrc_reuse.py -- batch-and-compress validation (Part X Phase 6a).

Feeds batches of RF-QRC step circuits (stage 6, frozen) through the
stage-5 reuse compiler (frozen) via qreuse_batch, and checks:

  1. LEGALITY   compiled batch passes validate_reuse_circuit (raised
                inside compile_batch).
  2. WIDTH      compiled width is (a) < logical width, (b) constant in
                the batch size B, and (c) constant in the MEMORY HORIZON
                -- recomputed live against the windowed protocol, whose
                physical count = T + 1 (stage-5 measured pathology).
  3. EQUIVALENCE  per-block TVD(direct, reuse) <= 2.5 x shot-noise floor
                + 0.02 (the reuse suite's self-calibrated gate), plus
                feature interchangeability: |features_direct -
                features_reuse| at the shot-noise scale.
  4. TOPOLOGY   within-block cone width vs entangler choice, including
                the all-to-all (haar_control) block as the honest
                negative control for WITHIN-block compression. Note the
                scan reports two separate facts: all-to-all blocks do
                not compress internally, yet the batch STILL compresses
                across block boundaries (independence is the resource
                the compiler exploits there).

Usage:
    python exp_rfqrc_reuse.py            full scan + table
    python exp_rfqrc_reuse.py --check    reduced; exit 0 iff gates 1-3
                                         pass (stage-7 promotion gate)

Requires stage5_qubit_reuse and stage6_rfqrc on PYTHONPATH (driver
provides them). All results are Aer simulation; no hardware claim.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from qiskit_aer import AerSimulator

from qrc_experiment import (build_qrc_circuit, counts_to_features,
                            make_reservoir_params)
from qreuse_batch import (batch_circuits, compile_batch,
                          compiled_width_of_batch, unbatch_counts)
from qreuse_validation import normalize_counts, total_variation_distance
from rfqrc_reservoir import RFQRCConfig, build_step_circuit, \
    make_entangler_params

SEED = 7


def rfqrc_step_circuits(B: int, cfg: RFQRCConfig, rng) -> list:
    params = make_entangler_params(cfg)
    return [build_step_circuit(np.array([[rng.uniform(0, 1)]]), cfg, params,
                               measure=True) for _ in range(B)]


def tvd_gate(circs, shots: int, seed: int):
    """Per-block self-calibrated TVD gate + feature interchangeability."""
    n = circs[0].num_qubits
    sizes = [c.num_clbits for c in circs]
    batched = batch_circuits(circs)
    compiled = compile_batch(circs)
    be = AerSimulator(seed_simulator=seed)

    def run(qc, s):
        return be.run(qc, shots=shots, seed_simulator=s).result().get_counts()

    direct = unbatch_counts(run(batched, seed), sizes)
    direct2 = unbatch_counts(run(batched, seed + 1), sizes)
    reuse = unbatch_counts(run(compiled.circuit, seed + 2), sizes)

    tvd_dr, floor, feat_d = [], [], []
    for b in range(len(circs)):
        p, p2, q = (normalize_counts(direct[b]), normalize_counts(direct2[b]),
                    normalize_counts(reuse[b]))
        tvd_dr.append(total_variation_distance(p, q))
        floor.append(total_variation_distance(p, p2))
        fd = counts_to_features(direct[b], n)
        fr = counts_to_features(reuse[b], n)
        feat_d.append(float(np.max(np.abs(fd - fr))))
    tvd_dr, floor = float(np.mean(tvd_dr)), float(np.mean(floor))
    ok_tvd = tvd_dr <= 2.5 * floor + 0.02
    sigma = 1.0 / np.sqrt(shots)
    ok_feat = max(feat_d) <= 6.0 * sigma   # two S-shot estimates differ
    return {"tvd": tvd_dr, "floor": floor, "ok_tvd": ok_tvd,
            "feat_maxdiff": max(feat_d), "feat_bound": 6.0 * sigma,
            "ok_feat": ok_feat,
            "physical": compiled.physical_qubit_count,
            "logical": batched.num_qubits}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    n_q = 4 if args.check else 6
    shots = 4096 if args.check else 8192
    B_scan = [2, 4] if args.check else [2, 4, 8, 16]
    T_scan = [2, 6] if args.check else [2, 4, 8, 16, 24]

    cfg = RFQRCConfig(n_qubits=n_q, n_uploads=2, entangler_layers=2,
                      n_virtual_nodes=1, readout="local_zz", seed=SEED)
    print(f"exp_rfqrc_reuse config: n={n_q} shots={shots} B_scan={B_scan} "
          f"T_scan={T_scan} seed={SEED} entangler={cfg.entangler}")

    # ---- gates 1-3: legality + equivalence on a B=3 batch
    circs = rfqrc_step_circuits(3, cfg, rng)
    g = tvd_gate(circs, shots, SEED)
    print(f"\nequivalence gate (B=3): TVD(direct,reuse)={g['tvd']:.4f} vs "
          f"floor {g['floor']:.4f} "
          f"[{'PASS' if g['ok_tvd'] else 'FAIL'}]")
    print(f"feature interchangeability: max|f_d - f_r| = "
          f"{g['feat_maxdiff']:.4f} (bound {g['feat_bound']:.4f}) "
          f"[{'PASS' if g['ok_feat'] else 'FAIL'}]")
    print(f"width: logical {g['logical']} -> physical {g['physical']} "
          f"[{'PASS' if g['physical'] < g['logical'] else 'FAIL'}]")
    ok = g["ok_tvd"] and g["ok_feat"] and g["physical"] < g["logical"]

    # ---- gate 2b: width constant in batch size B
    widths_B = []
    for B in B_scan:
        w = compiled_width_of_batch(rfqrc_step_circuits(B, cfg, rng))
        widths_B.append(w)
        print(f"  B={B:3d}: logical {B * n_q:3d} -> physical {w}")
    const_B = len(set(widths_B)) == 1
    print(f"width constant in B [{'PASS' if const_B else 'FAIL'}]")
    ok &= const_B

    # ---- gate 2c: width vs MEMORY HORIZON, both protocols, recomputed live
    print("\nwidth vs memory horizon T (the structural claim):")
    print(f"  {'T':>3s} {'windowed physical':>18s} {'rfqrc-batch physical':>21s}")
    widths_rf, widths_win = [], []
    wparams = make_reservoir_params(max(T_scan) + 1, max(T_scan), seed=SEED)
    for T in T_scan:
        # windowed protocol: ONE circuit encodes the last T inputs; the
        # stage-5 measured pathology physical = T+1 (n_qubits = T+1 wide,
        # shallow brickwork)
        wc = build_qrc_circuit(T + 1, T, rng.uniform(0, 1, T),
                               make_reservoir_params(T + 1, T, seed=SEED))
        ir_w = compiled_width_of_batch([wc])
        # rfqrc: memory horizon lives in the classical leak; the batch
        # of step circuits does not change with T at all -- recompute a
        # B=4 batch to show the invariance explicitly
        ir_r = compiled_width_of_batch(rfqrc_step_circuits(4, cfg, rng))
        widths_win.append(ir_w)
        widths_rf.append(ir_r)
        print(f"  {T:3d} {ir_w:18d} {ir_r:21d}")
    grows = all(b > a for a, b in zip(widths_win, widths_win[1:]))
    const_T = len(set(widths_rf)) == 1
    print(f"windowed grows with T [{'PASS' if grows else 'FAIL'}], "
          f"rfqrc constant in T [{'PASS' if const_T else 'FAIL'}]")
    ok &= grows and const_T

    # ---- topology scan (reported, not gated in --check)
    print("\ncompression vs within-block entangler topology (B=4):")
    from qiskit import transpile
    for ent in ("brickwork_zz", "haar_control"):
        c2 = RFQRCConfig(**{**cfg.__dict__, "entangler": ent})
        blocks = rfqrc_step_circuits(4, c2, rng)
        if ent == "haar_control":
            # the frozen parser has no 'unitary' gate; decompose the Haar
            # block to supported basis gates (u, cx) -- cones unchanged
            blocks = [transpile(b, basis_gates=["u", "cx"],
                                optimization_level=0) for b in blocks]
        w = compiled_width_of_batch(blocks)
        print(f"  {ent:14s}: logical {4 * n_q} -> physical {w}"
              + ("   (all-to-all block: no within-block compression;"
                 " cross-block reuse only)" if ent == "haar_control" else ""))

    print(f"\nexit: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
