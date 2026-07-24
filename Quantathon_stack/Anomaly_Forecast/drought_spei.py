"""drought_spei.py -- the first honest drought (SPEI) forecast result.

Two panels, both from real runs, for the drought pitch:

  (left)  Recursive-rollout skill decay of the QRC on the Mae Klong SPEI-03
          drought index itself, against persistence and a size-matched ESN,
          averaged over 5 seeds. The honest headline: real skill at 1-2 months,
          materially beating persistence at every lead.
  (right) The SST/ENSO -> SPEI lead-lag: El Nino (positive ONI) leads toward
          Mae Klong drought (negative SPEI) with the physically correct sign,
          but a MODEST correlation -- the basis for folding SST in as a driver
          next, not for claiming ENSO is a strong single predictor.

    python drought_spei.py    # -> results/drought_spei.png + printed numbers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "DataBase_Analysis"))

from dataloader import load                              # noqa: E402

QRC = "#0072B2"; ESN = "#D55E00"; PERSIST = "#8c8c8c"; DROUGHT = "#8a5a00"


def _lead_lag(max_lead=12):
    spei, oni = load("maeklong_spei"), load("oni")
    s = pd.Series(spei.x, index=pd.DatetimeIndex(spei.index)).asfreq("MS")
    o = pd.Series(oni.x, index=pd.DatetimeIndex(oni.index)).asfreq("MS")
    df = pd.concat({"spei": s, "oni": o}, axis=1).dropna()
    ks = list(range(0, max_lead + 1))
    cc = [df["oni"].shift(k).corr(df["spei"]) for k in ks]
    return ks, cc, len(df), (df.index.min(), df.index.max())


def main():
    sweep = json.loads((_HERE / "results"
                        / "step2_seedsweep_maeklong_spei.json").read_text())
    hs = list(range(1, sweep["horizon"] + 1))
    mean, base = sweep["mean"], sweep["baselines"]
    best = sweep["best"]

    ks, cc, n_ov, span = _lead_lag()
    kbest = int(np.argmax(np.abs(cc)))

    print("=== First SPEI (drought) forecast -- Mae Klong SPEI-03, 5 seeds ===")
    for h in (1, 2, 3):
        print(f"  h={h}mo  QRC({best})={mean[best][h-1]:.3f}  "
              f"esn={mean['esn'][h-1]:.3f}  persist={base['persistence'][h-1]:.3f}")
    print(f"  usable lead: {max(sweep['verdicts'][best]['useful'])} months")
    print(f"\n=== ENSO -> SPEI lead-lag ({span[0].date()}..{span[1].date()}, "
          f"N={n_ov}) ===")
    print(f"  strongest |corr| = {cc[kbest]:+.3f} at ONI leading by {ks[kbest]} mo"
          f"  (El Nino -> drought; modest teleconnection)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    # --- left: SPEI skill decay ---
    a1.axhline(1.0, color="k", lw=0.9, ls="--", alpha=0.55)
    a1.text(hs[0], 1.02, "no skill (mean predictor)", fontsize=7.5, color="#444",
            va="bottom")
    a1.plot(hs, base["persistence"], color=PERSIST, ls=":", marker="o", ms=3,
            label="persistence")
    a1.plot(hs, mean["esn"], color=ESN, ls="--", marker="s", ms=3,
            label="ESN (size-matched)")
    a1.plot(hs, mean[best], color=QRC, marker="o", ms=4, lw=1.9,
            label=f"QRC ({best})")
    useful = sweep["verdicts"][best]["useful"]
    if useful:
        a1.axvspan(hs[0] - 0.3, max(useful) + 0.3, color=QRC, alpha=0.08)
        a1.text(max(useful), 0.05, f" usable lead = {max(useful)} mo", fontsize=8,
                color=QRC, va="bottom", ha="left")
    a1.set_xlabel("lead time (months)"); a1.set_ylabel("NMSE (lower is better)")
    a1.set_title("First SPEI forecast: skill vs lead\n"
                 "(Mae Klong SPEI-03, 115 yr, 5 seeds)", fontsize=9)
    a1.set_ylim(bottom=0); a1.legend(fontsize=8); a1.grid(True, color="#ddd", lw=.6)
    a1.set_axisbelow(True)

    # --- right: ENSO -> SPEI lead-lag ---
    a2.axhline(0, color="#888", lw=0.8)
    a2.bar(ks, cc, color=[DROUGHT if c < 0 else "#4a7" for c in cc], alpha=0.8)
    a2.set_xlabel("ONI leads SPEI by (months)")
    a2.set_ylabel("correlation(ONI$_{t-k}$, SPEI$_t$)")
    a2.set_title("The driver: El Niño → Mae Klong drought\n"
                 "correct sign, modest strength (peak "
                 f"{cc[kbest]:+.2f})", fontsize=9)
    a2.grid(True, color="#ddd", lw=.6, axis="y"); a2.set_axisbelow(True)
    a2.text(ks[kbest], cc[kbest] - 0.01, " El Niño loads drought",
            fontsize=7.5, color=DROUGHT, va="top")

    out = _HERE / "results" / "drought_spei.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
