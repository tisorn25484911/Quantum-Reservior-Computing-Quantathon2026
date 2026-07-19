"""compare.py -- one cross-dataset view of everything analysis.py measures.

`analysis.py` characterises datasets one at a time, which answers "what is this
series?" but not "which of these is worth putting a reservoir on?". That second
question needs every dataset on one axis, and the only way to get there is to
strip the units:

  * T_lambda converted to a common clock (days) so the numbers are comparable;
  * samples per T_lambda -- a *resolution* check, dimensionless;
  * record length in e-foldings -- a *statistics* check, dimensionless;
  * T_lambda divided by the dominant spectral period -- the *credibility*
    check, and the one that catches the failure mode running through this
    whole directory.

That last ratio is the important one. Rosenstein inflates whenever a strong
periodic component lets embedded neighbours phase-match, and when it does, the
exponent it reports is set by the cycle rather than by the dynamics. The tell
is T_lambda landing at roughly the dominant period. A ratio near 1 means the
estimate is probably measuring seasonal or diurnal phase decorrelation; well
away from 1 means the two are at least independent.

    python compare.py                  # figure + markdown report
    python compare.py --tier real      # restrict
    python compare.py --md report.md   # choose the report path
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis
import dataloader as dl
from visualizer import PALETTE, _grid

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"

# Everything is put on a common clock so bars can be compared. The chaotic tier
# has no physical clock at all -- its "natural" unit is whatever the ODE was
# non-dimensionalised in -- so it is excluded from the absolute-time panel and
# compared only on the dimensionless ones.
TO_DAYS = {"yr": 365.25, "day": 1.0, "h": 1.0 / 24.0, "min": 1.0 / 1440.0}

TIER_COLOR = {"real": PALETTE["series"], "surrogate": PALETTE["fit"],
              "chaotic": PALETTE["curve"]}


def collect(tier=None, method="auto", peaks=5):
    """Run the standard pass over every dataset and add comparison columns."""
    rows = []
    for key, s in dl.load_all(tier).items():
        try:
            r = analysis.analyse_one(s, method=method, peaks=peaks, save=False)
        except Exception as exc:                              # noqa: BLE001
            print(f"  {key}: FAILED -- {exc}")
            continue

        unit = r["time_unit"]
        scale = TO_DAYS.get(unit)
        lam, tl = r["lyap"], r["lyap_time"]

        r["t_lyap_days"] = tl * scale if (scale and np.isfinite(tl)) else None
        r["samples_per_tlyap"] = tl / r["dt"] if np.isfinite(tl) else None
        top = r["dominant_periods"]
        r["dominant_period"] = top[0]["period"] if top else None
        r["tlyap_over_period"] = (
            tl / top[0]["period"] if top and np.isfinite(tl) and top[0]["period"]
            else None)
        r["positive"] = lam > 0
        rows.append(r)
    return rows


# ----------------------------------------------------------------------
def _bar(ax, rows, values, title, xlabel, logx=True, ref=None, ref_label=None):
    keep = [(r, v) for r, v in zip(rows, values) if v is not None and v > 0]
    keep.sort(key=lambda rv: rv[1])
    if not keep:
        ax.set_axis_off()
        return
    labels = [r["key"] for r, _ in keep]
    vals = [v for _, v in keep]
    colors = [TIER_COLOR.get(r["tier"], "#888888") for r, _ in keep]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, zorder=3)
    ax.set_yticks(y, labels, fontsize=8)
    if logx:
        ax.set_xscale("log")
    if ref is not None:
        ax.axvline(ref, color="#444444", ls="--", lw=1.0, zorder=4)
        ax.text(ref, len(vals) - 0.4, f" {ref_label}", fontsize=7.5,
                color="#444444", va="top")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    _grid(ax)


def figure(rows, outdir=FIGDIR, dpi=140):
    """Four dimensionless-or-normalised comparisons on one sheet."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    _bar(axes[0, 0], rows, [r["t_lyap_days"] for r in rows],
         "(a) predictability horizon, common clock",
         "$T_\\lambda$ (days)  --  chaotic tier omitted, no physical clock")

    _bar(axes[0, 1], rows, [r["samples_per_tlyap"] for r in rows],
         "(b) resolution: samples per $T_\\lambda$",
         "samples per e-folding", ref=10, ref_label="10 = marginal")

    _bar(axes[1, 0], rows, [r["e_foldings_in_record"] for r in rows],
         "(c) statistics: record length in e-foldings",
         "$\\lambda_1 \\times$ span", ref=10, ref_label="10 = marginal")

    _bar(axes[1, 1], rows, [r["tlyap_over_period"] for r in rows],
         "(d) credibility: $T_\\lambda$ / dominant period",
         "ratio  --  near 1 means the exponent may be tracking the cycle",
         ref=1.0, ref_label="1 = suspect")

    handles = [plt.Line2D([], [], color=c, lw=6, label=t)
               for t, c in TIER_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Cross-dataset comparison  --  "
                 "panels (b)-(d) are dimensionless", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "comparison.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def spectra_figure(rows, tier="real", outdir=FIGDIR, dpi=140):
    """Every spectrum on a shared *normalised* frequency axis.

    Frequency is divided by each series' own dominant line, so a periodic
    record shows a spike at 1 with harmonics at 2, 3, ... while a broadband
    record does not. It makes "is there a single cycle running this series?"
    answerable at a glance across datasets on different clocks.
    """
    sel = [r for r in rows if r["tier"] == tier and r["dominant_period"]]
    if not sel:
        return None
    n = len(sel)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.0 * nrow), squeeze=False)

    for ax, r in zip(axes.ravel(), sel):
        s = dl.load(r["key"])
        import fourier
        spec = fourier.spectrum(s)
        f0 = 1.0 / r["dominant_period"]
        m = (spec.f > 0) & (spec.f <= 6 * f0)
        ax.plot(spec.f[m] / f0, spec.amp[m], color=PALETTE["spectrum"], lw=0.8)
        for k in (1, 2, 3):
            ax.axvline(k, color="#bbbbbb", lw=0.7, zorder=0)
        ax.set_title(f"{r['key']}  --  dominant {r['dominant_period']:.4g} "
                     f"{r['time_unit']}", fontsize=9, loc="left")
        ax.set_xlabel("frequency / dominant frequency", fontsize=8)
        ax.tick_params(labelsize=7.5)
        _grid(ax)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()

    fig.suptitle(f"{tier} tier -- spectra on each series' own frequency scale",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    outdir = Path(outdir)
    path = outdir / f"spectra_{tier}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
def _fmt(v, spec=".3g", dash="--"):
    return dash if v is None or not np.isfinite(v) else format(v, spec)


def markdown(rows, figs):
    """The report table -- same numbers as the figure, in text form."""
    out = ["# Cross-dataset comparison", "",
           "Generated by `compare.py`. Per-dataset figures and the method "
           "itself are documented in [analysis.md](analysis.md); the datasets "
           "themselves in [../Data/README.md](../Data/README.md).", ""]

    for f in figs:
        if f:
            out += [f"![{Path(f).stem}](figures/{Path(f).name})", ""]

    out += ["## All datasets", "",
            "| dataset | tier | N | dt | span | λ₁ | T_λ | T_λ (days) | "
            "samples/T_λ | e-foldings | dominant period | T_λ/period | method |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    for r in sorted(rows, key=lambda r: (r["tier"], r["key"])):
        u = r["time_unit"] or "smp"
        out.append(
            f"| `{r['key']}` | {r['tier']} | {r['n']:,} | "
            f"{_fmt(r['dt'], '.4g')} {u} | {_fmt(r['span'], '.4g')} {u} | "
            f"{r['lyap']:+.4f} /{u} | {_fmt(r['lyap_time'])} {u} | "
            f"{_fmt(r.get('t_lyap_days'))} | "
            f"{_fmt(r.get('samples_per_tlyap'), '.0f')} | "
            f"{_fmt(r['e_foldings_in_record'], '.0f')} | "
            f"{_fmt(r.get('dominant_period'))} {u} | "
            f"{_fmt(r.get('tlyap_over_period'), '.2f')} | {r['lyap_method']} |")

    out += ["", "## How to read this", "",
            "- **λ₁, T_λ** are in each dataset's own time unit; **T_λ (days)** "
            "puts the physical records on one clock. The chaotic tier has no "
            "physical clock, so that column is blank for it.",
            "- **samples/T_λ** is a resolution check. Below ~10 the divergence "
            "curve has too few points inside its scaling region to fit.",
            "- **e-foldings** is a statistics check: how many times the record "
            "could have lost its initial condition. Below ~10 there is little "
            "to average over.",
            "- **T_λ/period** near 1 is the warning sign. It means the "
            "predictability horizon coincides with the dominant cycle, which "
            "is what a phase-matching artefact looks like.",
            "- **method**: `rosenstein` rows are single-trajectory upper "
            "bounds, roughly factor-2. Only `ensemble` rows are measurements, "
            "and those carry ~31% mean absolute error. See analysis.md.", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=("chaotic", "real", "surrogate"))
    ap.add_argument("--method", default="auto",
                    choices=("auto", "ensemble", "rosenstein"))
    ap.add_argument("--md", default=str(HERE / "comparison.md"))
    a = ap.parse_args(argv)

    print("running the standard pass over every dataset ...")
    rows = collect(a.tier, method=a.method)
    if not rows:
        raise SystemExit("no datasets found")

    figs = [figure(rows), spectra_figure(rows, "real")]
    Path(a.md).write_text(markdown(rows, figs))
    for f in figs:
        if f:
            print(f"figure -> {f}")
    print(f"report -> {a.md}")
    return rows


if __name__ == "__main__":
    main()
