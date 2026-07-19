"""visualizer.py -- the four panels, and the figure that stacks them.

Each `plot_*` draws onto an axis you supply, so panels compose into whatever
layout a caller wants; `figure()` is the standard 2x2 arrangement used by
analysis.py:

    (a) the series in physical time
    (b) its amplitude spectrum, linear in both axes
    (c) the divergence curve with the fitted scaling region marked
    (d) the same series with time rescaled to Lyapunov units, t_lambda = t/T_l

Panel (d) is the design-relevant one. On that axis, one unit is one e-folding
of initial error, so the readable question becomes "how many units wide is the
horizon I want to forecast?" -- and past t_lambda ~ 1 a reservoir is fighting
the dynamics, not the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # file output only; no display needed
import matplotlib.pyplot as plt

PALETTE = {"series": "#1f77b4", "spectrum": "#d62728",
           "curve": "#2ca02c", "fit": "#ff7f0e", "grid": "#dddddd"}


def _grid(ax):
    ax.grid(True, color=PALETTE["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------
def plot_series(series, ax=None, use_dates=True, **kw):
    """Panel (a): the raw record, in whatever form it arrived.

    Real/surrogate series are drawn against their actual timestamps when
    available -- a calendar axis is what makes an ENSO record legible -- while
    chaotic systems fall back to elapsed natural time units.
    """
    ax = ax or plt.gca()
    if use_dates and series.index is not None:
        ax.plot(series.index, series.x, lw=0.8, color=PALETTE["series"], **kw)
        ax.set_xlabel("date")
    else:
        ax.plot(series.t, series.x, lw=0.8, color=PALETTE["series"], **kw)
        ax.set_xlabel(f"time [{series.time_unit or 'sample'}]")
    ax.set_ylabel(f"{series.name}" + (f" [{series.unit}]" if series.unit else ""))
    ax.set_title("(a) series")
    _grid(ax)
    return ax


def plot_spectrum(spec, ax=None, peaks=None, fmax=None, **kw):
    """Panel (b): amplitude spectrum, linear frequency and linear amplitude.

    `fmax` crops the axis. Worth using on the fast records -- half-hourly
    solar has a Nyquist of 1 cycle/hour, and everything interesting lives in
    the first percent of that range, so an uncropped axis shows a spike at the
    origin and nothing else.
    """
    ax = ax or plt.gca()
    ax.plot(spec.f, spec.amp, lw=0.7, color=PALETTE["spectrum"], **kw)
    if fmax:
        ax.set_xlim(0, fmax)
    else:
        ax.set_xlim(0, spec.f_nyquist)
    ax.set_ylim(bottom=0)
    for p in (peaks or []):
        if p["f"] <= (fmax or spec.f_nyquist):
            ax.axvline(p["f"], color="k", lw=0.5, ls=":", alpha=0.6)
            ax.annotate(f"{p['period']:.3g} {spec.time_unit}",
                        xy=(p["f"], p["amp"]), xytext=(3, 3),
                        textcoords="offset points", fontsize=7)
    u = spec.time_unit or "sample"
    ax.set_xlabel(f"frequency [cycles/{u}]")
    ax.set_ylabel(f"amplitude" + (f" [{spec.unit}]" if spec.unit else ""))
    ax.set_title("(b) amplitude spectrum (linear-linear)")
    _grid(ax)
    return ax


def plot_divergence(est, ax=None):
    """Panel (c): the divergence curve with the fitted window highlighted.

    The fit line is drawn over the window actually used, so a bad window is
    visible rather than hidden inside a single reported number -- which
    matters here, since window placement is the dominant error source.
    """
    ax = ax or plt.gca()
    ax.plot(est.t, est.curve, lw=1.0, color=PALETTE["curve"], label="ln separation")
    lo, hi = est.window
    if hi > lo and np.isfinite(est.lyap):
        tt = est.t[lo:hi + 1]
        c0 = est.curve[lo]
        ax.plot(tt, c0 + est.lyap * (tt - tt[0]), lw=1.8, ls="--",
                color=PALETTE["fit"],
                label=f"fit: $\\lambda_1$={est.lyap:.4f}")
        ax.axvspan(est.t[lo], est.t[hi], color=PALETTE["fit"], alpha=0.10)
    ax.set_xlabel(f"time [{est.time_unit or 'sample'}]")
    ax.set_ylabel("ln separation")
    ax.set_title(f"(c) divergence -- {est.method}")
    ax.legend(fontsize=7, loc="lower right")
    _grid(ax)
    return ax


def plot_lyapunov_time(series, est, ax=None, max_units=None):
    """Panel (d): the series against t_lambda = lambda_1 * t.

    Gridlines every e-folding. A non-positive exponent leaves nothing to
    rescale by, so the panel says so instead of drawing a misleading axis.
    """
    ax = ax or plt.gca()
    if not np.isfinite(est.lyap) or est.lyap <= 0:
        ax.text(0.5, 0.5, "no positive $\\lambda_1$\n(no predictability horizon)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(d) Lyapunov time")
        return ax
    tl = est.lyap * series.t
    m = tl <= max_units if max_units else np.ones_like(tl, bool)
    ax.plot(tl[m], series.x[m], lw=0.8, color=PALETTE["series"])
    for k in range(1, int(tl[m].max()) + 1):
        ax.axvline(k, color="k", lw=0.4, alpha=0.25)
    ax.set_xlabel("Lyapunov time  $t_\\lambda = \\lambda_1 t$  [e-foldings]")
    ax.set_ylabel(f"{series.name}" + (f" [{series.unit}]" if series.unit else ""))
    ax.set_title(f"(d) $T_\\lambda$ = {est.lyap_time:.3g} {est.time_unit or 'sample'}"
                 f"  ({tl.max():.0f} e-foldings in record)")
    _grid(ax)
    return ax


# ----------------------------------------------------------------------
# Composite
# ----------------------------------------------------------------------
def figure(series, spec, est, peaks=None, fmax=None, max_units=None,
           outdir=None, show=False, dpi=140):
    """The 2x2 figure. Saves to `outdir/<key>_analysis.png` when given."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    plot_series(series, axes[0, 0])
    plot_spectrum(spec, axes[0, 1], peaks=peaks, fmax=fmax)
    plot_divergence(est, axes[1, 0])
    plot_lyapunov_time(series, est, axes[1, 1], max_units=max_units)

    tag = f"{series.tier} | N={len(series.x)} | dt={series.dt:.4g} {series.time_unit}"
    fig.suptitle(f"{series.name}   ({tag})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = None
    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{series.key}_analysis.png"
        fig.savefig(path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path
