"""charts.py -- server-rendered PNG figures.

Matplotlib rather than a JS charting library: it keeps the page free of a build
step and reuses the plotting conventions the rest of the project already uses,
so a figure on screen and a figure in the logbook look like the same work.

Colours are the Okabe-Ito colourblind-safe set, the same palette validated for
``Main_run_Evaluation/mc_visualizer.py``. QRC keeps one hue everywhere; the
baselines are grey and dashed, because they are reference lines rather than
competing results and should not fight the forecast for attention.
"""

from __future__ import annotations

import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QRC = "#0072B2"        # blue  -- the quantum reservoir
ESN = "#D55E00"        # vermillion -- size-matched classical baseline
PERSIST = "#8c8c8c"    # grey -- persistence floor
BAND = "#0072B2"
GRID = "#dddddd"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _grid(ax):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def forecast_png(result: dict, n_show: int = 240) -> bytes:
    """The headline figure: truth, forecast, conformal band, and baselines."""
    s = result["series"]
    y = np.asarray(s["y_true"], float)
    q = np.asarray(s["qrc"], float)
    lo = np.asarray(s["qrc_lo"], float)
    hi = np.asarray(s["qrc_hi"], float)
    e = np.asarray(s["esn"], float)
    p = np.asarray(s["persistence"], float)

    n = min(n_show, len(y))
    t = np.arange(n)
    ds = result["dataset"]
    unit = f" [{ds['unit']}]" if ds.get("unit") else ""
    nominal = result["conformal"]["nominal"]

    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.fill_between(t, lo[:n], hi[:n], color=BAND, alpha=0.16, zorder=2,
                    label=f"{nominal:.0%} conformal band")
    ax.plot(t, y[:n], color="k", lw=2.6, alpha=0.24, solid_capstyle="round",
            zorder=3, label="observed")
    ax.plot(t, p[:n], color=PERSIST, lw=1.0, ls=":", zorder=4,
            label="persistence")
    ax.plot(t, e[:n], color=ESN, lw=1.1, ls="--", alpha=0.85, zorder=5,
            label="ESN (size-matched)")
    ax.plot(t, q[:n], color=QRC, lw=1.6, zorder=6, label="QRC forecast")

    ax.set_xlabel(f"held-out step (horizon = {result['config']['horizon']})")
    ax.set_ylabel(f"{ds['name']}{unit}")
    ax.set_title(f"{ds['name']} -- {nominal:.0%} band, "
                 f"showing {n} of {len(y)} test steps")
    ax.legend(ncol=5, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    _grid(ax)
    return _png(fig)


def scatter_png(result: dict) -> bytes:
    """Predicted against observed, with the 1:1 line -- bias shows up here.

    A model that tracks the shape but systematically under-predicts peaks looks
    fine on a time axis and obviously wrong here, which is why both views ship.
    """
    s = result["series"]
    y = np.asarray(s["y_true"], float)
    q = np.asarray(s["qrc"], float)
    e = np.asarray(s["esn"], float)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lim = [float(min(y.min(), q.min(), e.min())),
           float(max(y.max(), q.max(), e.max()))]
    ax.plot(lim, lim, color="k", lw=1.0, ls="--", alpha=0.5, zorder=2,
            label="perfect (1:1)")
    ax.scatter(y, e, s=10, color=ESN, alpha=0.4, zorder=3, edgecolors="none",
               label="ESN")
    ax.scatter(y, q, s=12, color=QRC, alpha=0.6, zorder=4, edgecolors="none",
               label="QRC")
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted")
    ax.set_title("calibration of the point forecast")
    ax.legend(fontsize=8, loc="upper left")
    _grid(ax)
    return _png(fig)


def horizon_png(sweep: dict) -> bytes:
    """Skill against forecast horizon -- where the model stops being useful.

    The mean-predictor line at NRMSE = 1 is drawn explicitly: a curve crossing
    it has stopped forecasting, and without the line that failure is invisible.
    """
    hs = sweep["horizons"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(1.0, color="k", lw=0.9, ls="--", alpha=0.55, zorder=2)
    ax.text(hs[0], 1.02, "mean predictor -- no skill", fontsize=7.5,
            color="#444444", va="bottom")
    for key, colour, ls, lbl in (("qrc", QRC, "-", "QRC"),
                                 ("esn", ESN, "--", "ESN (size-matched)"),
                                 ("persistence", PERSIST, ":", "persistence")):
        ax.plot(hs, sweep["nrmse"][key], marker="o", ms=4, color=colour,
                ls=ls, zorder=4, label=lbl)
    ax.set_xlabel("forecast horizon (steps ahead)")
    ax.set_ylabel("NRMSE  (lower is better)")
    ax.set_title(f"{sweep['dataset']['name']} -- skill decay with horizon")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    _grid(ax)
    return _png(fig)


def memory_png(result: dict) -> bytes:
    """Memory function of the configured reservoir against its ESN control."""
    q, e = result["qrc"], result["esn"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    floor = max(q["noise_floor"], e["noise_floor"])
    ax.axhspan(0, floor, color="#c9c9c9", alpha=0.55, zorder=1,
               label=f"noise floor ({floor:.3f})")
    ax.plot(q["delays"], q["mf"], marker="o", ms=4, color=QRC, zorder=3,
            label=f"QRC  (MC = {q['mc']:.2f})")
    ax.plot(e["delays"], e["mf"], marker="s", ms=4, color=ESN, ls="--",
            zorder=3, label=f"ESN, size-matched  (MC = {e['mc']:.2f})")
    ax.set_xlabel("delay $d$")
    ax.set_ylabel("memory function  $MF_d$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("linear memory capacity (i.i.d. drive)")
    ax.legend(fontsize=8)
    _grid(ax)
    return _png(fig)


def series_png(x: np.ndarray, meta: dict, n_show: int = 1500) -> bytes:
    """A dataset preview for the catalogue page."""
    x = np.asarray(x, float)[-int(n_show):]
    unit = f" [{meta['unit']}]" if meta.get("unit") else ""
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(np.arange(len(x)), x, lw=0.8, color=QRC)
    ax.set_xlabel(f"step  (dt = {meta['dt']:.4g} {meta['time_unit']})")
    ax.set_ylabel(f"{meta['name']}{unit}", fontsize=8)
    ax.set_title(f"{meta['name']} -- last {len(x)} of {meta['n']} samples",
                 fontsize=9)
    _grid(ax)
    return _png(fig)
