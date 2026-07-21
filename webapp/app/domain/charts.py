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


def chaos_png(a: dict) -> bytes:
    """Divergence curve with the fitted window, plus the amplitude spectrum.

    Drawing the fit *over the window actually used* keeps the dominant error
    source visible: lambda_1 is a slope read off a hand-chosen region, and a
    badly placed window is obvious here while being invisible in the single
    number it produces.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    ax = axes[0]
    t = np.asarray(a["t"], float)
    curve = np.asarray(a["curve"], float)
    ax.plot(t, curve, lw=1.2, color=QRC, zorder=3, label="ln separation")
    lo, hi = a["window"]
    if hi > lo and np.isfinite(a["lambda1"]):
        tt = t[lo:hi + 1]
        ax.plot(tt, curve[lo] + a["lambda1"] * (tt - tt[0]), lw=1.9, ls="--",
                color=ESN, zorder=4,
                label=f"fit: $\\lambda_1$ = {a['lambda1']:.4g}")
        ax.axvspan(t[lo], t[hi], color=ESN, alpha=0.10, zorder=1)
    unit = a["dataset"]["time_unit"]
    ax.set_xlabel(f"time [{unit}]")
    ax.set_ylabel("ln separation")
    ax.set_title(f"divergence -- {a['method']}")
    ax.legend(fontsize=7.5, loc="lower right")
    _grid(ax)

    ax = axes[1]
    f = np.asarray(a["spectrum_f"], float)
    amp = np.asarray(a["spectrum_amp"], float)
    ax.plot(f, amp, lw=0.7, color=ESN, zorder=3)

    # Crop to where the signal actually is. On a fast record almost all the
    # amplitude sits in the first percent of the band, and an uncropped axis
    # shows a spike at the origin and nothing else.
    fmax = f.max() if len(f) else 1.0
    if len(f) > 1 and amp.sum() > 0:
        cum = np.cumsum(amp) / amp.sum()
        i99 = int(np.searchsorted(cum, 0.99))
        fmax = float(f[min(i99, len(f) - 1)]) * 1.5 or fmax
    ax.set_xlim(0, fmax)

    # Mark the dominant periods but do not label them in-chart: they cluster at
    # low frequency and the text overprints. The page carries an exact table.
    shown = [p for p in a["dominant_periods"][:3] if p["f"] <= fmax]
    for p in shown:
        ax.axvline(p["f"], color="k", lw=0.6, ls=":", alpha=0.55, zorder=2)
    if shown:
        ax.plot([], [], color="k", lw=0.6, ls=":", alpha=0.55,
                label="dominant periods")
        ax.legend(fontsize=7.5, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"frequency [cycles/{unit}]")
    ax.set_ylabel("amplitude")
    ax.set_title("amplitude spectrum")
    _grid(ax)

    fig.tight_layout()
    return _png(fig)


def anomaly_skill_png(r: dict) -> bytes:
    """Rollout skill decay against the two floors, with the ceiling marked."""
    hs = r["horizons"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axhline(1.0, color="k", lw=0.9, ls="--", alpha=0.55, zorder=2)
    ax.text(hs[0], 1.02, "mean predictor -- no skill", fontsize=7.5,
            color="#444", va="bottom")

    for key, colour, ls, lbl in (
            ("reservoir", QRC, "-", f"QRC ({r['config']['kind']})"),
            ("esn", ESN, "--", "ESN (size-matched)"),
            ("persistence", PERSIST, ":", "persistence")):
        ax.plot(hs, r["nmse"][key], marker="o", ms=3.5, color=colour, ls=ls,
                lw=1.8 if key == "reservoir" else 1.2, zorder=4, label=lbl)

    if r["useful_horizons"]:
        hi = max(r["useful_horizons"])
        ax.axvspan(hs[0] - 0.3, hi + 0.3, color=QRC, alpha=0.08, zorder=1)
        ax.text(hi, 0.04, f" usable lead = {hi}", fontsize=7.5, color=QRC,
                va="bottom", ha="right")

    hmax = r.get("H_max_steps")
    if hmax and np.isfinite(hmax) and hs[0] <= hmax <= hs[-1]:
        ax.axvline(hmax, color="#888", lw=1.0, ls="-.", zorder=2)
        ax.text(hmax, ax.get_ylim()[1] * 0.95, " $H_{max}$", fontsize=7.5,
                color="#666", va="top")

    ds = r["dataset"]
    ax.set_xlabel(f"lead time h (steps of {ds['dt']:.4g} {ds['time_unit']})")
    ax.set_ylabel("NMSE  (lower is better)")
    ax.set_title(f"{ds['name']} -- recursive-rollout skill decay "
                 f"({r['n_origins']} held-out origins)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    _grid(ax)
    return _png(fig)


def anomaly_example_png(r: dict) -> bytes:
    """One rollout against truth -- where the compounding becomes visible."""
    ex = r["example"]
    hist = np.asarray(ex["history"], float)
    truth = np.asarray(ex["truth"], float)
    yhat = np.asarray(ex["yhat"], float)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    t_h = np.arange(-len(hist) + 1, 1)
    t_f = np.arange(1, len(truth) + 1)
    ax.plot(t_h, hist, lw=1.3, color="#666", zorder=3, label="history (driven)")
    ax.plot(t_f, truth, lw=2.6, color="k", alpha=0.24, solid_capstyle="round",
            zorder=3, label="observed")
    ax.plot(t_f, yhat, lw=1.8, color=QRC, marker="o", ms=3, zorder=4,
            label="recursive rollout")
    ax.axvline(0, color=QRC, lw=0.9, ls=":", zorder=2)
    ax.text(0, ax.get_ylim()[1], " forecast origin", fontsize=7.5,
            color=QRC, va="top")

    ds = r["dataset"]
    unit = f" [{ds['unit']}]" if ds.get("unit") else ""
    ax.set_xlabel(f"steps from origin ({ds['dt']:.4g} {ds['time_unit']} each)")
    ax.set_ylabel(f"{ds['name']}{unit}", fontsize=8)
    ax.set_title(f"one rollout from origin {ex['origin']} "
                 f"-- the reservoir drives on its own output after step 1")
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
