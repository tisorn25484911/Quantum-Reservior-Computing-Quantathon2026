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

# Zone shading (Okabe-Ito, kept distinct from the line hues above).
ZONE_TRAIN = "#0072B2"   # blue   -- the read-out was fitted here
ZONE_CALIB = "#E69F00"   # amber  -- the conformal radius was set here
ZONE_TEST = "#009E73"    # green  -- scored here; nothing above ever saw it
ZONE_WASH = "#9a9a9a"    # grey   -- washout, discarded

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


def _zone_overview(ax, x_full: np.ndarray, z: dict, ds: dict,
                   test_index: np.ndarray, n_detail: int) -> None:
    """Top strip: the whole series with train / calib / test zones shaded.

    This is the part that makes the split legible -- the detail panel below
    only ever shows the test window, so without this the training data the
    read-out actually learned from is invisible.
    """
    T = int(z["T"])
    ti = np.arange(T)
    ax.plot(ti, x_full, color="k", lw=0.7, alpha=0.75, zorder=3)

    wash, tr, ca, te = (z["washout"], z["train_end"],
                        z["calib_end"], z["test_end"])
    spans = [(0, wash, ZONE_WASH, 0.22, "washout"),
             (wash, tr, ZONE_TRAIN, 0.12, "TRAIN"),
             (tr, ca, ZONE_CALIB, 0.16, "CALIB"),
             (ca, te, ZONE_TEST, 0.16, "TEST")]
    ytop = float(np.nanmax(x_full))
    for a, b, colour, alpha, label in spans:
        if b <= a:
            continue
        ax.axvspan(a, b, color=colour, alpha=alpha, zorder=1, lw=0)
        ax.text((a + b) / 2, ytop, label, fontsize=7.5, ha="center",
                va="top", fontweight="bold", color=colour, zorder=5)

    # Mark how much of the test zone the detail panel below actually shows.
    if len(test_index) and n_detail < len(test_index):
        cut = int(test_index[n_detail - 1])
        ax.axvline(cut, color=ZONE_TEST, lw=0.9, ls=":", zorder=4)
        ax.text(cut, float(np.nanmin(x_full)), " detail below ends here",
                fontsize=6.8, color="#00614a", va="bottom", ha="left")

    unit = f" [{ds['unit']}]" if ds.get("unit") else ""
    ax.set_ylabel(f"{ds['name']}{unit}", fontsize=8)
    ax.set_title("Full series, split into zones -- the read-out was fitted on "
                 "TRAIN, its band set on CALIB, and it was scored on TEST "
                 "(which nothing upstream ever saw)", fontsize=8.5)
    ax.set_xlim(0, T)
    ax.margins(x=0)
    _grid(ax)


def forecast_png(result: dict, n_show: int = 240) -> bytes:
    """The headline figure: a zone overview on top, the test forecast below."""
    s = result["series"]
    y = np.asarray(s["y_true"], float)
    q = np.asarray(s["qrc"], float)
    lo = np.asarray(s["qrc_lo"], float)
    hi = np.asarray(s["qrc_hi"], float)
    e = np.asarray(s["esn"], float)
    p = np.asarray(s["persistence"], float)
    x_full = np.asarray(s.get("observed_full", []), float)
    test_index = np.asarray(s.get("test_index", []), int)
    z = result.get("zones")

    n = min(n_show, len(y))
    t = np.arange(n)
    ds = result["dataset"]
    unit = f" [{ds['unit']}]" if ds.get("unit") else ""
    nominal = result["conformal"]["nominal"]

    # Two panels when we have the zone context; fall back to one otherwise
    # (e.g. a result cached before zones were added).
    have_zones = bool(z) and x_full.size > 0
    if have_zones:
        fig, (axo, ax) = plt.subplots(
            2, 1, figsize=(10, 5.6),
            gridspec_kw={"height_ratios": [1.0, 2.3], "hspace": 0.5})
        _zone_overview(axo, x_full, z, ds, test_index, n)
    else:
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

    ax.set_xlabel(f"held-out TEST step (horizon = {result['config']['horizon']})")
    ax.set_ylabel(f"{ds['name']}{unit}")
    title = (f"TEST zone in detail -- {nominal:.0%} band, "
             f"showing {n} of {len(y)} test steps")
    ax.set_title(title, fontsize=9)
    ax.legend(ncol=5, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
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

    # Shade the two regimes: up to the origin the reservoir is driven by real
    # observations; after it, only step 1 uses real state -- every step from 2
    # on is driven by the model's OWN fed-back output. That self-driven zone is
    # exactly where forecast error compounds, so it earns a distinct colour.
    ax.axvspan(t_h[0] - 0.5, 0.5, color=ZONE_WASH, alpha=0.16, zorder=1, lw=0)
    ax.axvspan(0.5, len(truth) + 0.5, color=QRC, alpha=0.08, zorder=1, lw=0)

    ax.plot(t_h, hist, lw=1.3, color="#666", zorder=3,
            label="history (driven by observations)")
    ax.plot(t_f, truth, lw=2.6, color="k", alpha=0.24, solid_capstyle="round",
            zorder=3, label="observed (ground truth)")
    ax.plot(t_f, yhat, lw=1.8, color=QRC, marker="o", ms=3, zorder=4,
            label="recursive rollout (self-driven)")
    ax.axvline(0, color=QRC, lw=0.9, ls=":", zorder=2)
    ylo, yhi = ax.get_ylim()
    ax.text(0, yhi, " forecast origin", fontsize=7.5, color=QRC, va="top")
    ax.text(len(truth) / 2 + 0.5, ylo, "self-driven: own output fed back",
            fontsize=7, color=QRC, va="bottom", ha="center",
            fontstyle="italic")

    ds = r["dataset"]
    unit = f" [{ds['unit']}]" if ds.get("unit") else ""
    ax.set_xlabel(f"steps from origin ({ds['dt']:.4g} {ds['time_unit']} each)")
    ax.set_ylabel(f"{ds['name']}{unit}", fontsize=8)
    ax.set_title(f"one rollout from origin {ex['origin']} "
                 f"-- the reservoir drives on its own output after step 1")
    ax.legend(fontsize=8)
    _grid(ax)
    return _png(fig)


def freerun_png(r: dict) -> bytes:
    """The free-running forecast: train history, then a blind self-fed rollout
    across the whole test span, against persistence and the seasonal average."""
    s = r["series"]
    x = np.asarray(s["observed_full"], float)
    truth = np.asarray(s["truth"], float)
    qrc = np.asarray(s["qrc"], float)
    esn = np.asarray(s["esn"], float)
    persist = np.asarray(s["persistence"], float)
    clim = (np.asarray(s["climatology"], float)
            if s.get("climatology") is not None else None)
    origin = int(r["origin"])
    z = r["zones"]
    sc = r["scores"]
    ds = r["dataset"]
    unit = f" [{ds['unit']}]" if ds.get("unit") else ""

    ctx0 = int(s.get("context_start", max(0, origin - 3 * len(truth))))
    t_ctx = np.arange(ctx0, origin + 1)
    t_fut = np.arange(origin + 1, origin + 1 + len(truth))

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    # Shade: everything up to the boundary is real data the model trained on;
    # everything after is generated blind from the model's own output.
    ax.axvspan(ctx0 - 0.5, origin + 0.5, color=ZONE_TRAIN, alpha=0.07, lw=0)
    ax.axvspan(origin + 0.5, origin + len(truth) + 0.5, color=ZONE_TEST,
               alpha=0.09, lw=0)

    ax.plot(t_ctx, x[ctx0:origin + 1], color="#666", lw=0.9, zorder=3,
            label="TRAIN history (real, primes the state)")
    ax.plot(t_fut, truth, color="k", lw=2.6, alpha=0.28, solid_capstyle="round",
            zorder=3, label="TEST actual (never seen)")
    ax.plot(t_fut, persist, color=PERSIST, lw=1.0, ls=":", zorder=4,
            label=_lbl("persistence", sc.get("persistence")))
    if clim is not None:
        ax.plot(t_fut, clim, color="#009E73", lw=1.2, ls="--", zorder=5,
                label=_lbl("seasonal average", sc.get("climatology")))
    ax.plot(t_fut, esn, color=ESN, lw=1.1, alpha=0.85, zorder=5,
            label=_lbl("ESN free-run", sc.get("esn")))
    ax.plot(t_fut, qrc, color=QRC, lw=1.7, zorder=6,
            label=_lbl(f"QRC free-run ({r['config']['kind']})", sc.get("qrc")))

    ax.axvline(origin, color=QRC, lw=0.9, ls=":", zorder=2)
    ax.text(origin, ax.get_ylim()[1], " train/test boundary", fontsize=8,
            color=QRC, va="top")
    ax.text((origin + len(truth)), ax.get_ylim()[0],
            "blind: self-fed, no test data  ", fontsize=7.5, color="#00614a",
            va="bottom", ha="right", fontstyle="italic")

    yrs = r["free_years"]
    ax.set_xlabel(f"step (dt = {ds['dt']:.4g} {ds['time_unit']})")
    ax.set_ylabel(f"{ds['name']}{unit}")
    ax.set_title(f"{ds['name']} -- trained on {r['config']['train_frac']:.0%}, "
                 f"free-run the remaining {len(truth)} steps "
                 f"(~{yrs:.0f} {ds['time_unit']}) BLIND. "
                 "NMSE in legend; 1.0 = mean predictor.", fontsize=9)
    ax.legend(ncol=3, fontsize=7.6, loc="upper center",
              bbox_to_anchor=(0.5, -0.2))
    _grid(ax)
    return _png(fig)


def _lbl(name: str, nmse_val) -> str:
    return f"{name}  (NMSE {nmse_val:.2f})" if nmse_val is not None else name


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
