"""mc_visualizer.py -- the four panels of a memory-capacity measurement.

Each ``plot_*`` draws onto an axis you supply, so panels compose into whatever
layout a caller wants; :func:`figure` is the standard 2x2 arrangement:

    (a) the memory function MF_d against delay, with the surrogate noise floor
    (b) memory capacity per stream, against its hard bound rank(X)
    (c) what MF_d actually measures -- the reconstruction at two delays
    (d) the feature covariance spectrum, i.e. where the read-out has room

Panel (a) is the one to read first: MC is the *area* under those curves above
the shaded floor, so the shape says what the single number cannot -- whether
memory decays gracefully, falls off a cliff, or never rises above noise. Panel
(b) is the honesty panel, since an MC that touches its rank bound is reporting
the size of the feature space rather than the dynamics.

Colours are the Okabe-Ito colourblind-safe set, checked for adjacent-pair
separation under deuteranopia and tritanopia. The two null controls are
deliberately drawn in grey and dashed: they are the floor, not results, and
should not compete for attention with the streams being measured.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # file output only; no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from memory_capacity import (
    delay_line_features,
    delayed_readout,
    esn_features,
    leaky_integrator_features,
    linear_memory_capacity,
    shuffled_features,
)

# Fixed hue order -- a stream keeps its colour across every panel, so identity
# never depends on which streams happen to be in the plot.
PALETTE = {"delay line": "#0072B2",         # blue
           "leaky integrators": "#D55E00",  # vermillion
           "ESN": "#009E73",                # bluish green
           "control": "#8c8c8c",            # nulls: recessive grey
           "floor": "#c9c9c9",
           "grid": "#dddddd"}
CONTROL_KEYS = ("shuffled", "noise", "control", "i.i.d.")

# Probe delays unpacked in panel (c): one short, one long enough that the loss
# of fidelity is plainly visible.
RECON_DELAYS = (1, 6)


def _colour(name: str) -> str:
    """Stream colour by name, falling back to the recessive control grey."""
    low = name.lower()
    if any(k in low for k in CONTROL_KEYS):
        return PALETTE["control"]
    for key, c in PALETTE.items():
        if key.lower() in low:
            return c
    return "#333333"


def _is_control(name: str) -> bool:
    return any(k in name.lower() for k in CONTROL_KEYS)


def _grid(ax):
    ax.grid(True, color=PALETTE["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------
def plot_memory_function(results, ax=None):
    """Panel (a): MF_d against delay, one line per stream.

    The shaded band is the surrogate noise floor -- the level a delay must
    clear to enter the sum. Drawing it makes the thresholding visible instead
    of leaving MC looking like an unexplained shortfall from the raw sum, and
    makes a dead reservoir obvious: its curve simply never leaves the band.
    """
    ax = ax or plt.gca()
    floor = max(r.noise_floor for r in results.values())
    ax.axhspan(0, floor, color=PALETTE["floor"], alpha=0.55, zorder=1,
               label=f"noise floor ({floor:.3f})")

    for name, res in results.items():
        ctrl = _is_control(name)
        ax.plot(res.delays, res.mf, marker="o", ms=3.5, lw=2.0 if not ctrl else 1.2,
                ls="--" if ctrl else "-", color=_colour(name), zorder=3,
                alpha=0.75 if ctrl else 1.0,
                label=f"{name}  (MC={res.mc:.2f})")

    ax.set_xlabel("delay $d$")
    ax.set_ylabel("memory function  $\\mathrm{MF}_d$")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("(a) memory function -- MC is the area above the floor")
    ax.legend(fontsize=7, loc="upper right")
    _grid(ax)
    return ax


def plot_capacity_bars(results, ax=None):
    """Panel (b): MC per stream against rank(X), its hard upper bound.

    The bound is drawn as a tick per bar rather than a single line, because it
    differs per stream. A bar reaching its tick is memory-saturated: the
    read-out has run out of feature space, so the measurement is reporting F,
    not the dynamics. Bars are annotated with the value since there are few
    enough to label directly.
    """
    ax = ax or plt.gca()
    names = list(results)
    mcs = [results[n].mc for n in names]
    ranks = [results[n].rank for n in names]
    ypos = np.arange(len(names))

    # Per-bar alpha has to ride on the RGBA colour; barh takes only a scalar.
    colours = [to_rgba(_colour(n), 0.55 if _is_control(n) else 1.0)
               for n in names]
    ax.barh(ypos, mcs, height=0.6, zorder=3, color=colours)
    for y, mc, rk in zip(ypos, mcs, ranks):
        ax.plot([rk, rk], [y - 0.34, y + 0.34], color="k", lw=1.6, zorder=4)
        ax.annotate(f"{mc:.2f}", xy=(mc, y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5)

    ax.plot([], [], color="k", lw=1.6, label="bound: rank$(X)$")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("memory capacity  MC")
    ax.set_xlim(0, max(max(ranks), max(mcs)) * 1.15)
    ax.set_title("(b) capacity against its bound")
    ax.legend(fontsize=7, loc="lower right")
    _grid(ax)
    return ax


def plot_reconstruction(X, u, name, ax=None, delays=RECON_DELAYS, max_delay=30,
                        n_show=70):
    """Panel (c): the read-out's reconstruction of the delayed input.

    MF_d is a squared correlation, which is easy to quote and hard to feel.
    This panel unpacks one stream: the true delayed input against what the
    fitted linear read-out recovered, on held-out data, at a short and a longer
    delay. The visible loss of fidelity between the two *is* the decay of MF_d
    in panel (a).
    """
    ax = ax or plt.gca()
    # The target goes down first as a broad, pale underlay so both read-outs
    # stay legible on top of it; drawn as a thin dark line it is simply hidden
    # by whichever reconstruction happens to be good.
    styles = [("-", 1.6, 1.0), ("--", 1.4, 0.85)]

    for (d, (ls, lw, alpha)) in zip(delays, styles):
        y_true, y_pred, mf = delayed_readout(X, u, d, max_delay=max_delay)
        t = np.arange(min(n_show, len(y_true)))
        if d == delays[0]:
            ax.plot(t, y_true[:len(t)], color="k", lw=3.2, alpha=0.22,
                    solid_capstyle="round", zorder=2,
                    label="true $u^{(k-d)}$")
        ax.plot(t, y_pred[:len(t)], ls=ls, lw=lw, alpha=alpha,
                color=_colour(name), zorder=3,
                label=f"read-out, $d$={d}  ($\\mathrm{{MF}}$={mf:.2f})")

    ax.set_xlabel("held-out step")
    ax.set_ylabel("input value")
    ax.set_title(f"(c) what $\\mathrm{{MF}}_d$ measures -- {name}")
    ax.legend(fontsize=7, loc="upper right", ncol=1)
    _grid(ax)
    return ax


def plot_feature_spectrum(results, features, ax=None, washout=60):
    """Panel (d): normalised covariance eigenvalues, largest first.

    A log axis over the sorted spectrum shows how fast the feature space runs
    out of independent directions. A steep cliff is exponential concentration:
    the nominal feature count is large, but the read-out has only a handful of
    usable directions, which caps memory regardless of how rich the dynamics
    look. The marker on each curve is that stream's effective rank.
    """
    ax = ax or plt.gca()
    for name, X in features.items():
        Xe = np.asarray(X, float)[washout:]
        Xc = Xe - Xe.mean(axis=0, keepdims=True)
        evals = np.sort(np.linalg.eigvalsh(np.cov(Xc, rowvar=False)))[::-1]
        evals = np.clip(evals, 1e-18, None)
        evals = evals / evals[0]
        idx = np.arange(1, len(evals) + 1)
        ctrl = _is_control(name)
        ax.semilogy(idx, evals, marker="o", ms=3, color=_colour(name),
                    lw=1.2 if ctrl else 1.8, ls="--" if ctrl else "-",
                    alpha=0.7 if ctrl else 1.0, zorder=3, label=name)
        er = results[name].effective_rank
        ax.axvline(er, color=_colour(name), lw=0.8, ls=":", alpha=0.6, zorder=2)

    ax.set_xlabel("feature-covariance eigenvalue index")
    ax.set_ylabel("normalised eigenvalue")
    ax.set_ylim(1e-8, 2.0)
    ax.set_title("(d) feature spectrum -- dotted line = effective rank")
    ax.legend(fontsize=7, loc="upper right")
    _grid(ax)
    return ax


# ----------------------------------------------------------------------
# Composite
# ----------------------------------------------------------------------
def figure(features, u, max_delay=30, focus=None, outdir=None, show=False,
           dpi=140, name="memory_capacity"):
    """The 2x2 figure for a dict of ``{label: feature matrix}`` under drive ``u``.

    ``focus`` names the stream unpacked in panel (c); it defaults to the
    highest-capacity non-control stream, which is usually the one being argued
    about. Saves to ``outdir/<name>.png`` when given, and returns the path.
    """
    results = {k: linear_memory_capacity(X, u, max_delay=max_delay)
               for k, X in features.items()}

    if focus is None:
        # Pick the stream that best *shows decay*, not the highest-MC one: a
        # perfect delay line reconstructs both probe delays exactly, so it
        # makes panel (c) three coincident curves that demonstrate nothing.
        real = [k for k in features if not _is_control(k)] or list(features)
        d1, d2 = RECON_DELAYS
        spread = {k: results[k].mf[d1 - 1] - results[k].mf[d2 - 1]
                  for k in real}
        focus = max(spread, key=spread.get)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_memory_function(results, axes[0, 0])
    plot_capacity_bars(results, axes[0, 1])
    plot_reconstruction(features[focus], u, focus, axes[1, 0],
                        max_delay=max_delay)
    plot_feature_spectrum(results, features, axes[1, 1])

    fig.suptitle(f"Linear memory capacity   (T={len(u)}, "
                 f"delays 1-{max_delay}, held-out ridge read-out)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = None
    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{name}.png"
        fig.savefig(path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path, results


def main() -> int:
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, size=4000)

    features = {
        "delay line (L=8)": delay_line_features(u, 8),
        "leaky integrators": leaky_integrator_features(u),
        "ESN (20 nodes)": esn_features(u, n_nodes=20),
        "ESN shuffled (control)": shuffled_features(esn_features(u, n_nodes=20)),
        "i.i.d. noise (control)": rng.normal(size=(len(u), 20)),
    }

    outdir = Path(__file__).resolve().parent / "figures"
    path, results = figure(features, u, max_delay=30, outdir=outdir)

    print(f"wrote {path}")
    for name, res in results.items():
        flag = "  <- " + "; ".join(res.warnings) if res.warnings else ""
        print(f"  {name:26s} MC={res.mc:6.3f}  rank={res.rank:3d}  "
              f"eff.rank={res.effective_rank:5.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
