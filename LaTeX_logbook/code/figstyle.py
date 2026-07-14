"""figstyle.py -- shared publication style: mostly-black, single muted accent."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT = "#1f4e79"     # muted dark blue (single accent colour)
GRAY = "#666666"
LIGHT = "#bbbbbb"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})


def save(fig, name):
    fig.savefig(f"../figures/{name}.pdf")
    plt.close(fig)
    print("wrote figures/" + name + ".pdf")
