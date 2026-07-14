"""figstyle.py -- shared matplotlib style for all script-generated figures.

figures/ is script-generated only (CLAUDE.md); every figure goes through
save() so path + write are printed alongside the numbers being plotted.
"""

import sys

import matplotlib

if "ipykernel" not in sys.modules:      # scripts headless; notebooks inline
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.2,
        "figure.figsize": (7.0, 3.2),
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
