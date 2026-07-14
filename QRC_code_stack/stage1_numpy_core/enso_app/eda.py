"""eda.py -- QC + exploratory analysis for the ENSO anomaly task.

Prints every number it plots (playbook rule). Figures -> figures/,
script-generated only. Run from repo root:  python code/eda.py
"""

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf

import datasets
import figstyle
from baselines import EVENT_THRESHOLD, HORIZON, L

import matplotlib.pyplot as plt

FIGDIR = datasets.REPO / "figures"
ACF_LAGS = 48
MONTH_ABBR = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def qc(d):
    sst, y, n_tr = d["sst"], d["y"], d["n_train_months"]
    print("\n-- QC --")
    print(f"sst: min={sst.min():.2f} max={sst.max():.2f} "
          f"mean={sst.mean():.2f} std={sst.std():.2f} C")
    print(f"anomaly train span: mean={y[:n_tr].mean():+.4f} "
          f"std={y[:n_tr].std():.4f} min={y[:n_tr].min():+.3f} "
          f"max={y[:n_tr].max():+.3f}")
    print(f"anomaly test span:  mean={y[n_tr:].mean():+.4f} "
          f"std={y[n_tr:].std():.4f} min={y[n_tr:].min():+.3f} "
          f"max={y[n_tr:].max():+.3f}")
    below, above = (y[n_tr:] < d["scaler_range"][0]).sum(), \
        (y[n_tr:] > d["scaler_range"][1]).sum()
    print(f"test months clipped by train-fit scaler: {below} below, {above} above")


def seasonal(d):
    month, sst, y, clim = d["month"], d["sst"], d["y"], d["clim"]
    astd = np.array([y[month == m + 1].std() for m in range(12)])
    print("\n-- seasonal cycle (train-only climatology) --")
    print("month:      " + " ".join(f"{m:>6s}" for m in datasets.MONTH_COLS))
    print("clim  [C]:  " + " ".join(f"{c:6.2f}" for c in clim))
    print("anom std:   " + " ".join(f"{s:6.3f}" for s in astd))
    print(f"anomaly-std phase locking: max month "
          f"{datasets.MONTH_COLS[int(astd.argmax())]} ({astd.max():.3f}), "
          f"min month {datasets.MONTH_COLS[int(astd.argmin())]} ({astd.min():.3f})")

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for m in range(12):
        ax[0].plot(np.full((month == m + 1).sum(), m + 1), sst[month == m + 1],
                   ".", color="0.7", ms=2)
    ax[0].plot(range(1, 13), clim, "o-", color="C3", label="train climatology")
    ax[0].set(xlabel="calendar month", ylabel="SST [C]",
              title="seasonal cycle", xticks=range(1, 13),
              xticklabels=MONTH_ABBR)
    ax[0].legend(fontsize=7)
    ax[1].bar(range(1, 13), astd, color="C0")
    ax[1].set(xlabel="calendar month", ylabel="anomaly std [C]",
              title="anomaly variance by month", xticks=range(1, 13),
              xticklabels=MONTH_ABBR)
    figstyle.save(fig, FIGDIR / "enso_seasonal.png")


def distribution(d):
    y, n_tr = d["y"], d["n_train_months"]
    sk_full, sk_tr = stats.skew(y), stats.skew(y[:n_tr])
    ku = stats.kurtosis(y)
    ev = np.abs(y) > EVENT_THRESHOLD
    print("\n-- anomaly distribution --")
    print(f"skew: full={sk_full:+.3f} train={sk_tr:+.3f} "
          f"(handbook expectation ~ +1.15); excess kurtosis full={ku:+.3f}")
    print(f"event rate |anom|>{EVENT_THRESHOLD} C: "
          f"full={ev.mean():.3f} train={ev[:n_tr].mean():.3f} "
          f"test={ev[n_tr:].mean():.3f}")

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(y, bins=40, color="C0", alpha=0.85)
    for s in (-EVENT_THRESHOLD, EVENT_THRESHOLD):
        ax.axvline(s, color="C3", ls="--", lw=1)
    ax.set(xlabel="SST anomaly [C]", ylabel="months",
           title=f"anomaly distribution (skew {sk_full:+.2f})")
    figstyle.save(fig, FIGDIR / "enso_anomaly_hist.png")


def autocorrelation(d):
    y = d["y"]
    a = acf(y, nlags=ACF_LAGS, fft=True)
    print("\n-- anomaly autocorrelation --")
    for lag in (1, 2, 3, 6, 12, 24, 36, 48):
        note = "  <- horizon H" if lag == HORIZON else ""
        print(f"acf[{lag:2d}] = {a[lag]:+.3f}{note}")

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.stem(range(ACF_LAGS + 1), a, basefmt=" ")
    ax.axvline(HORIZON, color="C3", ls="--", lw=1, label=f"H={HORIZON}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(xlabel="lag [months]", ylabel="ACF", title="anomaly ACF")
    ax.legend(fontsize=7)
    figstyle.save(fig, FIGDIR / "enso_anomaly_acf.png")


def series(d):
    dates, y, n_tr = d["dates"], d["y"], d["n_train_months"]
    fig, ax = plt.subplots()
    ax.axvspan(dates[n_tr], dates[-1], color="C1", alpha=0.15,
               label="test span")
    ax.plot(dates, y, color="C0", lw=0.9)
    for s in (-EVENT_THRESHOLD, EVENT_THRESHOLD):
        ax.axhline(s, color="C3", ls="--", lw=0.8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(xlabel="date", ylabel="SST anomaly [C]",
           title="ENSO anomaly, train-only climatology")
    ax.legend(fontsize=7, loc="upper left")
    figstyle.save(fig, FIGDIR / "enso_anomaly_series.png")


if __name__ == "__main__":
    figstyle.apply()
    FIGDIR.mkdir(exist_ok=True)
    datasets.print_config()
    d = datasets.prepare()
    qc(d)
    seasonal(d)
    distribution(d)
    autocorrelation(d)
    series(d)
