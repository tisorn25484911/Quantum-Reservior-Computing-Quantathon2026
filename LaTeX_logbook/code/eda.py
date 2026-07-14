"""eda.py -- Exploratory data analysis for the three selected datasets.

Produces the Part VI figures and prints the descriptive statistics quoted in
the tables of the document. Methodology (overview -> seasonal structure ->
autocorrelation -> spectrum -> cross-correlation -> gaps) is dataset-agnostic.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.stattools import acf, pacf

import figstyle as fs
from datasets import solar_surrogate, load_surrogate, enso_real


def describe(name, s):
    s = s.dropna()
    q = s.quantile([0.05, 0.5, 0.95])
    print(f"[{name}] n={len(s)} mean={s.mean():.2f} sd={s.std():.2f} "
          f"min={s.min():.2f} p5={q.iloc[0]:.2f} med={q.iloc[1]:.2f} "
          f"p95={q.iloc[2]:.2f} max={s.max():.2f} skew={s.skew():.2f}")


# ------------------------------------------------------------------ solar
sol = solar_surrogate()
day = sol[sol.elevation_deg > 5]
describe("solar GHI (day)", day.ghi)
describe("solar kt  (day)", day.kt)
gaps = sol.ghi.isna()
print(f"[solar gaps] missing={gaps.sum()} ({100*gaps.mean():.2f}%), "
      f"longest run={int((gaps.astype(int).groupby((~gaps).cumsum()).sum()).max())} steps")

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
wk = sol.loc["2025-03-03":"2025-03-09"]
ax[0].plot(wk.index, wk.ghi_clear, color=fs.LIGHT, lw=0.9, label="clear-sky")
ax[0].plot(wk.index, wk.ghi, color=fs.ACCENT, lw=0.9, label="GHI")
ax[0].set_ylabel(r"irradiance (W\,m$^{-2}$)".replace("\\,", " "))
ax[0].set_title("(a) sample week", loc="left")
ax[0].legend(loc="upper right")
ax[0].tick_params(axis="x", rotation=45)
prof = day.assign(h=day.index.hour + day.index.minute / 60).groupby("h").ghi
for ql, qh in [(0.25, 0.75), (0.05, 0.95)]:
    ax[1].fill_between(prof.quantile(ql).index, prof.quantile(ql), prof.quantile(qh),
                       color=fs.ACCENT, alpha=0.18, lw=0)
ax[1].plot(prof.median().index, prof.median(), color=fs.ACCENT, lw=1.2)
ax[1].set_xlabel("hour of day"); ax[1].set_title("(b) diurnal quantiles", loc="left")
fs.save(fig, "eda_solar_overview")

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
kt_day = day.kt.to_numpy()
a = acf(kt_day, nlags=3 * 26, fft=True)          # ~26 daytime steps per day
ax[0].stem(np.arange(len(a)) * 0.5, a, linefmt=fs.ACCENT, markerfmt=" ", basefmt="k-")
ax[0].set_xlabel("lag (daytime hours)"); ax[0].set_ylabel("ACF of $k_t$")
ax[0].set_title("(a) clear-sky-index ACF", loc="left")
ghi_filled = sol.ghi.interpolate(limit=12).fillna(0).to_numpy()
f, p = signal.welch(ghi_filled, fs=48.0, nperseg=4096)   # cycles per day
ax[1].loglog(f[1:], p[1:], color="k", lw=0.8)
for c, lab in [(1, "24 h"), (2, "12 h"), (3, "8 h")]:
    ax[1].axvline(c, color=fs.LIGHT, lw=0.7, zorder=0)
    ax[1].text(c * 1.05, p.max() * 0.5, lab, fontsize=6.5, color=fs.GRAY)
ax[1].set_xlabel("frequency (cycles day$^{-1}$)"); ax[1].set_ylabel("PSD")
ax[1].set_title("(b) GHI power spectrum", loc="left")
fs.save(fig, "eda_solar_stats")

# ------------------------------------------------------------------- load
ld = load_surrogate()
describe("load (MW)", ld.load_mw)
print("[load] corr(load,temp)=%.2f corr(load,cdd)=%.2f" %
      (ld.load_mw.corr(ld.temp_c), ld.load_mw.corr(ld.cdd)))

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
wk = ld.loc["2024-06-03":"2024-06-16"]
ax[0].plot(wk.index, wk.load_mw, color=fs.ACCENT, lw=0.8)
ax[0].set_ylabel("load (MW)"); ax[0].set_title("(a) two sample weeks", loc="left")
ax[0].tick_params(axis="x", rotation=45)
prof = ld.assign(h=ld.index.hour, we=ld.index.dayofweek >= 5)
for we, lab, c in [(False, "weekday", fs.ACCENT), (True, "weekend", fs.GRAY)]:
    m = prof[prof.we == we].groupby("h").load_mw.mean()
    ax[1].plot(m.index, m, color=c, label=lab)
ax[1].set_xlabel("hour of day"); ax[1].legend()
ax[1].set_title("(b) mean diurnal profile", loc="left")
fs.save(fig, "eda_load_overview")

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
a = acf(ld.load_mw, nlags=24 * 8, fft=True)
ax[0].plot(np.arange(len(a)) / 24.0, a, color="k", lw=0.9)
for d in (1, 7):
    ax[0].axvline(d, color=fs.LIGHT, lw=0.7, zorder=0)
ax[0].set_xlabel("lag (days)"); ax[0].set_ylabel("ACF")
ax[0].set_title("(a) load autocorrelation", loc="left")
feat = pd.DataFrame({
    "load": ld.load_mw, "temp": ld.temp_c, "CDD": ld.cdd,
    "hr sin": np.sin(2 * np.pi * ld.index.hour / 24),
    "hr cos": np.cos(2 * np.pi * ld.index.hour / 24),
    "wknd": (ld.index.dayofweek >= 5).astype(float)})
C = feat.corr().to_numpy()
im = ax[1].imshow(C, cmap="gray_r", vmin=-1, vmax=1)
ax[1].set_xticks(range(len(feat.columns)), feat.columns, rotation=45, ha="right")
ax[1].set_yticks(range(len(feat.columns)), feat.columns)
for i in range(len(C)):
    for j in range(len(C)):
        ax[1].text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                   fontsize=6, color="white" if abs(C[i, j]) > 0.55 else "black")
ax[1].set_title("(b) feature correlations", loc="left")
fig.colorbar(im, ax=ax[1], shrink=0.8)
fs.save(fig, "eda_load_stats")

# ------------------------------------------------------------------- ENSO
en = enso_real()
describe("ENSO SST (C)", en.sst_c)
describe("ENSO anomaly", en.anomaly_c)

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3), width_ratios=[1.7, 1])
ax[0].plot(en.index, en.anomaly_c, color="k", lw=0.7)
ax[0].axhline(0.5, color=fs.LIGHT, lw=0.7); ax[0].axhline(-0.5, color=fs.LIGHT, lw=0.7)
ax[0].fill_between(en.index, 0.5, en.anomaly_c.where(en.anomaly_c > 0.5),
                   color=fs.ACCENT, alpha=0.5, lw=0)
ax[0].fill_between(en.index, -0.5, en.anomaly_c.where(en.anomaly_c < -0.5),
                   color=fs.GRAY, alpha=0.5, lw=0)
ax[0].set_ylabel(r"SST anomaly ($^\circ$C)")
ax[0].set_title("(a) Ni\u00f1o-region SST anomaly, 1950--2010", loc="left")
clim = en.sst_c.groupby(en.index.month).mean()
ax[1].plot(clim.index, clim, color=fs.ACCENT, marker="o", ms=2.5)
ax[1].set_xlabel("month"); ax[1].set_ylabel(r"SST ($^\circ$C)")
ax[1].set_title("(b) climatology", loc="left")
fs.save(fig, "eda_enso_overview")

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
a = acf(en.anomaly_c, nlags=48, fft=True)
ax[0].stem(range(len(a)), a, linefmt="k-", markerfmt=" ", basefmt="k-")
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_xlabel("lag (months)"); ax[0].set_ylabel("ACF")
ax[0].set_title("(a) anomaly ACF", loc="left")
f, p = signal.periodogram(en.anomaly_c.to_numpy(), fs=12.0)  # cycles/yr
ax[1].semilogx(1 / f[1:], p[1:] * f[1:], color="k", lw=0.9)  # variance-preserving
ax[1].axvspan(2, 7, color=fs.ACCENT, alpha=0.15, lw=0)
ax[1].text(3.0, (p[1:] * f[1:]).max() * 0.9, "ENSO band\n2--7 yr", fontsize=6.5, color=fs.GRAY)
ax[1].set_xlabel("period (years)"); ax[1].set_ylabel(r"$f\,S(f)$")
ax[1].set_title("(b) variance-preserving spectrum", loc="left")
fs.save(fig, "eda_enso_stats")
print("EDA complete.")
