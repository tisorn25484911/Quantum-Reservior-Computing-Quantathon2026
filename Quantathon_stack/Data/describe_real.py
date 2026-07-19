"""describe_real.py -- measure every file in real/ so the README quotes facts.

Prints one block per dataset: file size, rows, columns, time span, sample
spacing, missing fraction, longest gap, and the value range of each numeric
column. Everything in the README's tables comes from this script's output.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REAL = Path(__file__).resolve().parent / "real"

TIME_COL = {"timestamp", "date", "time"}


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def describe(path):
    size = path.stat().st_size
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if c.lower() in TIME_COL), None)
    print(f"\n=== {path.name} ===")
    print(f"  size        {human(size)} ({size} bytes)")
    print(f"  rows x cols {df.shape[0]} x {df.shape[1]}")

    if tcol:
        t = pd.to_datetime(df[tcol], errors="coerce", format="mixed")
        t = t.dropna()
        span_days = (t.max() - t.min()).days
        print(f"  span        {t.min():%Y-%m-%d} -> {t.max():%Y-%m-%d} "
              f"({span_days} days = {span_days / 365.25:.1f} yr)")
        d = t.diff().dropna()
        if len(d):
            md = d.median()
            print(f"  time step   median {md} "
                  f"(min {d.min()}, max {d.max()})")
            # irregularity matters for embedding: a nominal daily series with
            # scattered 2-day steps is not the same as a gapless one
            off = (d != md).mean()
            print(f"  irregular   {100 * off:.2f}% of steps differ from median")

    for c in df.columns:
        if c == tcol:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        miss = s.isna()
        runs = (miss != miss.shift()).cumsum()[miss]
        longest = int(runs.value_counts().max()) if miss.any() else 0
        print(f"  {c:<22s} n={s.notna().sum():>7d}  missing={100 * miss.mean():5.2f}%"
              f"  longest_gap={longest:>6d} steps"
              f"  range=[{s.min():.4g}, {s.max():.4g}]  sd={s.std():.4g}")


def main():
    for p in sorted(REAL.glob("*.csv")):
        describe(p)
    total = sum(p.stat().st_size for p in REAL.glob("*"))
    print(f"\nTOTAL real/  {human(total)}")


if __name__ == "__main__":
    main()
