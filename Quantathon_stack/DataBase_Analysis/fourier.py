"""fourier.py -- amplitude spectra on a linear scale.

Everything returned here is in physical units on linear axes: frequency in
cycles per `time_unit`, amplitude in the same unit as the series. No dB, no
power, no log-log. The reason is that the questions asked of this data are
"which period dominates, and how big is it in degrees C" -- both of which a
log axis makes harder to read, not easier.

Normalisation is one-sided and window-corrected, so a pure sinusoid of
amplitude A appears as a line of height A:

    x(t) = 3 * sin(2 pi 0.4 t)   ->   amp peaks at f = 0.4 with value ~3

    from fourier import fft_amplitude, dominant_periods
    f, amp = fft_amplitude(s.x, s.dt)
    peaks  = dominant_periods(f, amp, k=5)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class Spectrum:
    """One-sided amplitude spectrum, linear in both axes."""

    f: np.ndarray            # cycles per time_unit
    amp: np.ndarray          # same unit as the input series
    dt: float
    time_unit: str = ""
    unit: str = ""

    @property
    def period(self) -> np.ndarray:
        """Period in `time_unit`; f=0 maps to inf."""
        with np.errstate(divide="ignore"):
            return np.where(self.f > 0, 1.0 / np.where(self.f > 0, self.f, 1), np.inf)

    @property
    def f_nyquist(self) -> float:
        return 0.5 / self.dt


def fft_amplitude(x, dt=1.0, detrend="linear", window="hann") -> tuple:
    """One-sided FFT amplitude spectrum of `x`.

    Returns (f, amp) with `f` in cycles per unit of `dt` and `amp` scaled so a
    sinusoid's peak equals its amplitude.

    The mean is removed and a linear trend detrended before windowing: a
    non-zero mean puts all its energy in the f=0 bin and a drift smears power
    into the lowest few bins, either of which swamps the low-frequency
    structure that matters for climate series. The Hann window then suppresses
    the spectral leakage caused by a record that does not contain a whole
    number of cycles -- and the 2/sum(w) factor undoes the amplitude loss the
    window itself introduces.
    """
    x = np.asarray(x, float)
    if x.ndim != 1:
        raise ValueError("fft_amplitude takes a 1-D series")
    n = len(x)
    xd = signal.detrend(x, type=detrend) if detrend else x - x.mean()
    w = {"hann": np.hanning, "hamming": np.hamming,
         "none": np.ones}[window or "none"](n)
    amp = np.abs(np.fft.rfft(xd * w)) * 2.0 / w.sum()
    f = np.fft.rfftfreq(n, d=dt)
    return f, amp


def spectrum(series, **kw) -> Spectrum:
    """`fft_amplitude` applied to a dataloader.Series, units carried through."""
    f, amp = fft_amplitude(series.x, series.dt, **kw)
    return Spectrum(f=f, amp=amp, dt=series.dt,
                    time_unit=series.time_unit, unit=series.unit)


def welch_psd(x, dt=1.0, nperseg=None, detrend="linear") -> tuple:
    """Welch power spectral density -- the honest estimator of the noise floor.

    The raw FFT resolves sharp deterministic lines but its variance does not
    fall as the record lengthens, so a broadband floor looks like grass.
    Welch averages overlapping segments: resolution drops, variance falls, and
    the broadband continuum that distinguishes chaos from quasi-periodicity
    becomes readable. Returned as (f, psd), linear axes, unit^2 per frequency.
    """
    x = np.asarray(x, float)
    n = len(x)
    if nperseg is None:
        nperseg = min(n, max(64, 2 ** int(np.log2(max(n / 4, 64)))))
    return signal.welch(x, fs=1.0 / dt, nperseg=int(nperseg), detrend=detrend)


def dominant_periods(f, amp, k=5, fmin=None) -> list[dict]:
    """The `k` strongest spectral peaks, reported as periods.

    Local maxima only, so a single broad hump is reported once rather than as
    a run of adjacent bins. `fmin` drops everything below a frequency, which
    is how you exclude the residual near-DC bins on a short record.
    """
    f, amp = np.asarray(f, float), np.asarray(amp, float)
    keep = f > (fmin if fmin is not None else 0.0)
    idx, _ = signal.find_peaks(np.where(keep, amp, 0.0))
    if idx.size == 0:
        return []
    idx = idx[np.argsort(amp[idx])[::-1][:k]]
    return [{"f": float(f[i]),
             "period": float(1.0 / f[i]) if f[i] > 0 else np.inf,
             "amp": float(amp[i])} for i in idx]


def band_power(f, amp, lo, hi) -> float:
    """Summed squared amplitude in [lo, hi) -- a crude band energy."""
    m = (f >= lo) & (f < hi)
    return float(np.sum(amp[m] ** 2))


if __name__ == "__main__":
    # self-check: a known sinusoid must come back at the right height
    dt, n, a, f0 = 0.01, 4096, 3.0, 0.4
    t = np.arange(n) * dt
    f, amp = fft_amplitude(a * np.sin(2 * np.pi * f0 * t), dt)
    top = dominant_periods(f, amp, k=1)[0]
    print(f"injected  f={f0}  amp={a}")
    print(f"recovered f={top['f']:.4f}  amp={top['amp']:.4f}")
