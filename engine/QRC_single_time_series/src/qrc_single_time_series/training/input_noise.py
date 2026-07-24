"""input_noise.py -- driving-input noise during teacher forcing (spec 22.3).

A DIFFERENT intervention from feature noise (spec s22.2): perturbing the driving
input changes which reservoir states are visited, so it is not equivalent to a
ridge penalty. Kept separate for the comparison in the phase-5/10 noise studies.
Implemented in P5.
"""
from __future__ import annotations

import numpy as np


def add_input_noise(inputs, sigma, rng, lo=0.0, hi=1.0):
    """Perturb the encoded driving inputs by uniform[-sigma,sigma], clipped to [lo,hi]."""
    u = np.asarray(inputs, dtype=float)
    return np.clip(u + rng.uniform(-sigma, sigma, u.shape), lo, hi)
