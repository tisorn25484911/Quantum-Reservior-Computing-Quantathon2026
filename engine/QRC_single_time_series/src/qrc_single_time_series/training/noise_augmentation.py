"""noise_augmentation.py -- reservoir-feature noise (noise~ridge study, spec s22.2).

Adding i.i.d. uniform[-sigma, sigma] noise to the reservoir FEATURES during the
readout solve is, in expectation, equivalent to Tikhonov (ridge) regularisation.
Under this repo's sum-of-squares loss, over L training SAMPLES (rows), each row
contributing E[(E_row . w)^2] = (sigma^2/3)||w||^2,

    E|| y - (X + E) w ||^2 = || y - X w ||^2 + L (sigma^2 / 3) || w ||^2

since Var(uniform[-s,s]) = s^2/3. Hence the expected-equivalent ridge penalty is
``lambda_eff = L * sigma^2 / 3`` with L the number of training ROWS (not the
feature count) -- "L" is the series length in this repo (cf. the MV+1 <= L/5 rule). This module provides that mapping and the
augmentation; the phase-5 script verifies the equivalence over ~50 noise draws and
reports the single-draw scatter. Implemented in P5.
"""
from __future__ import annotations

import numpy as np


def lambda_eff(sigma, n_samples):
    """Expected-equivalent ridge penalty for uniform[-sigma,sigma] feature noise.

    ``n_samples`` = number of training ROWS used in the solve (the series length L).
    """
    return float(n_samples * sigma ** 2 / 3.0)


def sigma_for_lambda(lam, n_samples):
    """Inverse: the feature-noise sigma whose expected ridge penalty is ``lam``."""
    return float(np.sqrt(3.0 * lam / n_samples))


def add_feature_noise(X, sigma, rng, skip_last_col=True):
    """Return ``X`` + uniform[-sigma,sigma] noise (bias column optionally untouched)."""
    E = rng.uniform(-sigma, sigma, X.shape)
    if skip_last_col:
        E[:, -1] = 0.0
    return X + E
