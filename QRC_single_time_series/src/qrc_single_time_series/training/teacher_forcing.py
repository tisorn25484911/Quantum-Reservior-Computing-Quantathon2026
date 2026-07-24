"""teacher_forcing.py -- teacher-forced one-step training (s_{k+1}=y_k).

Builds the reservoir feature matrix over a driven sequence and fits the linear
readout to predict the NEXT scaled value (FN teacher forcing / next-step
convention, G3). Everything is fit on TRAIN rows only; the returned readout is
then driven autonomously by the rollout engine (spec s16). Implemented in P5.
"""
from __future__ import annotations

import numpy as np

from ..data.windows import align
from ..models import readout as R
from ..evaluation import metrics as M


def train_teacher_forced(qrc, series, n_train, lam="gcv", washout=100,
                         feature_noise_sigma=0.0, noise_seed=0):
    """Fit a readout to predict series[k+1] from the reservoir features at step k.

    ``feature_noise_sigma`` optionally adds uniform[-s,s] reservoir-feature noise to
    the TRAIN features only (the noise~ridge study, spec s22.2). Returns
    ``(readout, info)`` with train/dev one-step NMSE.
    """
    s = np.asarray(series, dtype=float)
    X = qrc.features(s[:-1], check_budget=False)     # feature at step k
    y = s[1:]                                         # target = next value
    Xa, ya = align(X, y, horizon=0)                  # rows already paired k -> k+1

    Xtr, ytr = Xa[washout:n_train], ya[washout:n_train]
    if feature_noise_sigma > 0:
        rng = np.random.default_rng(noise_seed)
        noise = rng.uniform(-feature_noise_sigma, feature_noise_sigma, Xtr.shape)
        noise[:, -1] = 0.0                            # never perturb the bias column
        Xtr = Xtr + noise
    ro = R.fit(Xtr, ytr, lam=lam)

    info = {"nmse_train": M.nmse_variance(ytr, ro.predict(Xtr)),
            "lambda": float(ro.lam[0]), "rank": ro.rank}
    if n_train < len(Xa):
        Xdev, ydev = Xa[n_train:], ya[n_train:]
        info["nmse_dev_onestep"] = M.nmse_variance(ydev, ro.predict(Xdev))
    return ro, info
