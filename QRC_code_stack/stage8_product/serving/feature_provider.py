"""feature_provider.py -- L3: the forecast core's ONLY coupling to
quantum execution (Part XI listing; Phase 9).

    class FeatureProvider(Protocol):
        def features(self, window, cfg) -> np.ndarray

Implementations:
    SimProvider       exact Aer/statevector snapshots via stage-6
                      rfqrc_reservoir (production default at n <= 11 --
                      the exact simulator IS the production backend:
                      fast, deterministic, identical in distribution to
                      ideal hardware; stated plainly, no marketing).
    CachedProvider    memo keyed by (config hash, quantised window).
                      Valid because RF-QRC step features are STATELESS
                      in the input window; the leak is applied
                      downstream of the provider. Cache-hit accounting
                      is part of the cost axis (Phase 9 acceptance).
    HardwareProvider  stub wired through the stage-7 batcher and the
                      frozen stage-5 compiler; gated by the
                      self-calibrated TVD equivalence check as CI.
                      Raises until a backend is configured -- the
                      hardware path is a scaling option, not a prop.

Requires stage5_qubit_reuse and stage6_rfqrc on PYTHONPATH.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from rfqrc_reservoir import (RFQRCConfig, make_entangler_params,
                             step_features_exact)


class FeatureProvider(Protocol):
    def features(self, window: np.ndarray, cfg: RFQRCConfig) -> np.ndarray:
        ...


class SimProvider:
    """Exact snapshots (stage 6, frozen interface)."""

    def __init__(self):
        self._params_cache: dict[str, dict] = {}

    def _params(self, cfg: RFQRCConfig) -> dict:
        key = _cfg_hash(cfg)
        if key not in self._params_cache:
            self._params_cache[key] = make_entangler_params(cfg)
        return self._params_cache[key]

    def features(self, window: np.ndarray, cfg: RFQRCConfig) -> np.ndarray:
        return step_features_exact(np.atleast_2d(window), cfg,
                                   self._params(cfg))


def _cfg_hash(cfg: RFQRCConfig) -> str:
    return hashlib.sha256(repr(sorted(cfg.__dict__.items()))
                          .encode()).hexdigest()[:12]


class CachedProvider:
    """Memoising wrapper. Windows are quantised to `decimals` before
    keying, trading a bounded encoding perturbation (< 10^-decimals,
    far below shot noise for decimals >= 3) for cache hits on revisited
    conditions -- exactly the memoisability RF-QRC buys by being
    stateless (Part XI)."""

    def __init__(self, inner: FeatureProvider, decimals: int = 3):
        self.inner = inner
        self.decimals = decimals
        self._cache: dict[tuple, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def features(self, window: np.ndarray, cfg: RFQRCConfig) -> np.ndarray:
        w = np.round(np.atleast_2d(window), self.decimals)
        key = (_cfg_hash(cfg), w.tobytes(), w.shape)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        out = self.inner.features(w, cfg)
        self._cache[key] = out
        return out

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / n if n else 0.0}


class HardwareProvider:
    """Reuse-compiled batched execution (stage 7 -> stage 5), CI-gated.

    Wiring exists (qreuse_batch.compile_batch + the TVD gate of
    exp_rfqrc_reuse); a physical backend does not. Raises with the
    integration recipe until one is configured."""

    def __init__(self, backend=None):
        self.backend = backend

    def features(self, window: np.ndarray, cfg: RFQRCConfig) -> np.ndarray:
        raise NotImplementedError(
            "HardwareProvider: no backend configured. Recipe: (1) build "
            "step circuits via stage-6 build_step_circuit(measure=True); "
            "(2) batch+compile via stage-7 qreuse_batch.compile_batch; "
            "(3) run on the backend; (4) unbatch_counts -> "
            "counts_to_features; (5) CI gate = the self-calibrated TVD "
            "equivalence check of exp_rfqrc_reuse.py against SimProvider "
            "before ANY hardware feature enters production.")
