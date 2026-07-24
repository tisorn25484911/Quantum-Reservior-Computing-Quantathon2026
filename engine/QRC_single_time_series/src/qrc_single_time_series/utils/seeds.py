"""seeds.py -- the five-seed registry (spec s31, G4)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class SeedBundle:
    reservoir_seed: int = 0   # Hamiltonian couplings / frozen reservoir
    model_seed: int = 0       # readout / NN init
    sampling_seed: int = 0    # finite-shot RNG
    noise_seed: int = 0       # noise-model RNG
    transpiler_seed: int = 0  # Qiskit transpiler

    def rng(self, which: str) -> np.random.Generator:
        return np.random.default_rng(getattr(self, which))

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = SeedBundle(reservoir_seed=7, model_seed=11, sampling_seed=101,
                     noise_seed=202, transpiler_seed=303)
