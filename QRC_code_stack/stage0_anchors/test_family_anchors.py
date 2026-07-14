"""Stage-4 anchors: <r> hits the analytic RMT limits; shuffling the
spectrum destroys the GOE signature (sabotage); dense_from_terms agrees
with hand-built matrices and rejects non-Hermitian input."""

import numpy as np
import pytest

from hamiltonians import (GOE_R, POISSON_R, PAULI, dense_from_terms,
                          level_spacing_ratio)


def _goe(dim, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim))
    return (A + A.T) / 2


def test_goe_anchor():
    r = np.mean([level_spacing_ratio(_goe(1500, s)) for s in (0, 1)])
    assert abs(r - GOE_R) < 0.02


def test_poisson_anchor():
    rng = np.random.default_rng(0)
    r = np.mean([
        level_spacing_ratio(np.diag(np.sort(rng.uniform(size=1500))))
        for _ in range(2)])
    assert abs(r - POISSON_R) < 0.02


def test_uncorrelated_spectrum_loses_goe():
    """Sabotage: replace the GOE spectrum by uncorrelated levels (a true
    Poisson process, iid exponential gaps) -> <r> must leave the GOE value
    and land at the Poisson one. An anchor that cannot fail is decoration."""
    rng = np.random.default_rng(3)
    fake = np.diag(np.cumsum(rng.exponential(size=1500)))
    r = level_spacing_ratio(fake)
    assert abs(r - GOE_R) > 0.05
    assert abs(r - POISSON_R) < 0.03   # single realisation: sigma ~ 0.01


def test_dense_from_terms_single_z():
    H = dense_from_terms([(2.0, {0: "Z"})], 1)
    assert np.allclose(H, 2.0 * PAULI["Z"])


def test_dense_from_terms_zz_two_qubit():
    H = dense_from_terms([(1.0, {0: "Z", 1: "Z"})], 2)
    assert np.allclose(H, np.diag([1.0, -1.0, -1.0, 1.0]))


def test_dense_from_terms_rejects_non_hermitian():
    with pytest.raises(ValueError):
        dense_from_terms([(1.0j, {0: "Z"})], 1)  # imaginary coefficient
