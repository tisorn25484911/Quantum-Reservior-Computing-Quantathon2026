"""Stage-3 anchors: noise channels are CPTP; p=0 model is noiseless;
a sabotaged (non-trace-preserving) channel is caught."""

import numpy as np
from qiskit.quantum_info import Kraus
from qiskit_aer.noise import depolarizing_error

from noise_models import channels_are_cptp, zero_noise_is_noiseless


def test_all_channels_cptp():
    assert channels_are_cptp()


def test_zero_noise_model_is_noiseless():
    assert zero_noise_is_noiseless()


def test_sabotaged_kraus_fails_cptp():
    """R1: an anchor that cannot fail is decoration. Break one Kraus
    normalisation and require the CPTP check to turn red."""
    err = depolarizing_error(0.01, 1)
    kraus_ops = Kraus(err.to_quantumchannel()).data
    broken = [k.copy() for k in kraus_ops]
    broken[0] = broken[0] * 1.3          # violates sum K^dag K = I
    assert not Kraus(broken).is_cptp()
