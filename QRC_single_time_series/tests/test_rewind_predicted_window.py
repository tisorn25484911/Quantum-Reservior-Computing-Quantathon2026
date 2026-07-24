"""P4: rewind buffer provenance mask -- window fills with predictions over time."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qrc_single_time_series.models.recursive_forecaster import RewindBuffer  # noqa: E402


def test_warmup_window_is_all_real():
    rb = RewindBuffer(5)
    rb.warm(np.arange(20.0))
    assert not rb.predicted.any()
    assert np.array_equal(rb.values, np.arange(15.0, 20.0))


def test_provenance_fills_monotonically():
    rb = RewindBuffer(4)
    rb.warm(np.arange(10.0))
    fracs = [rb.provenance_fraction]
    for i in range(4):
        rb.push(100 + i, predicted=True)
        fracs.append(rb.provenance_fraction)
    assert fracs == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert rb.all_predicted


def test_short_history_is_left_padded():
    rb = RewindBuffer(6)
    buf = rb.warm(np.array([2.0, 3.0]))
    assert len(buf) == 6 and buf[-1] == 3.0 and buf[0] == 2.0


def test_push_drops_oldest():
    rb = RewindBuffer(3)
    rb.warm(np.array([1.0, 2.0, 3.0]))
    rb.push(4.0, predicted=True)
    assert np.array_equal(rb.values, [2.0, 3.0, 4.0])
    assert list(rb.predicted) == [False, False, True]


def test_real_rewind_model_driven_through_buffer():
    # P6: wire the real rewind reservoir through the provenance buffer -- as the
    # window fills with fed-back predictions, features stay finite and the mask
    # tracks provenance to fully-predicted.
    import numpy as np
    from qrc_single_time_series.quantum.hamiltonians import nn_tfi
    from qrc_single_time_series.quantum.rewind_qrc import RewindReservoir
    from qrc_single_time_series.models import readout as R
    from qrc_single_time_series.dynamical_systems import mackey_glass as MG
    from qrc_single_time_series.models.recursive_forecaster import RewindBuffer

    d = MG.generate(16, n=700, washout=600)["series"]
    tw = 6
    rw = RewindReservoir(nn_tfi(4, J=1.0, h=0.5, seed=7), t_w=tw, tau=1.0)
    X = rw.features_series(d)
    y = d[tw:]
    X = X[:len(y)]
    ro = R.fit(X[:400], y[:400], lam="gcv")

    buf = RewindBuffer(tw)
    buf.warm(d[:450])
    for _ in range(tw):
        feat = np.concatenate([rw.window_features(buf.values), [1.0]])
        yhat = float(ro.predict(feat[None, :])[0])
        assert np.isfinite(yhat)
        buf.push(yhat, predicted=True)
    assert buf.all_predicted
