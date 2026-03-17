"""Test that both mechanisms produce valid ReleaseBundle objects."""

from __future__ import annotations

import numpy as np

from rpbench.config import PrivacySpec
from rpbench.mechanisms.rp_ndis import MechRP
from rpbench.mechanisms.baselines.blocki12_jl import Blocki12JL
from rpbench.mechanisms.base import ReleaseBundle


def _make_data(n=100, d=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = X @ rng.randn(d) + rng.randn(n) * 0.1
    A = np.column_stack([X, y])
    pub_meta = {
        "n": n, "d": d, "d_aug": d + 1,
        "l": float(np.max(np.linalg.norm(A, axis=1))),
    }
    return A, pub_meta


def test_mech_rp_release():
    A, meta = _make_data()
    mech = MechRP(r=24)
    ps = PrivacySpec(epsilon=1.0, delta=1e-4)
    mech.calibrate(ps, meta)
    rb = mech.release(A, seed=42)

    assert isinstance(rb, ReleaseBundle)
    assert rb.mechanism_name == "Mech_RP"
    assert rb.xtx_hat is not None
    assert rb.xty_hat is not None
    assert rb.xtx_hat.shape == (meta["d"], meta["d"])
    assert rb.xty_hat.shape == (meta["d"],)
    assert rb.runtime_sec >= 0
    assert "lambda_ridge" in rb.calibration


def test_blocki_jl_release():
    A, meta = _make_data()
    mech = Blocki12JL(r=24)
    ps = PrivacySpec(epsilon=1.0, delta=1e-4)
    mech.calibrate(ps, meta)
    rb = mech.release(A, seed=42)

    assert isinstance(rb, ReleaseBundle)
    assert rb.mechanism_name == "Blocki12_JL"
    assert rb.xtx_hat is not None
    assert rb.xty_hat is not None
    assert rb.xtx_hat.shape == (meta["d"], meta["d"])
    assert rb.xty_hat.shape == (meta["d"],)
    assert rb.runtime_sec >= 0
    assert "w" in rb.calibration


def test_both_same_shape():
    A, meta = _make_data()
    ps = PrivacySpec(epsilon=2.0, delta=1e-4)

    rp = MechRP(r=24)
    rp.calibrate(ps, meta)
    rb_rp = rp.release(A, seed=0)

    jl = Blocki12JL(r=24)
    jl.calibrate(ps, meta)
    rb_jl = jl.release(A, seed=0)

    assert rb_rp.xtx_hat.shape == rb_jl.xtx_hat.shape
    assert rb_rp.xty_hat.shape == rb_jl.xty_hat.shape
