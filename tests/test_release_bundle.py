"""Test that both mechanisms produce valid ReleaseBundle objects."""

from __future__ import annotations

import numpy as np

from rpbench.config import PrivacySpec
from rpbench.mechanisms.rp_ndis import MechRP, MechRPPois
from rpbench.mechanisms.baselines.blocki12_jl import Blocki12JL
from rpbench.mechanisms.base import ReleaseBundle
from rpbench.tasks.ols_from_release import OLSFromRelease
from rpbench.datasets.autompg import AutoMPGAdapter
from rpbench.config import SplitSpec, PreprocessSpec


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


def test_mech_rp_pois_release():
    A, meta = _make_data()
    mech = MechRPPois(r=24, q=0.5)
    ps = PrivacySpec(epsilon=1.0, delta=1e-4)
    mech.calibrate(ps, meta)
    rb = mech.release(A, seed=42)

    assert isinstance(rb, ReleaseBundle)
    assert rb.mechanism_name == "Mech_RP_Pois"
    assert rb.xtx_hat is not None
    assert rb.xty_hat is not None
    assert rb.xtx_hat.shape == (meta["d"], meta["d"])
    assert rb.xty_hat.shape == (meta["d"],)
    assert rb.runtime_sec >= 0
    assert rb.calibration["q"] == 0.5
    assert "subsample_size" in rb.diagnostics


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


def test_ols_from_release_stabilizes_indefinite_xtx():
    task = OLSFromRelease()
    rb = ReleaseBundle(
        mechanism_name="test",
        release_kind="gram_blocks",
        xtx_hat=np.array([[2.0, 0.0], [0.0, -1.0]]),
        xty_hat=np.array([4.0, 7.0]),
    )

    beta = task.fit_from_release(rb, train_meta={})

    np.testing.assert_allclose(beta, np.array([2.0, 0.0]))


def test_ols_from_release_uses_rp_sketch_decoder():
    task = OLSFromRelease()
    rb = ReleaseBundle(
        mechanism_name="Mech_RP",
        release_kind="gram_blocks",
        xtx_hat=np.zeros((2, 2)),
        xty_hat=np.zeros(2),
        sketch_matrix=np.array([
            [1.0, 0.0],
            [0.0, 2.0],
            [5.0, 8.0],
        ]),
        calibration={"lambda_ridge": 1.0},
    )

    beta = task.fit_from_release(rb, train_meta={})

    np.testing.assert_allclose(beta, np.array([5.0, 4.0]))


def test_autompg_public_clipping_bounds_are_enforced():
    bundle = AutoMPGAdapter().load(
        SplitSpec(),
        PreprocessSpec(clip_x=True, clip_x_bound=2.5, clip_y=True, clip_y_bound=1.5),
    )
    x_train_norms = np.linalg.norm(bundle.X_train, axis=1)
    x_test_norms = np.linalg.norm(bundle.X_test, axis=1)

    assert float(np.max(x_train_norms)) <= 2.5 + 1e-9
    assert float(np.max(x_test_norms)) <= 2.5 + 1e-9
    assert float(np.max(np.abs(bundle.y_train))) <= 1.5 + 1e-9
    assert float(np.max(np.abs(bundle.y_test))) <= 1.5 + 1e-9

    pub_meta = AutoMPGAdapter().public_meta(bundle)
    np.testing.assert_allclose(pub_meta["l"], np.sqrt(2.5 ** 2 + 1.5 ** 2))
