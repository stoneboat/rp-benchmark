"""Test that both mechanisms produce valid ReleaseBundle objects."""

from __future__ import annotations

import numpy as np

from rpbench.config import PrivacySpec
from rpbench.mechanisms.modified_gaussmix import MechModifiedGaussMix
from rpbench.mechanisms.rp_ndis import MechRP, MechRPPois, MechRPPTR
from rpbench.mechanisms.sheffet_rp import MechImprovedSheffetRP, MechSheffetRP
from rpbench.mechanisms.baselines.blocki12_jl import Blocki12JL
from rpbench.mechanisms.base import ReleaseBundle
from rpbench.metrics.release import relative_frobenius_xtx_normalized
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


def _make_autompg_bounded_data(n=100, d=5, seed=0):
    """Synthetic augmented data satisfying the Auto MPG public row bound."""
    A, _ = _make_data(n=n, d=d, seed=seed)
    l = float(np.sqrt(3.0 ** 2 + 1.0 ** 2))
    max_norm = float(np.max(np.linalg.norm(A, axis=1)))
    A = A * min(1.0, 0.9 * l / max_norm)
    pub_meta = {"n": n, "d": d, "d_aug": d + 1, "l": l}
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


def test_mech_sheffet_rp_release():
    A, meta = _make_data()
    mech = MechSheffetRP(r=24)
    ps = PrivacySpec(epsilon=1.0, delta=1e-4)
    mech.calibrate(ps, meta)
    rb = mech.release(A, seed=42)

    assert isinstance(rb, ReleaseBundle)
    assert rb.mechanism_name == "Mech_Sheffet_RP"
    assert rb.release_kind == "sketch"
    assert rb.xtx_hat is not None
    assert rb.xty_hat is not None
    assert rb.sketch_matrix is not None
    assert rb.xtx_hat.shape == (meta["d"], meta["d"])
    assert rb.xty_hat.shape == (meta["d"],)
    assert rb.sketch_matrix.shape == (meta["d"] + 1, 24)
    assert rb.runtime_sec >= 0
    assert "noise_var" in rb.calibration
    assert "released_clean_sketch" in rb.diagnostics


def test_mech_improved_sheffet_rp_release():
    A, meta = _make_data()
    mech = MechImprovedSheffetRP(r=24)
    ps = PrivacySpec(epsilon=1.0, delta=1e-4)
    mech.calibrate(ps, meta)
    rb = mech.release(A, seed=42)

    assert isinstance(rb, ReleaseBundle)
    assert rb.mechanism_name == "Mech_Improved_Sheffet_RP"
    assert rb.release_kind == "sketch"
    assert rb.xtx_hat is not None
    assert rb.xty_hat is not None
    assert rb.sketch_matrix is not None
    assert rb.xtx_hat.shape == (meta["d"], meta["d"])
    assert rb.xty_hat.shape == (meta["d"],)
    assert rb.sketch_matrix.shape == (meta["d"] + 1, 24)
    assert rb.runtime_sec >= 0
    assert rb.calibration["calibration_kind"] == "gaussmix_improved_sheffet"
    assert rb.calibration["threshold_core"] == rb.calibration["noise_var"] / 2.0
    assert "released_clean_sketch" in rb.diagnostics


def test_mech_modified_gaussmix_release():
    A, meta = _make_data()
    mech = MechModifiedGaussMix(r=24)
    ps = PrivacySpec(epsilon=1.0, delta=1e-4)
    mech.calibrate(ps, meta)
    rb = mech.release(A, seed=42)

    assert isinstance(rb, ReleaseBundle)
    assert rb.mechanism_name == "Mech_Modified_GaussMix"
    assert rb.release_kind == "sketch"
    assert rb.xtx_hat is not None
    assert rb.xty_hat is not None
    assert rb.sketch_matrix is not None
    assert rb.xtx_hat.shape == (meta["d"], meta["d"])
    assert rb.xty_hat.shape == (meta["d"],)
    assert rb.sketch_matrix.shape == (24, meta["d"] + 1)
    assert rb.runtime_sec >= 0
    assert rb.calibration["calibration_kind"] == "modified_gaussmix_full_dp"
    assert rb.calibration["aug_row_bound_sq"] == meta["l"] ** 2
    for key in (
        "branch",
        "epsilon",
        "delta",
        "r",
        "tau",
        "sigma_matrix",
        "lambda_tilde",
        "effective_noise_variance",
    ):
        assert key in rb.diagnostics


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


def test_normalized_covariance_metric_debiases_sheffet_sketch():
    xtx_true = np.diag([2.0, 5.0])
    noise_var = 3.0
    r = 4
    rb = ReleaseBundle(
        mechanism_name="Mech_Sheffet_RP",
        release_kind="sketch",
        xtx_hat=r * (xtx_true + noise_var * np.eye(2)),
        xty_hat=np.zeros(2),
        calibration={"r": r, "noise_var": noise_var},
        diagnostics={"released_clean_sketch": False, "noise_var": noise_var},
    )

    assert relative_frobenius_xtx_normalized(xtx_true, rb) == 0.0


def test_normalized_covariance_metric_debiases_gaussmix_sketch():
    xtx_true = np.array([[2.0, 0.5], [0.5, 5.0]])
    noise_var = 1.5
    r = 6
    rb = ReleaseBundle(
        mechanism_name="Mech_Modified_GaussMix",
        release_kind="sketch",
        xtx_hat=r * (xtx_true + noise_var * np.eye(2)),
        xty_hat=np.zeros(2),
        calibration={"r": r, "sigma_matrix": noise_var},
        diagnostics={"r": r, "effective_noise_variance": noise_var},
    )

    assert relative_frobenius_xtx_normalized(xtx_true, rb) == 0.0


def test_mech_rp_ptr_infeasible_test_falls_back_to_baseline(monkeypatch):
    A, meta = _make_autompg_bounded_data()
    delta = 1e-6
    ps = PrivacySpec(epsilon=2.0, delta=delta)

    ptr = MechRPPTR(r=24, tau=0.5, delta_r=5e-7, delta_t=3e-7, delta_ptr=2e-7)
    ptr.calibrate(ps, meta)
    cal = ptr.diagnostics()

    np.testing.assert_allclose(cal["epsilon_T"], 298.9320644028485, rtol=0.0, atol=1e-8)
    assert cal["epsilon_T"] > ps.epsilon
    assert cal["mode"] == "fallback_rp"
    assert cal["ptr_test_executed"] is False
    assert cal["fallback_reason"] == "epsilon_T > epsilon"
    assert "epsilon_R" not in cal
    assert cal["baseline_calibration"]["epsilon"] == ps.epsilon
    assert cal["baseline_calibration"]["delta"] == ps.delta
    assert cal["baseline_calibration"]["r"] == ptr.r

    baseline = MechRP(r=ptr.r)
    baseline.calibrate(ps, meta)

    def fail_private_eigenvalue_query(*args, **kwargs):
        raise AssertionError("fallback must not execute the private eigenvalue test")

    monkeypatch.setattr(np.linalg, "eigvalsh", fail_private_eigenvalue_query)
    rb = ptr.release(A, seed=42)
    rb_baseline = baseline.release(A, seed=42)

    assert isinstance(rb, ReleaseBundle)
    assert rb.mechanism_name == "Mech_RP_PTR"
    assert rb.release_kind == "gram_blocks"
    assert rb.xtx_hat is not None
    assert rb.xty_hat is not None
    assert rb.xtx_hat.shape == (meta["d"], meta["d"])
    assert rb.xty_hat.shape == (meta["d"],)
    assert rb.runtime_sec >= 0
    assert rb.sketch_matrix is not None
    assert rb.sketch_matrix.shape == (meta["d_aug"], 24)
    assert rb.diagnostics["mode"] == "fallback_rp"
    assert rb.diagnostics["ptr_test_executed"] is False
    for key in ("lambda_min_raw", "eta", "lambda_lb", "lambda_ptr"):
        assert key not in rb.diagnostics

    np.testing.assert_array_equal(rb.sketch_matrix, rb_baseline.sketch_matrix)
    np.testing.assert_array_equal(rb.xtx_hat, rb_baseline.xtx_hat)
    np.testing.assert_array_equal(rb.xty_hat, rb_baseline.xty_hat)

    task = OLSFromRelease()
    beta_ptr = task.fit_from_release(rb, train_meta=meta)
    beta_baseline = task.fit_from_release(rb_baseline, train_meta=meta)
    np.testing.assert_array_equal(beta_ptr, beta_baseline)
    assert task.evaluate(beta_ptr, A[:, :-1], A[:, -1]) == task.evaluate(
        beta_baseline, A[:, :-1], A[:, -1]
    )


def test_mech_rp_ptr_feasible_release_uses_exact_remaining_budget():
    A, meta = _make_autompg_bounded_data()
    delta = 1e-6
    mech = MechRPPTR(r=24, tau=30.0, delta_r=5e-7, delta_t=3e-7, delta_ptr=2e-7)
    ps = PrivacySpec(epsilon=2.0, delta=delta)
    mech.calibrate(ps, meta)

    cal = mech.diagnostics()
    assert cal["mode"] == "ptr"
    assert cal["ptr_test_executed"] is True
    assert cal["epsilon_T"] <= ps.epsilon
    assert cal["epsilon_R"] == ps.epsilon - cal["epsilon_T"]
    assert cal["epsilon_R"] > 0.0
    np.testing.assert_allclose(
        cal["delta_R"] + cal["delta_T"] + cal["delta_ptr"],
        delta,
        rtol=0.0,
        atol=1e-12,
    )

    rb = mech.release(A, seed=42)
    assert isinstance(rb, ReleaseBundle)
    assert rb.mechanism_name == "Mech_RP_PTR"
    assert rb.diagnostics["mode"] == "ptr"
    assert rb.diagnostics["ptr_test_executed"] is True
    # Required feasible-PTR diagnostics fields.
    for key in ("epsilon_T", "epsilon_R", "delta_R", "delta_T", "delta_ptr",
                "tau", "p_star", "p_star_ptr", "p_star_rp", "lambda_min_raw",
                "alpha", "eta", "lambda_lb", "lambda_ptr", "lambda_rp",
                "prop6_threshold", "prop6_condition_met", "lambda_ptr_lt_lambda_rp"):
        assert key in rb.diagnostics, f"missing diagnostics key: {key}"
    # Backward-compatible alias and baseline-vs-PTR distinction.
    assert rb.diagnostics["p_star"] == rb.diagnostics["p_star_ptr"]
    assert rb.diagnostics["p_star_rp"] >= rb.diagnostics["p_star_ptr"]
    # lambda_ptr >= 0; under the corrected baseline comparison it may be either
    # smaller or larger than the true full-budget baseline ridge.
    assert rb.diagnostics["lambda_ptr"] >= 0.0
    # Proposition 6 diagnostic reflects the stated raw-eigenvalue inequality.
    lhs = rb.diagnostics["lambda_min_raw"]
    rhs = rb.diagnostics["prop6_threshold"]
    assert rb.diagnostics["prop6_condition_met"] == bool(lhs > rhs)
    assert rb.diagnostics["lambda_ptr_lt_lambda_rp"] == bool(
        rb.diagnostics["lambda_ptr"] < rb.diagnostics["lambda_rp"]
    )


def test_mech_rp_ptr_deterministic():
    A, meta = _make_autompg_bounded_data()
    delta = 1e-6
    mech1 = MechRPPTR(r=24, tau=30.0, delta_r=5e-7, delta_t=3e-7, delta_ptr=2e-7)
    mech2 = MechRPPTR(r=24, tau=30.0, delta_r=5e-7, delta_t=3e-7, delta_ptr=2e-7)
    ps = PrivacySpec(epsilon=2.0, delta=delta)
    mech1.calibrate(ps, meta)
    mech2.calibrate(ps, meta)
    rb1 = mech1.release(A, seed=77)
    rb2 = mech2.release(A, seed=77)
    np.testing.assert_array_equal(rb1.xtx_hat, rb2.xtx_hat)
    np.testing.assert_array_equal(rb1.xty_hat, rb2.xty_hat)
    assert rb1.diagnostics["eta"] == rb2.diagnostics["eta"]
    assert rb1.diagnostics["lambda_ptr"] == rb2.diagnostics["lambda_ptr"]


def test_mech_modified_gaussmix_deterministic():
    A, meta = _make_data()
    ps = PrivacySpec(epsilon=1.5, delta=1e-4)
    mech1 = MechModifiedGaussMix(r=24)
    mech2 = MechModifiedGaussMix(r=24)
    mech1.calibrate(ps, meta)
    mech2.calibrate(ps, meta)
    rb1 = mech1.release(A, seed=77)
    rb2 = mech2.release(A, seed=77)

    np.testing.assert_array_equal(rb1.sketch_matrix, rb2.sketch_matrix)
    np.testing.assert_array_equal(rb1.xtx_hat, rb2.xtx_hat)
    np.testing.assert_array_equal(rb1.xty_hat, rb2.xty_hat)
    assert rb1.diagnostics["branch"] == rb2.diagnostics["branch"]


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
