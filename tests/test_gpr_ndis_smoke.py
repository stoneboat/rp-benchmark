"""Smoke tests for GPR + NDIS-Calibrated Gaussian Mechanism (Phase 2).

Covers:
  - RBFKernel: construction, evaluation, Gram matrix, diagonal = 1
  - GPRGaussianOutput: construction, fit shapes, posterior variance non-negative
  - GPRGaussianOutput.sensitivity: Proposition 8 formulas, tau=0 sentinel,
    decreasing in tau (larger tau → smaller sensitivity)
  - Calibration: delta_bar_cov with large rho_inf returns 1.0 (overflow guard)
  - NDISGaussianWrapper + GPR: calibrate, release shape, seed reproducibility
  - End-to-end synthetic data: full pipeline
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# RBFKernel tests
# ---------------------------------------------------------------------------

class TestRBFKernel:
    def test_constructible(self):
        from ndis_gaussian.gpr.model import RBFKernel
        k = RBFKernel(lengthscale=1.5)
        assert k.lengthscale == 1.5
        assert k.l_bound == 1.0  # k(x,x) = 1 always

    def test_diagonal_is_one(self):
        """k(x, x) = 1 for the RBF kernel, regardless of lengthscale."""
        from ndis_gaussian.gpr.model import RBFKernel
        rng = np.random.default_rng(0)
        for ls in [0.1, 1.0, 10.0]:
            kernel = RBFKernel(lengthscale=ls)
            x = rng.standard_normal(8)
            assert math.isclose(kernel(x, x), 1.0, rel_tol=1e-9)

    def test_symmetry(self):
        from ndis_gaussian.gpr.model import RBFKernel
        kernel = RBFKernel(lengthscale=1.0)
        rng = np.random.default_rng(1)
        x, y = rng.standard_normal(5), rng.standard_normal(5)
        assert math.isclose(kernel(x, y), kernel(y, x), rel_tol=1e-9)

    def test_gram_shape_and_diagonal(self):
        from ndis_gaussian.gpr.model import RBFKernel
        kernel = RBFKernel(lengthscale=1.0)
        X = np.random.default_rng(2).standard_normal((10, 4))
        K = kernel.gram(X)
        assert K.shape == (10, 10)
        np.testing.assert_allclose(np.diag(K), 1.0, rtol=1e-9)

    def test_gram_psd(self):
        from ndis_gaussian.gpr.model import RBFKernel
        kernel = RBFKernel(lengthscale=1.0)
        X = np.random.default_rng(3).standard_normal((8, 4))
        K = kernel.gram(X)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-8), f"K not PSD: min eigval={eigvals.min():.3e}"

    def test_k_star_shape(self):
        from ndis_gaussian.gpr.model import RBFKernel
        kernel = RBFKernel(lengthscale=1.0)
        rng = np.random.default_rng(4)
        X = rng.standard_normal((12, 5))
        x_star = rng.standard_normal(5)
        k_vec = kernel.k_star(X, x_star)
        assert k_vec.shape == (12,)

    def test_k_starstar_is_one(self):
        from ndis_gaussian.gpr.model import RBFKernel
        kernel = RBFKernel(lengthscale=2.0)
        x_star = np.ones(4)
        assert kernel.k_starstar(x_star) == 1.0

    def test_invalid_lengthscale(self):
        from ndis_gaussian.gpr.model import RBFKernel
        with pytest.raises(ValueError, match="lengthscale must be > 0"):
            RBFKernel(lengthscale=0.0)


# ---------------------------------------------------------------------------
# GPRGaussianOutput construction
# ---------------------------------------------------------------------------

class TestGPRConstruction:
    def test_constructible(self):
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        x_star = np.zeros(4)
        gpr = GPRGaussianOutput(
            x_star=x_star,
            kernel=RBFKernel(1.0),
            sigma_n2=0.1,
            B_bound=1.0,
            l_bound=1.0,
        )
        assert gpr.sigma_n2 == 0.1
        assert gpr.B_bound == 1.0
        assert gpr.l_bound == 1.0

    def test_invalid_sigma_n2(self):
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        with pytest.raises(ValueError, match="sigma_n2 must be > 0"):
            GPRGaussianOutput(np.zeros(2), RBFKernel(1.0), sigma_n2=0.0,
                              B_bound=1.0, l_bound=1.0)

    def test_invalid_B_bound(self):
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        with pytest.raises(ValueError, match="B_bound must be > 0"):
            GPRGaussianOutput(np.zeros(2), RBFKernel(1.0), sigma_n2=1.0,
                              B_bound=0.0, l_bound=1.0)


# ---------------------------------------------------------------------------
# GPRGaussianOutput.fit
# ---------------------------------------------------------------------------

class TestGPRFit:
    @pytest.fixture
    def small_gpr_data(self):
        rng = np.random.default_rng(7)
        n, d = 20, 4
        X = rng.standard_normal((n, d))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # unit rows
        # Simple regression: y ≈ sin(sum(x)) clipped to [-1, 1]
        y = np.sin(X.sum(axis=1))
        y = np.clip(y, -1.0, 1.0)
        x_star = rng.standard_normal(d)
        x_star = x_star / np.linalg.norm(x_star)
        return X, y, x_star, n, d

    def test_fit_output_shapes(self, small_gpr_data):
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        X, y, x_star, n, d = small_gpr_data
        gpr = GPRGaussianOutput(x_star=x_star, kernel=RBFKernel(1.0),
                                sigma_n2=0.1, B_bound=1.0, l_bound=1.0)
        go = gpr.fit(X, y)
        assert go.mu.shape == (1,), f"Expected mu shape (1,), got {go.mu.shape}"
        assert go.Sigma.shape == (1, 1), f"Expected Sigma (1,1), got {go.Sigma.shape}"

    def test_posterior_variance_nonnegative(self, small_gpr_data):
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        X, y, x_star, n, d = small_gpr_data
        gpr = GPRGaussianOutput(x_star=x_star, kernel=RBFKernel(1.0),
                                sigma_n2=0.1, B_bound=1.0, l_bound=1.0)
        go = gpr.fit(X, y)
        Sigma_star = float(go.Sigma[0, 0])
        assert Sigma_star >= 0.0, f"Posterior variance must be ≥ 0, got {Sigma_star}"

    def test_posterior_variance_bounded_by_prior(self, small_gpr_data):
        """Posterior variance ≤ prior variance k_★★ = 1 (Schur complement)."""
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        X, y, x_star, n, d = small_gpr_data
        gpr = GPRGaussianOutput(x_star=x_star, kernel=RBFKernel(1.0),
                                sigma_n2=0.1, B_bound=1.0, l_bound=1.0)
        go = gpr.fit(X, y)
        assert float(go.Sigma[0, 0]) <= 1.0 + 1e-9, \
            f"Posterior variance > prior k_★★ = 1: {go.Sigma[0,0]}"

    def test_posterior_variance_decreases_with_more_data(self):
        """Adding more training points should not increase posterior variance."""
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        rng = np.random.default_rng(9)
        d = 3
        x_star = rng.standard_normal(d)
        kernel = RBFKernel(1.0)

        X_small = rng.standard_normal((5, d))
        X_large = rng.standard_normal((50, d))
        y_small = rng.standard_normal(5)
        y_large = rng.standard_normal(50)

        gpr = GPRGaussianOutput(x_star=x_star, kernel=kernel,
                                sigma_n2=0.1, B_bound=5.0, l_bound=1.0)
        Sigma_small = float(gpr.fit(X_small, y_small).Sigma[0, 0])
        Sigma_large = float(gpr.fit(X_large, y_large).Sigma[0, 0])

        assert Sigma_large <= Sigma_small + 1e-6, \
            f"Variance with more data ({Sigma_large:.4f}) > with less ({Sigma_small:.4f})"

    def test_diagnostics_available_after_fit(self, small_gpr_data):
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        X, y, x_star, n, d = small_gpr_data
        gpr = GPRGaussianOutput(x_star=x_star, kernel=RBFKernel(1.0),
                                sigma_n2=0.1, B_bound=1.0, l_bound=1.0)
        gpr.fit(X, y)
        diag = gpr.diagnostics
        assert diag["n_train"] == n
        assert diag["d"] == d


# ---------------------------------------------------------------------------
# GPRGaussianOutput.sensitivity
# ---------------------------------------------------------------------------

class TestGPRSensitivity:
    @pytest.fixture
    def public_meta(self):
        return {"n": 100, "d": 1, "B": 1.0, "l": 1.0, "sigma_n2": 0.1}

    def test_sentinel_at_tau_zero(self, public_meta):
        """At tau=0, sensitivity returns sentinel with Delta=inf, rho_inf=0."""
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        gpr = GPRGaussianOutput(np.zeros(3), RBFKernel(1.0), 0.1, 1.0, 1.0)
        sens = gpr.sensitivity(tau=0.0, public_meta=public_meta)
        assert math.isinf(sens.Delta)
        assert sens.rho_inf == 0.0
        assert sens.nu == 0.0
        assert sens.tau == 0.0

    def test_sentinel_maps_to_delta_one(self, public_meta):
        """Sentinel at tau=0 should give delta_bar_LC = 1 (no privacy)."""
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        from ndis_gaussian.calibration import delta_bar_LC
        gpr = GPRGaussianOutput(np.zeros(3), RBFKernel(1.0), 0.1, 1.0, 1.0)
        sens = gpr.sensitivity(0.0, public_meta)
        # delta_bar_mean(eps, inf) = 1, delta_bar_cov(eps, 0, 0, 1) = 0
        val = delta_bar_LC(1.0, sens.Delta, sens.rho_inf, sens.nu, d=1)
        assert math.isclose(val, 1.0, rel_tol=1e-6)

    def test_proposition8_formula(self, public_meta):
        """Verify Proposition 8 formulas against hand-computed values."""
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        gpr = GPRGaussianOutput(np.zeros(3), RBFKernel(1.0), 0.1, 1.0, 1.0)
        n, B, l, sigma_n2 = 100, 1.0, 1.0, 0.1
        tau = 2.0

        expected_Delta = l * B * math.sqrt(n) / (math.sqrt(sigma_n2) * math.sqrt(tau))
        expected_rho = math.log((tau + l**2) / tau)
        expected_nu = expected_rho  # scalar output

        sens = gpr.sensitivity(tau, public_meta)
        assert math.isclose(sens.Delta, expected_Delta, rel_tol=1e-9)
        assert math.isclose(sens.rho_inf, expected_rho, rel_tol=1e-9)
        assert math.isclose(sens.nu, expected_nu, rel_tol=1e-9)
        assert sens.tau == tau

    def test_nu_equals_rho_inf(self, public_meta):
        """For scalar output (m=1), nu must equal rho_inf."""
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        gpr = GPRGaussianOutput(np.zeros(3), RBFKernel(1.0), 0.1, 1.0, 1.0)
        for tau in [0.01, 0.1, 1.0, 10.0]:
            sens = gpr.sensitivity(tau, public_meta)
            assert math.isclose(sens.nu, sens.rho_inf, rel_tol=1e-9), \
                f"nu != rho_inf at tau={tau}: {sens.nu} vs {sens.rho_inf}"

    def test_sensitivity_decreasing_in_tau(self, public_meta):
        """Delta and rho_inf must both decrease as tau increases."""
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        gpr = GPRGaussianOutput(np.zeros(3), RBFKernel(1.0), 0.1, 1.0, 1.0)
        taus = [0.01, 0.1, 1.0, 10.0, 100.0]
        deltas = [gpr.sensitivity(t, public_meta).Delta for t in taus]
        rhos = [gpr.sensitivity(t, public_meta).rho_inf for t in taus]
        assert all(deltas[i] > deltas[i+1] for i in range(len(deltas)-1)), \
            f"Delta not strictly decreasing: {deltas}"
        assert all(rhos[i] > rhos[i+1] for i in range(len(rhos)-1)), \
            f"rho_inf not strictly decreasing: {rhos}"


# ---------------------------------------------------------------------------
# Calibration: large rho_inf overflow guard
# ---------------------------------------------------------------------------

class TestCalibrationOverflowGuard:
    def test_delta_bar_cov_large_rho_inf_returns_one(self):
        """For rho_inf > 700, delta_bar_cov should return 1.0 (overflow guard)."""
        from ndis_gaussian.calibration import delta_bar_cov
        val = delta_bar_cov(1.0, rho_inf=750.0, nu=750.0, d=1)
        assert val == 1.0, f"Expected 1.0 for large rho_inf, got {val}"

    def test_delta_bar_cov_rho_inf_699_finite(self):
        """For rho_inf = 699, should compute without overflow (eps < nu/2)."""
        from ndis_gaussian.calibration import delta_bar_cov
        # eps=1.0 < nu/2 = 699/2 = 349.5, so non-trivial case
        val = delta_bar_cov(1.0, rho_inf=699.0, nu=699.0, d=1)
        # Very large rho_inf → large sensitivity → delta_bar_cov close to 1
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# NDISGaussianWrapper + GPR
# ---------------------------------------------------------------------------

class TestWrapperWithGPR:
    @pytest.fixture
    def small_gpr_setup(self):
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        rng = np.random.default_rng(10)
        n, d = 15, 3
        X = rng.standard_normal((n, d))
        y = np.clip(rng.standard_normal(n), -1.0, 1.0)
        x_star = rng.standard_normal(d)
        # Use large sigma_n2 for tractable tau* (small sensitivity)
        gpr = GPRGaussianOutput(
            x_star=x_star,
            kernel=RBFKernel(1.0),
            sigma_n2=100.0,   # large obs noise → small Delta
            B_bound=1.0,
            l_bound=1.0,
        )
        go = gpr.fit(X, y)
        public_meta = {"n": n, "d": 1, "B": 1.0, "l": 1.0, "sigma_n2": 100.0}
        return gpr, go, public_meta

    def test_calibrate_and_release(self, small_gpr_setup):
        from ndis_gaussian.wrapper import NDISGaussianWrapper
        gpr, go, public_meta = small_gpr_setup
        wrapper = NDISGaussianWrapper()
        wrapper.calibrate(gpr, epsilon=1.0, delta=1e-3, public_meta=public_meta)
        assert wrapper.tau_star > 0.0
        release = wrapper.release(go, seed=42)
        assert release.theta_tilde.shape == (1,), \
            f"Expected theta_tilde shape (1,), got {release.theta_tilde.shape}"
        assert release.Sigma_noised.shape == (1, 1)

    def test_release_shape_is_scalar(self, small_gpr_setup):
        """GPR output is scalar (m=1): theta_tilde must be shape (1,)."""
        from ndis_gaussian.wrapper import NDISGaussianWrapper
        gpr, go, public_meta = small_gpr_setup
        wrapper = NDISGaussianWrapper()
        wrapper.calibrate(gpr, epsilon=2.0, delta=1e-2, public_meta=public_meta)
        release = wrapper.release(go, seed=7)
        assert release.theta_tilde.shape == (1,)
        # Sigma_noised = [[Sigma_star + tau_star]]
        expected = go.Sigma + wrapper.tau_star * np.eye(1)
        np.testing.assert_allclose(release.Sigma_noised, expected)

    def test_release_reproducibility(self, small_gpr_setup):
        from ndis_gaussian.wrapper import NDISGaussianWrapper
        gpr, go, public_meta = small_gpr_setup
        wrapper = NDISGaussianWrapper()
        wrapper.calibrate(gpr, epsilon=1.0, delta=1e-3, public_meta=public_meta)
        r1 = wrapper.release(go, seed=99)
        r2 = wrapper.release(go, seed=99)
        np.testing.assert_array_equal(r1.theta_tilde, r2.theta_tilde)

    def test_calibrate_achieves_delta(self, small_gpr_setup):
        """tau* must satisfy delta_bar_LC(eps; ...) ≤ delta."""
        from ndis_gaussian.wrapper import NDISGaussianWrapper
        from ndis_gaussian.calibration import delta_bar_LC
        gpr, go, public_meta = small_gpr_setup
        wrapper = NDISGaussianWrapper()
        epsilon, delta = 1.0, 1e-3
        wrapper.calibrate(gpr, epsilon=epsilon, delta=delta, public_meta=public_meta)
        sens = gpr.sensitivity(wrapper.tau_star, public_meta)
        achieved = delta_bar_LC(epsilon, sens.Delta, sens.rho_inf, sens.nu, d=1)
        assert achieved <= delta + 1e-7, \
            f"tau* does not achieve delta: achieved={achieved:.3e} > {delta:.3e}"


# ---------------------------------------------------------------------------
# LinnerudAdapter
# ---------------------------------------------------------------------------

class TestLinnerudAdapter:
    def test_load_default_target_is_scalar_and_bounded(self):
        from rpbench.config import SplitSpec, PreprocessSpec
        from rpbench.datasets.linnerud import LinnerudAdapter

        adapter = LinnerudAdapter()
        bundle = adapter.load(
            SplitSpec(train_fraction=0.8, seed=42),
            PreprocessSpec(
                scale_x=True,
                clip_x=True,
                clip_y=True,
                clip_x_bound=3.0,
                clip_y_bound=1.0,
            ),
        )

        assert bundle.X_train.shape == (16, 3)
        assert bundle.X_test.shape == (4, 3)
        assert bundle.y_train.shape == (16,)
        assert bundle.y_test.shape == (4,)
        assert bundle.meta["target_index"] == 0
        assert bundle.meta["target_name"] == "Weight"
        assert bundle.meta["C_Y"] == pytest.approx(1.0)
        assert np.max(np.abs(bundle.y_train)) <= bundle.meta["C_Y"] + 1e-12

    def test_target_index_selects_different_scalar_target(self):
        from rpbench.config import SplitSpec, PreprocessSpec
        from rpbench.datasets.linnerud import LinnerudAdapter

        adapter = LinnerudAdapter(target_index=2)
        bundle = adapter.load(SplitSpec(train_fraction=0.8, seed=42), PreprocessSpec())

        assert bundle.y_train.ndim == 1
        assert bundle.meta["target_index"] == 2
        assert bundle.meta["target_name"] == "Pulse"

    def test_invalid_target_index_raises(self):
        from rpbench.config import SplitSpec, PreprocessSpec
        from rpbench.datasets.linnerud import LinnerudAdapter

        adapter = LinnerudAdapter(target_index=99)
        with pytest.raises(ValueError, match="target_index=99 out of range"):
            adapter.load(SplitSpec(train_fraction=0.8, seed=42), PreprocessSpec())

    def test_linnerud_gpr_end_to_end(self):
        from rpbench.config import SplitSpec, PreprocessSpec
        from rpbench.datasets.linnerud import LinnerudAdapter
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        from ndis_gaussian.wrapper import NDISGaussianWrapper

        adapter = LinnerudAdapter(target_index=0)
        bundle = adapter.load(
            SplitSpec(train_fraction=0.8, seed=42),
            PreprocessSpec(
                scale_x=True,
                clip_x=True,
                clip_y=True,
                clip_x_bound=3.0,
                clip_y_bound=1.0,
            ),
        )
        x_star = bundle.X_test[0]
        gpr = GPRGaussianOutput(
            x_star=x_star,
            kernel=RBFKernel(1.0),
            sigma_n2=10.0,
            B_bound=1.0,
            l_bound=1.0,
        )
        go = gpr.fit(bundle.X_train, bundle.y_train)
        public_meta = {"n": bundle.meta["n_train"], "d": 1, "B": 1.0, "l": 1.0, "sigma_n2": 10.0}

        wrapper = NDISGaussianWrapper()
        wrapper.calibrate(gpr, epsilon=1.0, delta=1e-3, public_meta=public_meta)
        release = wrapper.release(go, seed=42)

        assert go.mu.shape == (1,)
        assert go.Sigma.shape == (1, 1)
        assert release.theta_tilde.shape == (1,)
        assert wrapper.tau_star > 0.0


# ---------------------------------------------------------------------------
# End-to-end: synthetic data (avoids OpenML download)
# ---------------------------------------------------------------------------

class TestGPREndToEnd:
    """Full pipeline test using synthetic data (no OpenML dependency)."""

    def test_full_pipeline_synthetic(self):
        """End-to-end: synthetic regression → GPR fit → calibrate → release."""
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        from ndis_gaussian.wrapper import NDISGaussianWrapper

        rng = np.random.default_rng(42)
        n_train, n_test, d = 30, 5, 6

        X_train = rng.standard_normal((n_train, d))
        X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
        y_train = np.clip(rng.standard_normal(n_train), -1.0, 1.0)

        X_test = rng.standard_normal((n_test, d))
        x_star = X_test[0]
        y_star = rng.standard_normal()

        # Use large sigma_n2 so tau* is tractable
        gpr = GPRGaussianOutput(
            x_star=x_star,
            kernel=RBFKernel(1.0),
            sigma_n2=50.0,
            B_bound=1.0,
            l_bound=1.0,
        )
        go = gpr.fit(X_train, y_train)

        # Check GPR outputs
        assert go.mu.shape == (1,)
        assert go.Sigma.shape == (1, 1)
        assert go.Sigma[0, 0] >= 0.0

        public_meta = {"n": n_train, "d": 1, "B": 1.0, "l": 1.0, "sigma_n2": 50.0}

        wrapper = NDISGaussianWrapper()
        wrapper.calibrate(gpr, epsilon=1.0, delta=1e-3, public_meta=public_meta)
        assert wrapper.tau_star > 0.0

        release = wrapper.release(go, seed=0)
        theta_tilde = float(release.theta_tilde[0])
        assert math.isfinite(theta_tilde), f"theta_tilde is not finite: {theta_tilde}"

    def test_sensitivity_monotone_implies_calibration_monotone(self):
        """Higher epsilon should require smaller tau* (less noise needed)."""
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel
        from ndis_gaussian.wrapper import NDISGaussianWrapper

        rng = np.random.default_rng(13)
        n, d = 20, 4
        X = rng.standard_normal((n, d))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        y = np.clip(rng.standard_normal(n), -1.0, 1.0)
        x_star = rng.standard_normal(d)

        gpr = GPRGaussianOutput(
            x_star=x_star,
            kernel=RBFKernel(1.0),
            sigma_n2=100.0,  # large sigma_n2 for tractability
            B_bound=1.0,
            l_bound=1.0,
        )
        gpr.fit(X, y)
        public_meta = {"n": n, "d": 1, "B": 1.0, "l": 1.0, "sigma_n2": 100.0}

        delta = 1e-3
        w1 = NDISGaussianWrapper()
        w1.calibrate(gpr, epsilon=1.0, delta=delta, public_meta=public_meta)

        w2 = NDISGaussianWrapper()
        w2.calibrate(gpr, epsilon=2.0, delta=delta, public_meta=public_meta)

        assert w2.tau_star <= w1.tau_star + 1e-6, \
            f"tau*(eps=2) > tau*(eps=1): {w2.tau_star} > {w1.tau_star}"

    def test_package_import(self):
        """GPRGaussianOutput and RBFKernel importable from top-level package."""
        from ndis_gaussian import GPRGaussianOutput, RBFKernel
        k = RBFKernel(1.0)
        gpr = GPRGaussianOutput(np.zeros(3), k, sigma_n2=1.0, B_bound=1.0, l_bound=1.0)
        assert gpr is not None
