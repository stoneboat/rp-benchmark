"""Smoke tests for BLR + NDIS-Calibrated Gaussian Mechanism (Phase 1).

Covers:
  - Types constructible (GaussianOutput, NDISSensitivity, WrappedRelease)
  - BLR fit shapes and Böhning covariance
  - Sensitivity monotonicity in tau (Delta and rho_inf decrease as tau grows)
  - Calibration functions (delta_bar_mean, delta_bar_cov, delta_bar_LC, find_tau_star)
  - NDISGaussianWrapper.calibrate() and .release()
  - BreastCancerAdapter: label set, stratification, no d_aug in public_meta
  - End-to-end pipeline: load -> fit -> calibrate -> release -> evaluate
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable from the tests directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TestTypes:
    def test_gaussian_output_constructible(self):
        from ndis_gaussian.types import GaussianOutput
        d = 5
        mu = np.zeros(d)
        Sigma = np.eye(d)
        go = GaussianOutput(mu=mu, Sigma=Sigma)
        assert go.mu.shape == (d,)
        assert go.Sigma.shape == (d, d)

    def test_ndis_sensitivity_constructible(self):
        from ndis_gaussian.types import NDISSensitivity
        sens = NDISSensitivity(Delta=0.5, rho_inf=0.1, nu=3.0, tau=0.0)
        assert sens.Delta == 0.5
        assert sens.rho_inf == 0.1
        assert sens.nu == 3.0
        assert sens.tau == 0.0

    def test_wrapped_release_constructible(self):
        from ndis_gaussian.types import WrappedRelease, NDISSensitivity
        d = 4
        sens = NDISSensitivity(Delta=0.3, rho_inf=0.05, nu=1.5, tau=1.0)
        wr = WrappedRelease(
            theta_tilde=np.ones(d),
            mu=np.zeros(d),
            Sigma_noised=np.eye(d),
            tau_star=1.0,
            sensitivity=sens,
        )
        assert wr.theta_tilde.shape == (d,)
        assert wr.tau_star == 1.0


# ---------------------------------------------------------------------------
# BLR model
# ---------------------------------------------------------------------------

class TestBLR:
    """Tests for BLRGaussianOutput (Definition 7, Proposition 7)."""

    @pytest.fixture
    def small_blr_data(self):
        """Small synthetic binary classification dataset."""
        rng = np.random.default_rng(0)
        n, d = 60, 5
        X = rng.standard_normal((n, d))
        # Linearly separable-ish labels
        y = np.sign(X @ rng.standard_normal(d))
        y[y == 0] = 1.0  # avoid exact zero
        return X, y, n, d

    def test_fit_returns_correct_shapes(self, small_blr_data):
        from ndis_gaussian.blr.model import BLRGaussianOutput
        X, y, n, d = small_blr_data
        blr = BLRGaussianOutput(lambda_reg=0.1)
        out = blr.fit(X, y)
        assert out.mu.shape == (d,), f"expected mu shape ({d},), got {out.mu.shape}"
        assert out.Sigma.shape == (d, d), f"expected Sigma shape ({d},{d})"

    def test_bohning_covariance_is_psd(self, small_blr_data):
        """Böhning Sigma = (lambda*I + 1/4 * X^T X)^{-1} must be PSD."""
        from ndis_gaussian.blr.model import BLRGaussianOutput
        X, y, n, d = small_blr_data
        blr = BLRGaussianOutput(lambda_reg=0.1)
        out = blr.fit(X, y)
        # All eigenvalues positive
        eigvals = np.linalg.eigvalsh(out.Sigma)
        assert np.all(eigvals > 0), f"Sigma not PD: min eigval={eigvals.min():.3e}"

    def test_sensitivity_positive_at_tau0(self, small_blr_data):
        from ndis_gaussian.blr.model import BLRGaussianOutput
        X, y, n, d = small_blr_data
        blr = BLRGaussianOutput(lambda_reg=0.1)
        public_meta = {"n": n, "d": d, "l": 3.0}
        sens = blr.sensitivity(tau=0.0, public_meta=public_meta)
        assert sens.Delta > 0, "Delta must be > 0"
        assert sens.rho_inf > 0, "rho_inf must be > 0"
        assert sens.nu > 0, "nu must be > 0"
        assert sens.tau == 0.0

    def test_sensitivity_decreasing_in_tau(self, small_blr_data):
        """Proposition 7: Delta and rho_inf are strictly decreasing in tau."""
        from ndis_gaussian.blr.model import BLRGaussianOutput
        X, y, n, d = small_blr_data
        blr = BLRGaussianOutput(lambda_reg=0.1)
        public_meta = {"n": n, "d": d, "l": 3.0}
        taus = [0.0, 0.1, 1.0, 10.0]
        deltas = [blr.sensitivity(t, public_meta).Delta for t in taus]
        rhos = [blr.sensitivity(t, public_meta).rho_inf for t in taus]
        assert all(deltas[i] > deltas[i + 1] for i in range(len(deltas) - 1)), \
            f"Delta not decreasing in tau: {deltas}"
        assert all(rhos[i] > rhos[i + 1] for i in range(len(rhos) - 1)), \
            f"rho_inf not decreasing in tau: {rhos}"

    def test_sensitivity_proposition7_formula(self):
        """Check Proposition 7 formula against hand-computed values."""
        from ndis_gaussian.blr.model import BLRGaussianOutput
        blr = BLRGaussianOutput(lambda_reg=1.0)
        public_meta = {"n": 100, "d": 10, "l": 1.0}
        lam, n, l, d = 1.0, 100, 1.0, 10
        tau = 0.5

        denom_noise = tau + 1.0 / (lam + n * l**2 / 4.0)
        denom_clean = tau + 1.0 / lam
        expected_Delta = (l / lam) / math.sqrt(denom_noise)
        expected_rho = math.log(denom_clean / denom_noise)
        expected_nu = d * expected_rho

        sens = blr.sensitivity(tau=tau, public_meta=public_meta)
        assert math.isclose(sens.Delta, expected_Delta, rel_tol=1e-9)
        assert math.isclose(sens.rho_inf, expected_rho, rel_tol=1e-9)
        assert math.isclose(sens.nu, expected_nu, rel_tol=1e-9)

    def test_lambda_reg_validation(self):
        from ndis_gaussian.blr.model import BLRGaussianOutput
        with pytest.raises(ValueError, match="lambda_reg must be > 0"):
            BLRGaussianOutput(lambda_reg=0.0)
        with pytest.raises(ValueError, match="lambda_reg must be > 0"):
            BLRGaussianOutput(lambda_reg=-1.0)


# ---------------------------------------------------------------------------
# Calibration functions
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_delta_bar_mean_at_zero_epsilon(self):
        """delta_bar_mean(0, Delta) should equal Phi(Delta/2) - Phi(-Delta/2) = erf-like."""
        from ndis_gaussian.calibration import delta_bar_mean
        from scipy.stats import norm
        Delta = 1.0
        val = delta_bar_mean(0.0, Delta)
        expected = norm.cdf(0.5) - norm.cdf(-0.5)
        assert math.isclose(val, expected, rel_tol=1e-9)

    def test_delta_bar_mean_zero_Delta(self):
        from ndis_gaussian.calibration import delta_bar_mean
        assert delta_bar_mean(1.0, 0.0) == 0.0

    @pytest.mark.parametrize(
        ("rho_inf", "nu"),
        [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)],
    )
    def test_delta_bar_cov_zero_sensitivity(self, rho_inf, nu):
        from ndis_gaussian.calibration import delta_bar_cov
        assert delta_bar_cov(1.0, rho_inf=rho_inf, nu=nu, d=10) == 0.0

    def test_delta_bar_cov_one_dimensional_hard_direction(self):
        """The hard covariance direction remains nonzero past eps=nu/2."""
        from ndis_gaussian.calibration import delta_bar_cov
        from scipy.stats import norm

        epsilon = rho = 1.0
        radius = math.sqrt((2.0 * epsilon + rho) / (1.0 - math.exp(-rho)))
        exact = (
            2.0 * norm.sf(radius * math.exp(-rho / 2.0))
            - 2.0 * math.exp(epsilon) * norm.sf(radius)
        )
        bound = delta_bar_cov(epsilon, rho_inf=rho, nu=rho, d=1)

        assert exact == pytest.approx(0.10656, abs=1e-5)
        assert math.isfinite(bound)
        assert 0.0 < bound <= 1.0
        assert bound >= exact - 1e-8

    def test_delta_bar_cov_large_epsilon_is_finite(self):
        from ndis_gaussian.calibration import delta_bar_cov

        bound = delta_bar_cov(710.0, rho_inf=0.1, nu=0.1, d=1)
        assert math.isfinite(bound)
        assert 0.0 <= bound <= 1.0

    def test_covariance_envelope_scalar_formulas(self):
        from ndis_gaussian.calibration import _phi_s, _psi_u

        s, u, ell = 0.1, 0.75, 0.4
        expected_phi = -s * ell - 0.5 * math.log1p(-2.0 * s * math.expm1(ell))
        expected_psi = (
            u * ell - 0.5 * math.log1p(2.0 * u * (-math.expm1(-ell)))
        )

        assert _phi_s(s, ell) == pytest.approx(expected_phi, rel=1e-12, abs=1e-12)
        assert _psi_u(u, ell) == pytest.approx(expected_psi, rel=1e-12, abs=1e-12)

        rho = 1.0
        k = 1.0 / (2.0 * math.expm1(rho))
        assert math.isfinite(_phi_s(0.99 * k, rho))
        assert math.isinf(_phi_s(k, rho))
        assert math.isfinite(_psi_u(10.0 * k, rho))

    def test_covariance_envelope_constraints(self):
        from ndis_gaussian.calibration import _A_star, _B_star, _phi_s, _psi_u

        rho_inf, d = 0.4, 3
        s, u = 0.1, 0.75
        for nu, coordinates in [
            (0.9, [0.4, 0.4, 0.1]),
            (2.0, [0.4, 0.4, 0.4]),
        ]:
            expected_A = sum(_phi_s(s, ell) for ell in coordinates)
            expected_B = sum(_psi_u(u, ell) for ell in coordinates)
            assert _A_star(s, rho_inf, nu, d) == pytest.approx(expected_A, abs=1e-12)
            assert _B_star(u, rho_inf, nu, d) == pytest.approx(expected_B, abs=1e-12)

    def test_delta_bar_cov_optimizer_domains(self, monkeypatch):
        import ndis_gaussian.calibration as calibration

        calls = []

        def _recording_objective(s, u, epsilon, rho_inf, nu, d):
            calls.append((s, u))
            k = 1.0 / (2.0 * math.expm1(rho_inf))
            return (s - k / 2.0) ** 2 + (u - 2.0 * k) ** 2

        monkeypatch.setattr(calibration, "_cov_objective", _recording_objective)
        value = calibration.delta_bar_cov(1.0, rho_inf=1.0, nu=1.0, d=1)

        k = 1.0 / (2.0 * math.expm1(1.0))
        assert calls
        assert all(0.0 <= s < k for s, _ in calls)
        assert all(u >= 0.0 for _, u in calls)
        assert any(math.isfinite(u) and u > k for _, u in calls)
        assert value < 1e-6

    def test_delta_bar_cov_nontrivial(self):
        """For small epsilon and moderate rho/nu, delta_bar_cov should be in (0, 1)."""
        from ndis_gaussian.calibration import delta_bar_cov
        val = delta_bar_cov(1.0, rho_inf=0.1, nu=3.0, d=30)
        assert 0.0 < val < 1.0, f"expected value in (0,1), got {val}"

    def test_delta_bar_LC_at_large_epsilon(self):
        """For large enough epsilon, delta_bar_LC should approach 0."""
        from ndis_gaussian.calibration import delta_bar_LC
        # Very large epsilon -> tight privacy, both terms -> 0
        val = delta_bar_LC(100.0, Delta=0.01, rho_inf=0.001, nu=0.03, d=5)
        assert val <= 1e-6, f"expected near 0 for large eps, got {val}"

    def test_delta_bar_LC_is_clamped(self):
        """Output of delta_bar_LC must be in [0, 1]."""
        from ndis_gaussian.calibration import delta_bar_LC
        val = delta_bar_LC(0.01, Delta=5.0, rho_inf=1.0, nu=30.0, d=30)
        assert 0.0 <= val <= 1.0, f"delta_bar_LC out of [0,1]: {val}"

    def test_delta_bar_LC_decreasing_in_epsilon(self):
        """delta_bar_LC should decrease as epsilon increases (more privacy budget -> smaller bound)."""
        from ndis_gaussian.calibration import delta_bar_LC
        epsilons = [0.5, 1.0, 2.0, 4.0]
        deltas = [delta_bar_LC(e, Delta=1.0, rho_inf=0.1, nu=3.0, d=30) for e in epsilons]
        assert all(deltas[i] >= deltas[i + 1] for i in range(len(deltas) - 1)), \
            f"delta_bar_LC not non-increasing in eps: {deltas}"

    def test_find_tau_star_achieves_delta(self):
        """tau* found by binary search should satisfy delta_bar_LC(eps; ...) <= delta."""
        from ndis_gaussian.blr.model import BLRGaussianOutput
        from ndis_gaussian.calibration import find_tau_star, delta_bar_LC

        rng = np.random.default_rng(1)
        n, d = 50, 8
        X = rng.standard_normal((n, d))
        y = np.sign(X[:, 0])
        y[y == 0] = 1.0

        blr = BLRGaussianOutput(lambda_reg=0.1)
        blr.fit(X, y)
        public_meta = {"n": n, "d": d, "l": float(np.max(np.linalg.norm(X, axis=1)))}

        epsilon, delta = 1.0, 1e-3
        tau_star = find_tau_star(epsilon, delta, blr, public_meta)
        assert tau_star >= 0.0

        sens = blr.sensitivity(tau_star, public_meta)
        achieved = delta_bar_LC(epsilon, sens.Delta, sens.rho_inf, sens.nu, d)
        assert achieved <= delta + 1e-7, \
            f"tau* does not achieve delta: achieved={achieved:.3e} > {delta:.3e}"

    def test_find_tau_star_monotone_in_epsilon(self):
        """Higher epsilon (more privacy budget) should require less noise (smaller tau*)."""
        from ndis_gaussian.blr.model import BLRGaussianOutput
        from ndis_gaussian.calibration import find_tau_star

        rng = np.random.default_rng(2)
        n, d = 50, 8
        X = rng.standard_normal((n, d))
        y = np.sign(X[:, 0])
        y[y == 0] = 1.0

        blr = BLRGaussianOutput(lambda_reg=0.1)
        blr.fit(X, y)
        public_meta = {"n": n, "d": d, "l": float(np.max(np.linalg.norm(X, axis=1)))}

        delta = 1e-3
        tau1 = find_tau_star(1.0, delta, blr, public_meta)
        tau2 = find_tau_star(2.0, delta, blr, public_meta)
        assert tau2 <= tau1, \
            f"tau* should decrease as epsilon increases: tau(eps=1)={tau1:.4f}, tau(eps=2)={tau2:.4f}"


# ---------------------------------------------------------------------------
# NDISGaussianWrapper
# ---------------------------------------------------------------------------

class TestWrapper:
    @pytest.fixture
    def fitted_blr_and_meta(self):
        from ndis_gaussian.blr.model import BLRGaussianOutput
        rng = np.random.default_rng(5)
        n, d = 80, 6
        X = rng.standard_normal((n, d))
        y = np.sign(X @ rng.standard_normal(d))
        y[y == 0] = 1.0
        blr = BLRGaussianOutput(lambda_reg=0.05)
        go = blr.fit(X, y)
        public_meta = {"n": n, "d": d, "l": float(np.max(np.linalg.norm(X, axis=1)))}
        return blr, go, public_meta, d

    def test_calibrate_and_release(self, fitted_blr_and_meta):
        from ndis_gaussian.wrapper import NDISGaussianWrapper
        blr, go, public_meta, d = fitted_blr_and_meta
        wrapper = NDISGaussianWrapper()
        wrapper.calibrate(blr, epsilon=1.0, delta=1e-3, public_meta=public_meta)
        assert wrapper.tau_star >= 0.0
        assert wrapper.sigma_std == pytest.approx(math.sqrt(wrapper.tau_star), rel=1e-9)

        release = wrapper.release(go, seed=42)
        assert release.theta_tilde.shape == (d,), \
            f"expected theta_tilde shape ({d},), got {release.theta_tilde.shape}"
        assert release.Sigma_noised.shape == (d, d)
        # Sigma_noised = Sigma + tau* * I
        expected = go.Sigma + wrapper.tau_star * np.eye(d)
        np.testing.assert_allclose(release.Sigma_noised, expected)

    def test_release_before_calibrate_raises(self, fitted_blr_and_meta):
        from ndis_gaussian.wrapper import NDISGaussianWrapper
        from ndis_gaussian.types import GaussianOutput
        wrapper = NDISGaussianWrapper()
        with pytest.raises(RuntimeError, match="calibrate\\(\\) must be called"):
            go = GaussianOutput(mu=np.zeros(3), Sigma=np.eye(3))
            wrapper.release(go, seed=0)

    def test_d_aug_in_public_meta_raises(self, fitted_blr_and_meta):
        """Passing d_aug in public_meta should raise ValueError (regression adapter guard)."""
        from ndis_gaussian.wrapper import NDISGaussianWrapper
        blr, go, public_meta, d = fitted_blr_and_meta
        bad_meta = dict(public_meta, d_aug=d + 1)
        wrapper = NDISGaussianWrapper()
        with pytest.raises(ValueError, match="d_aug"):
            wrapper.calibrate(blr, epsilon=1.0, delta=1e-3, public_meta=bad_meta)

    def test_invalid_epsilon_raises(self, fitted_blr_and_meta):
        from ndis_gaussian.wrapper import NDISGaussianWrapper
        blr, go, public_meta, d = fitted_blr_and_meta
        wrapper = NDISGaussianWrapper()
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            wrapper.calibrate(blr, epsilon=0.0, delta=1e-3, public_meta=public_meta)

    def test_invalid_delta_raises(self, fitted_blr_and_meta):
        from ndis_gaussian.wrapper import NDISGaussianWrapper
        blr, go, public_meta, d = fitted_blr_and_meta
        wrapper = NDISGaussianWrapper()
        with pytest.raises(ValueError, match="delta must be in"):
            wrapper.calibrate(blr, epsilon=1.0, delta=0.0, public_meta=public_meta)

    def test_release_seed_reproducibility(self, fitted_blr_and_meta):
        """Two releases with the same seed should give the same theta_tilde."""
        from ndis_gaussian.wrapper import NDISGaussianWrapper
        blr, go, public_meta, d = fitted_blr_and_meta
        wrapper = NDISGaussianWrapper()
        wrapper.calibrate(blr, epsilon=1.0, delta=1e-3, public_meta=public_meta)
        r1 = wrapper.release(go, seed=99)
        r2 = wrapper.release(go, seed=99)
        np.testing.assert_array_equal(r1.theta_tilde, r2.theta_tilde)

    def test_release_different_seeds_differ(self, fitted_blr_and_meta):
        from ndis_gaussian.wrapper import NDISGaussianWrapper
        blr, go, public_meta, d = fitted_blr_and_meta
        wrapper = NDISGaussianWrapper()
        wrapper.calibrate(blr, epsilon=1.0, delta=1e-3, public_meta=public_meta)
        r1 = wrapper.release(go, seed=10)
        r2 = wrapper.release(go, seed=11)
        assert not np.allclose(r1.theta_tilde, r2.theta_tilde), \
            "Different seeds should give different theta_tilde"


# ---------------------------------------------------------------------------
# BreastCancerAdapter
# ---------------------------------------------------------------------------

class TestBreastCancerAdapter:
    @pytest.fixture
    def bundle_and_adapter(self):
        from rpbench.config import SplitSpec, PreprocessSpec
        from rpbench.datasets.breast_cancer import BreastCancerAdapter
        adapter = BreastCancerAdapter()
        split = SplitSpec(train_fraction=0.8, seed=42)
        pre = PreprocessSpec(scale_x=True, clip_x=True, clip_x_bound=3.0,
                             clip_y=False, clip_y_bound=3.0)
        bundle = adapter.load(split, pre)
        return bundle, adapter

    def test_label_set(self, bundle_and_adapter):
        """Labels must be in {-1, +1}."""
        bundle, _ = bundle_and_adapter
        unique = set(np.unique(bundle.y_train)) | set(np.unique(bundle.y_test))
        assert unique <= {-1.0, 1.0}, f"Unexpected labels: {unique}"

    def test_shapes(self, bundle_and_adapter):
        bundle, _ = bundle_and_adapter
        n_tr = bundle.meta["n_train"]
        n_te = bundle.meta["n_test"]
        d = bundle.meta["d"]
        assert bundle.X_train.shape == (n_tr, d)
        assert bundle.y_train.shape == (n_tr,)
        assert bundle.X_test.shape == (n_te, d)
        assert bundle.y_test.shape == (n_te,)
        assert d == 30  # Wisconsin breast cancer has 30 features

    def test_cx_bound_respected(self, bundle_and_adapter):
        """With clip_x=True, all train rows must have ||x||_2 <= C_X + tol."""
        bundle, _ = bundle_and_adapter
        norms = np.linalg.norm(bundle.X_train, axis=1)
        assert np.all(norms <= bundle.meta["C_X"] + 1e-9), \
            f"Some train rows exceed C_X={bundle.meta['C_X']:.4f}"

    def test_public_meta_no_d_aug(self, bundle_and_adapter):
        """public_meta must NOT contain 'd_aug' (would break BLR wrapper guard)."""
        bundle, adapter = bundle_and_adapter
        meta = adapter.public_meta(bundle)
        assert "d_aug" not in meta, "public_meta must not contain 'd_aug' for BLR adapter"
        assert "C_Y" not in meta, "public_meta must not contain 'C_Y' for BLR adapter"

    def test_public_meta_l_equals_cx(self, bundle_and_adapter):
        """l in public_meta must equal C_X (feature norm only, per Proposition 7)."""
        bundle, adapter = bundle_and_adapter
        meta = adapter.public_meta(bundle)
        assert "l" in meta
        assert "n" in meta
        assert "d" in meta
        assert math.isclose(meta["l"], bundle.meta["C_X"], rel_tol=1e-9), \
            f"l={meta['l']} != C_X={bundle.meta['C_X']}"

    def test_stratification(self, bundle_and_adapter):
        """Stratified split: class proportions should be similar in train and test."""
        bundle, _ = bundle_and_adapter
        def class_ratio(y):
            return float(np.mean(y == 1.0))
        r_train = class_ratio(bundle.y_train)
        r_test = class_ratio(bundle.y_test)
        # Proportions should be within 5 percentage points
        assert abs(r_train - r_test) < 0.05, \
            f"Train/test class proportions differ too much: {r_train:.3f} vs {r_test:.3f}"

    def test_no_clip_cx_is_max_norm(self):
        """When clip_x=False, C_X should equal the max row norm of X_train."""
        from rpbench.config import SplitSpec, PreprocessSpec
        from rpbench.datasets.breast_cancer import BreastCancerAdapter
        adapter = BreastCancerAdapter()
        split = SplitSpec(train_fraction=0.8, seed=42)
        pre = PreprocessSpec(scale_x=True, clip_x=False)
        bundle = adapter.load(split, pre)
        expected_cx = float(np.max(np.linalg.norm(bundle.X_train, axis=1)))
        assert math.isclose(bundle.meta["C_X"], expected_cx, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Full pipeline: load -> fit -> calibrate -> release -> evaluate."""

    def test_full_pipeline(self):
        from rpbench.config import SplitSpec, PreprocessSpec
        from rpbench.datasets.breast_cancer import BreastCancerAdapter
        from ndis_gaussian.blr.model import BLRGaussianOutput
        from ndis_gaussian.wrapper import NDISGaussianWrapper

        # Load
        adapter = BreastCancerAdapter()
        split = SplitSpec(train_fraction=0.8, seed=0)
        pre = PreprocessSpec(scale_x=True, clip_x=True, clip_x_bound=3.0, clip_y=False)
        bundle = adapter.load(split, pre)
        public_meta = adapter.public_meta(bundle)

        # Fit BLR
        blr = BLRGaussianOutput(lambda_reg=0.01)
        go = blr.fit(bundle.X_train, bundle.y_train)
        assert go.mu.shape == (30,)

        # Calibrate
        wrapper = NDISGaussianWrapper()
        wrapper.calibrate(blr, epsilon=1.0, delta=1e-3, public_meta=public_meta)
        assert wrapper.tau_star > 0.0

        # Release
        release = wrapper.release(go, seed=42)
        assert release.theta_tilde.shape == (30,)

        # Evaluate — private release should achieve > 50% accuracy (better than chance)
        margins = bundle.y_test * (bundle.X_test @ release.theta_tilde)
        accuracy = float(np.mean(margins > 0.0))
        assert accuracy > 0.5, \
            f"Private BLR accuracy too low ({accuracy:.3f}); expected > 0.5"

    def test_diagnostics_in_release(self):
        """WrappedRelease.diagnostics should contain epsilon, delta, seed."""
        from rpbench.config import SplitSpec, PreprocessSpec
        from rpbench.datasets.breast_cancer import BreastCancerAdapter
        from ndis_gaussian.blr.model import BLRGaussianOutput
        from ndis_gaussian.wrapper import NDISGaussianWrapper

        adapter = BreastCancerAdapter()
        bundle = adapter.load(
            SplitSpec(train_fraction=0.8, seed=0),
            PreprocessSpec(scale_x=True, clip_x=True, clip_x_bound=3.0, clip_y=False),
        )
        public_meta = adapter.public_meta(bundle)

        blr = BLRGaussianOutput(lambda_reg=0.01)
        go = blr.fit(bundle.X_train, bundle.y_train)

        wrapper = NDISGaussianWrapper()
        wrapper.calibrate(blr, epsilon=2.0, delta=1e-4, public_meta=public_meta)
        release = wrapper.release(go, seed=7)

        diag = release.diagnostics
        assert diag["epsilon"] == 2.0
        assert diag["delta"] == 1e-4
        assert diag["seed"] == 7
