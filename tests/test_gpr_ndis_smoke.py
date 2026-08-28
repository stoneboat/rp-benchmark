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
# Specialized exact-NDIS calibration for scalar GPR
# ---------------------------------------------------------------------------

class TestScalarExactNDISCalibration:
    """Focused regressions for the application-specific scalar GPR path."""

    @staticmethod
    def _linnerud_gpr_and_meta():
        """Return the public sensitivity setup used by the Section 6 demo."""
        from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel

        public_meta = {
            "n": 16,
            "d": 1,
            "B": 0.5,
            "l": 1.0,
            "sigma_n2": 250.0,
        }
        gpr = GPRGaussianOutput(
            x_star=np.zeros(3),
            kernel=RBFKernel(5.0),
            sigma_n2=250.0,
            B_bound=0.5,
            l_bound=1.0,
        )
        return gpr, public_meta

    def test_scalar_exact_covariance_diagnostic(self):
        """Reproduce the hard-direction diagnostic that exposed the old cutoff."""
        from ndis_gaussian.calibration import _delta_cov_scalar_exact

        value = _delta_cov_scalar_exact(epsilon=1.0, ell=1.0)
        assert value == pytest.approx(
            0.10655957800717819,
            rel=5e-13,
            abs=5e-15,
        )
        # In particular, epsilon >= ell/2 does not make the hard direction zero.
        assert value > 0.0

    def test_scalar_exact_covariance_is_numerically_conservative(self):
        """Known high-precision values must not be rounded downward."""
        from ndis_gaussian.calibration import _delta_cov_scalar_exact

        references = [
            (0.0, 0.1, 0.0241920327600449687895901235841147),
            (0.1, 0.5, 0.0974754302909317833718539134181214),
            (1.0, 1.0, 0.106559578007178159870503715334422),
            (2.0, 0.2, 0.00000232161788974197523928831295655),
            (4.0, 2.0, 0.174219752988084658452050060431961),
        ]
        for epsilon, ell, high_precision_value in references:
            assert _delta_cov_scalar_exact(epsilon, ell) >= high_precision_value

    @pytest.mark.parametrize("epsilon", [0.0, 0.1, 1.0, 10.0, 710.0])
    def test_scalar_exact_covariance_identical(self, epsilon):
        from ndis_gaussian.calibration import _delta_cov_scalar_exact

        assert _delta_cov_scalar_exact(epsilon=epsilon, ell=0.0) == 0.0

    @pytest.mark.parametrize(
        ("epsilon", "ell"),
        [(0.0, 0.1), (0.1, 0.5), (1.0, 1.0), (2.0, 0.2), (4.0, 2.0)],
    )
    def test_scalar_exact_covariance_matches_density_quadrature(
        self, epsilon, ell
    ):
        """Compare the tail formula with direct Gaussian-density integration."""
        from scipy.integrate import quad
        from scipy.stats import norm

        from ndis_gaussian.calibration import _delta_cov_scalar_exact

        tau = math.exp(ell)
        # The likelihood-ratio event in the original x coordinate is
        # |x| >= x_cut.  Integrating densities is independent of the helper's
        # chi-square/Gaussian-tail subtraction.
        x_cut = math.sqrt((ell + 2.0 * epsilon) / (-math.expm1(-ell)))

        def _positive_density_difference(x):
            return (
                norm.pdf(x, scale=math.sqrt(tau))
                - math.exp(epsilon) * norm.pdf(x)
            )

        one_tail, integration_error = quad(
            _positive_density_difference,
            x_cut,
            math.inf,
            epsabs=1e-14,
            epsrel=1e-12,
            limit=300,
        )
        direct = 2.0 * one_tail
        assert 2.0 * integration_error < 5e-13
        assert _delta_cov_scalar_exact(epsilon, ell) == pytest.approx(
            direct,
            rel=2e-11,
            abs=2e-13,
        )

    @pytest.mark.parametrize(
        ("epsilon", "ell"),
        [(0.0, 0.1), (0.1, 0.5), (1.0, 1.0), (2.0, 0.2), (4.0, 2.0)],
    )
    def test_scalar_exact_covariance_is_bounded_by_generic(self, epsilon, ell):
        from ndis_gaussian.calibration import (
            _delta_cov_scalar_exact,
            delta_bar_cov,
        )

        exact = _delta_cov_scalar_exact(epsilon, ell)
        generic = delta_bar_cov(epsilon, rho_inf=ell, nu=ell, d=1)
        assert exact <= generic + 1e-12
        if epsilon == 1.0 and ell == 1.0:
            assert generic == pytest.approx(0.9099902959890521, abs=5e-8)

    def test_scalar_exact_covariance_envelope_endpoint_is_worst_case(self):
        """Numerically exercise the proved monotonicity over broad scalar grids."""
        from ndis_gaussian.calibration import _delta_cov_scalar_exact

        for epsilon in [0.0, 0.1, 1.0, 4.0]:
            for rho_inf in [1e-8, 1e-4, 0.1, 1.0, 5.0, 20.0]:
                ell_grid = np.linspace(0.0, rho_inf, 201)
                values = np.array(
                    [_delta_cov_scalar_exact(epsilon, ell) for ell in ell_grid]
                )
                assert np.all(np.diff(values) >= -2e-14)
                assert values.max() == pytest.approx(
                    _delta_cov_scalar_exact(epsilon, rho_inf),
                    rel=2e-12,
                    abs=2e-15,
                )

    def test_scalar_exact_hard_direction_dominates_reverse(self):
        """Exercise covariance-order asymmetry for the scalar variance pair."""
        from scipy.stats import norm

        from ndis_gaussian.calibration import _delta_cov_scalar_exact

        for ell in [1e-4, 0.1, 1.0, 5.0, 20.0]:
            tau = math.exp(ell)
            for epsilon in np.linspace(0.0, ell, 21):
                if 2.0 * epsilon >= ell:
                    reverse = 0.0
                else:
                    cutoff_sq = tau * (ell - 2.0 * epsilon) / (tau - 1.0)
                    reverse = (
                        2.0 * norm.cdf(math.sqrt(cutoff_sq))
                        - 1.0
                        - math.exp(epsilon)
                        * (
                            2.0 * norm.cdf(math.sqrt(cutoff_sq / tau))
                            - 1.0
                        )
                    )
                hard = _delta_cov_scalar_exact(epsilon, ell)
                assert reverse <= hard + 1e-14

    def test_scalar_exact_lc_no_worse_than_generic_same_order(self):
        """Replace only covariance while holding covariance-first composition fixed."""
        from scipy.optimize import minimize_scalar

        from ndis_gaussian.calibration import (
            _delta_cov_scalar_exact,
            delta_bar_LC,
            delta_bar_LC_scalar_gpr,
            delta_bar_cov,
            delta_bar_mean,
        )

        gpr, public_meta = self._linnerud_gpr_and_meta()
        epsilon = 1.0
        for tau in [1.0, 4.0, 8.0]:
            sens = gpr.sensitivity(tau, public_meta)

            def _specialized_at_split(epsilon_cov):
                return (
                    _delta_cov_scalar_exact(epsilon_cov, sens.rho_inf)
                    + math.exp(epsilon_cov)
                    * delta_bar_mean(epsilon - epsilon_cov, sens.Delta)
                )

            def _generic_covariance_first_at_split(epsilon_cov):
                return (
                    delta_bar_cov(
                        epsilon_cov,
                        rho_inf=sens.rho_inf,
                        nu=sens.rho_inf,
                        d=1,
                    )
                    + math.exp(epsilon_cov)
                    * delta_bar_mean(epsilon - epsilon_cov, sens.Delta)
                )

            # Pointwise comparison is the rigorous substitution argument; it
            # avoids mixing the covariance-first rule with the repository's
            # historical mean-first generic implementation.
            for epsilon_cov in np.linspace(0.0, epsilon, 11):
                assert _specialized_at_split(epsilon_cov) <= (
                    _generic_covariance_first_at_split(epsilon_cov) + 1e-12
                )

            generic_result = minimize_scalar(
                _generic_covariance_first_at_split,
                bounds=(0.0, epsilon),
                method="bounded",
                options={"xatol": 1e-9},
            )
            generic_covariance_first = min(
                float(generic_result.fun),
                _generic_covariance_first_at_split(0.0),
                _generic_covariance_first_at_split(epsilon),
            )
            specialized = delta_bar_LC_scalar_gpr(
                epsilon, sens.Delta, sens.rho_inf
            )
            assert specialized <= generic_covariance_first + 5e-8

            # Also preserve the literal requested comparison with the existing
            # generic wrapper, whose composition order is mean-first.
            generic_existing = delta_bar_LC(
                epsilon,
                sens.Delta,
                sens.rho_inf,
                sens.nu,
                d=1,
            )
            assert specialized <= generic_existing + 5e-8

    def test_scalar_exact_gpr_calibration_certifies_and_is_near_minimal(self):
        from ndis_gaussian.calibration import (
            delta_bar_LC_scalar_gpr,
            find_tau_star_scalar_gpr,
            scalar_gpr_privacy_certificate,
        )
        from ndis_gaussian.wrapper import ScalarGPRExactNDISWrapper

        gpr, public_meta = self._linnerud_gpr_and_meta()
        epsilon, target_delta = 1.0, 1e-5
        search_tol = 1e-6
        tau_star = find_tau_star_scalar_gpr(
            epsilon=epsilon,
            delta=target_delta,
            algorithm=gpr,
            public_meta=public_meta,
            tol=search_tol,
        )
        assert tau_star == pytest.approx(8.647640513419423, abs=2e-6)

        sensitivity = gpr.sensitivity(tau_star, public_meta)
        certificate = scalar_gpr_privacy_certificate(
            epsilon,
            sensitivity.Delta,
            sensitivity.rho_inf,
        )
        assert certificate["epsilon_cov"] + certificate["epsilon_mean"] \
            == pytest.approx(epsilon, abs=1e-10)
        assert certificate["epsilon_cov"] == pytest.approx(
            0.8284878615473259,
            abs=2e-5,
        )
        assert certificate["delta_mean_contribution"] == pytest.approx(
            math.exp(certificate["epsilon_cov"])
            * certificate["delta_mean_raw"],
            rel=2e-12,
            abs=1e-15,
        )
        assert certificate["delta_total"] == pytest.approx(
            certificate["delta_covariance"]
            + certificate["delta_mean_contribution"],
            rel=2e-12,
            abs=1e-15,
        )
        assert certificate["delta_total"] <= target_delta

        # tau_star is the passing (upper) side of a binary-search interval.
        # One search tolerance below it must still fail the predicate.
        tau_below = tau_star - search_tol
        sensitivity_below = gpr.sensitivity(tau_below, public_meta)
        assert delta_bar_LC_scalar_gpr(
            epsilon,
            sensitivity_below.Delta,
            sensitivity_below.rho_inf,
        ) > target_delta

        wrapper = ScalarGPRExactNDISWrapper()
        wrapper.calibrate(
            gpr,
            epsilon=epsilon,
            delta=target_delta,
            public_meta=public_meta,
            tol=search_tol,
        )
        assert wrapper.tau_star == pytest.approx(tau_star, abs=search_tol)
        assert wrapper.sigma_std == pytest.approx(math.sqrt(wrapper.tau_star))
        assert wrapper.calibration_mode == "scalar_gpr_exact"
        assert wrapper.calibration_diagnostics["delta_total"] <= target_delta

    def test_scalar_exact_covariance_extreme_parameters_are_finite(self):
        from scipy.stats import norm

        from ndis_gaussian.calibration import _delta_cov_scalar_exact

        for epsilon in [0.0, 1.0, 50.0, 710.0]:
            for ell in [1e-12, 1e-8, 0.1, 1.0, 20.0, 800.0]:
                value = _delta_cov_scalar_exact(epsilon, ell)
                assert math.isfinite(value)
                assert 0.0 <= value <= 1.0

        near_zero = _delta_cov_scalar_exact(epsilon=0.0, ell=1e-8)
        assert near_zero / 1e-8 == pytest.approx(norm.pdf(1.0), rel=2e-6)

    def test_scalar_exact_stable_mean_and_large_epsilon_certificate(self):
        from ndis_gaussian.calibration import (
            _delta_mean_scalar_stable_upper,
            scalar_gpr_privacy_certificate,
        )

        # 100-digit reference at the production split/sensitivity.
        mean_upper = _delta_mean_scalar_stable_upper(
            0.17151213844016522,
            0.04301413337968837,
        )
        assert mean_upper >= 3.5408902523682561e-7

        for epsilon in [710.0, 1000.0]:
            certificate = scalar_gpr_privacy_certificate(
                epsilon,
                Delta=0.1,
                rho_inf=0.1,
            )
            assert math.isfinite(certificate["delta_total"])
            assert 0.0 <= certificate["delta_total"] <= 1.0


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
