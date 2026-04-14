"""NDIS-Calibrated Gaussian Mechanism (Figure 5, Theorem 7).

Wraps any GaussianOutputAlgorithm with isotropic covariance inflation tau*
calibrated to satisfy (epsilon, delta)-DP under add/remove adjacency.

Paper: Figure 5 — M^NDIS_q(D):
  Step 1: Binary search for tau* = min{tau >= 0 : delta_bar_LC(eps; ...) <= delta}
  Step 2: Return theta_tilde ~ N(mu, Sigma + tau* * I_m)

Release sampling:
  L = cholesky(Sigma + tau* * I_d)
  z ~ N(0, I_d)
  theta_tilde = mu + L @ z

NOTE: tau* is a covariance (units of variance), NOT a standard deviation.
      The additive noise matrix is tau* * I_m, NOT (tau*)^2 * I_m.
"""

from __future__ import annotations

import numpy as np

from ndis_gaussian.calibration import find_tau_star
from ndis_gaussian.interfaces import GaussianOutputAlgorithm
from ndis_gaussian.types import GaussianOutput, NDISSensitivity, WrappedRelease


class NDISGaussianWrapper:
    """NDIS-Calibrated Gaussian Mechanism (Figure 5, Theorem 7).

    Usage
    -----
    wrapper = NDISGaussianWrapper()
    wrapper.calibrate(algorithm, epsilon, delta, public_meta)
    gaussian_output = algorithm.fit(X_train, y_train)
    release = wrapper.release(gaussian_output, seed=42)

    Attributes (after calibrate())
    ------------------------------
    tau_star : float
        Calibrated covariance inflation (NOT a std dev).
    sigma_std : float
        Standard deviation of the added noise: sqrt(tau_star).
        Convenience property; tau_star is the primary representation.
    """

    def __init__(self) -> None:
        self._tau_star: float | None = None
        self._algorithm: GaussianOutputAlgorithm | None = None
        self._public_meta: dict | None = None
        self._epsilon: float | None = None
        self._delta: float | None = None
        self._sensitivity_at_tau_star: NDISSensitivity | None = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        algorithm: GaussianOutputAlgorithm,
        epsilon: float,
        delta: float,
        public_meta: dict,
        tau_lo: float = 0.0,
        tau_hi_init: float = 1.0,
        tol: float = 1e-8,
    ) -> None:
        """Find tau* via delta_bar_LC binary search. Must call before release().

        Parameters
        ----------
        algorithm : GaussianOutputAlgorithm
            The fitted algorithm (must implement sensitivity()).
        epsilon, delta : float
            Target (epsilon, delta)-DP privacy parameters.
        public_meta : dict
            Public dataset metadata, including at minimum:
              "n" (int), "d" (int), "l" (float, feature row-norm bound).
            Must NOT contain "d_aug" — that key indicates an augmented-data
            regression adapter was accidentally used with a BLR wrapper.
        tau_lo : float
            Lower bracket for the binary search (default 0).
        tau_hi_init : float
            Initial upper bracket (doubled until condition is met).
        tol : float
            Convergence tolerance on tau.
        """
        if "d_aug" in public_meta:
            raise ValueError(
                "public_meta contains 'd_aug', which indicates an augmented-data "
                "regression adapter (e.g. AutoMPG). BLRGaussianOutput expects "
                "plain feature rows (l = C_X only). Check that you are using "
                "BreastCancerAdapter or a compatible binary-classification adapter."
            )
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        tau_star = find_tau_star(
            epsilon=epsilon,
            delta=delta,
            algorithm=algorithm,
            public_meta=public_meta,
            tau_lo=tau_lo,
            tau_hi_init=tau_hi_init,
            tol=tol,
        )
        self._tau_star = tau_star
        self._algorithm = algorithm
        self._public_meta = dict(public_meta)
        self._epsilon = epsilon
        self._delta = delta
        self._sensitivity_at_tau_star = algorithm.sensitivity(tau_star, public_meta)

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def release(
        self,
        gaussian_output: GaussianOutput,
        seed: int,
    ) -> WrappedRelease:
        """Sample theta_tilde ~ N(mu, Sigma + tau* * I_m).

        Paper: Figure 5, Step 2.

        Sampling via Cholesky:
          L = cholesky(Sigma + tau* * I_d)
          z ~ N(0, I_d)
          theta_tilde = mu + L @ z

        Parameters
        ----------
        gaussian_output : GaussianOutput
            Output of algorithm.fit() — contains mu and Sigma.
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        WrappedRelease
        """
        if self._tau_star is None:
            raise RuntimeError("calibrate() must be called before release().")

        tau_star = self._tau_star
        mu = gaussian_output.mu
        Sigma = gaussian_output.Sigma
        d = mu.shape[0]

        Sigma_noised = Sigma + tau_star * np.eye(d)

        # Cholesky sampling: theta_tilde = mu + L @ z
        L = np.linalg.cholesky(Sigma_noised)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(d)
        theta_tilde = mu + L @ z

        return WrappedRelease(
            theta_tilde=theta_tilde,
            mu=mu,
            Sigma_noised=Sigma_noised,
            tau_star=tau_star,
            sensitivity=self._sensitivity_at_tau_star,  # type: ignore[arg-type]
            diagnostics={
                "epsilon": self._epsilon,
                "delta": self._delta,
                "seed": seed,
            },
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tau_star(self) -> float:
        """Calibrated covariance inflation. Available after calibrate()."""
        if self._tau_star is None:
            raise RuntimeError("calibrate() must be called first.")
        return self._tau_star

    @property
    def sigma_std(self) -> float:
        """Standard deviation of added noise: sqrt(tau_star).

        Convenience property. tau_star (not sigma_std) is the primary
        representation — the noise matrix is tau_star * I, not sigma_std^2 * I.
        """
        return float(np.sqrt(self.tau_star))
