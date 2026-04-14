"""BLR Gaussian-output algorithm: Definition 7 + Proposition 7.

Bayesian Logistic Regression surrogate (Böhning-style covariance).

Paper references:
  Definition 7  — BLR surrogate: theta_hat via L-BFGS-B, Sigma via Böhning bound
  Proposition 7 — BLR NDIS sensitivity under covariance inflation tau

Math (tau convention: N = tau * I_d, covariance inflation):
  Loss:  L(theta; D) = sum_i log(1 + exp(-y_i * x_i^T theta)) + lambda/2 * ||theta||^2
  Mean:  theta_hat = argmin_theta L(theta; D)
  Cov:   Sigma = (lambda * I_d + 1/4 * X^T X)^{-1}   [Böhning bound, NOT Laplace at theta_hat]

Sensitivity (Proposition 7, with sigma -> tau substitution):
  Delta(tau)   = (l / lambda) / sqrt(tau + 1 / (lambda + n*l^2/4))
  rho_inf(tau) = log((tau + 1/lambda) / (tau + 1/(lambda + n*l^2/4)))
  nu(tau)      = d * rho_inf(tau)

where l = public row-norm bound (||x_i||_2 <= l for all records).

CRITICAL: l = C_X only (feature row norm), NOT sqrt(C_X^2 + C_Y^2).
          Labels are NOT included in x_i for logistic regression.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from ndis_gaussian.interfaces import GaussianOutputAlgorithm
from ndis_gaussian.types import GaussianOutput, NDISSensitivity


class BLRGaussianOutput(GaussianOutputAlgorithm):
    """Böhning-style BLR surrogate (Definition 7, Proposition 7).

    Fits a regularized logistic regression and returns the (theta_hat, Sigma)
    pair where Sigma uses the Böhning upper-bound on the Hessian rather than
    the standard Laplace approximation at theta_hat. This is required for
    Proposition 7 to hold.

    Parameters
    ----------
    lambda_reg : float
        Ridge regularization strength (lambda > 0 required by Proposition 7).
    """

    def __init__(self, lambda_reg: float) -> None:
        if lambda_reg <= 0.0:
            raise ValueError(f"lambda_reg must be > 0, got {lambda_reg}")
        self.lambda_reg = float(lambda_reg)
        self._fitted_output: GaussianOutput | None = None

    # ------------------------------------------------------------------
    # GaussianOutputAlgorithm interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianOutput:
        """Fit BLR via L-BFGS-B and return (theta_hat, Sigma_Bohning).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,), values in {-1, +1}

        Returns
        -------
        GaussianOutput
            mu = theta_hat (shape d), Sigma = Böhning covariance (shape d x d).
        """
        n, d = X.shape
        lam = self.lambda_reg

        def _loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            # Paper Definition 7: L(theta; D) = sum log(1+exp(-y_i x_i^T theta)) + lambda/2 ||theta||^2
            margins = y * (X @ theta)          # shape (n,)
            logits = np.logaddexp(0.0, -margins)  # log(1 + exp(-margin)), numerically stable
            loss = logits.sum() + 0.5 * lam * np.dot(theta, theta)

            # Gradient: d/dtheta = -sum_i y_i * x_i * sigma(-margin_i) + lambda * theta
            # sigma(-z) = 1/(1+exp(z)) = exp(-z)/(1+exp(-z))
            sigmoid_neg = 1.0 / (1.0 + np.exp(margins))  # sigma(-margin_i)
            grad = -(y * sigmoid_neg) @ X + lam * theta

            return float(loss), grad

        theta0 = np.zeros(d)
        result = minimize(
            _loss_and_grad,
            theta0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
        )
        theta_hat = result.x

        # Böhning covariance: Sigma = (lambda * I_d + 1/4 * X^T X)^{-1}
        # Paper Definition 7. NOT the Laplace Hessian at theta_hat.
        gram = X.T @ X          # shape (d, d)
        H = lam * np.eye(d) + 0.25 * gram
        Sigma = np.linalg.inv(H)

        self._fitted_output = GaussianOutput(mu=theta_hat, Sigma=Sigma)
        return self._fitted_output

    def sensitivity(self, tau: float, public_meta: dict) -> NDISSensitivity:
        """Return (Delta, rho_inf, nu) from Proposition 7 at covariance inflation tau.

        Formulae (Proposition 7, with sigma -> tau substitution):
          denom_noise = tau + 1 / (lambda + n * l^2 / 4)
          denom_clean = tau + 1 / lambda

          Delta(tau)   = (l / lambda) / sqrt(denom_noise)
          rho_inf(tau) = log(denom_clean / denom_noise)
          nu(tau)      = d * rho_inf(tau)

        Parameters
        ----------
        tau : float
            Covariance inflation level (N = tau * I_d). tau = 0 is valid.
        public_meta : dict
            Must contain:
              "l" — row-norm bound (||x_i||_2 <= l, feature rows only)
              "n" — number of training records
              "d" — feature dimension

        Returns
        -------
        NDISSensitivity
        """
        l = float(public_meta["l"])
        n = int(public_meta["n"])
        d = int(public_meta["d"])
        lam = self.lambda_reg

        # Proposition 7 denominators
        denom_noise = tau + 1.0 / (lam + n * l ** 2 / 4.0)
        denom_clean = tau + 1.0 / lam

        Delta = (l / lam) / math.sqrt(denom_noise)
        rho_inf = math.log(denom_clean / denom_noise)
        nu = d * rho_inf

        return NDISSensitivity(Delta=Delta, rho_inf=rho_inf, nu=nu, tau=tau)
