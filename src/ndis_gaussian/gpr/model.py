"""GPR Gaussian-output algorithm: Definition 8 + Proposition 8.

Gaussian Process Regression at a fixed public query point x_★ (scalar output).

Paper references:
  Definition 8  — GPR as Gaussian-output algorithm
  Proposition 8 — GPR NDIS sensitivity under covariance inflation tau

Math (scalar output, m = 1):
  Gram matrix:   K_{ij} = k(x_i, x_j)
  Kernel vector: (k_★)_i = k(x_★, x_i)
  k_★★         = k(x_★, x_★)
  A            = K + σ_n^2 * I_n

  Mean:     μ_★ = k_★^T A^{-1} y          (scalar)
  Variance: Σ_★ = k_★★ - k_★^T A^{-1} k_★  (scalar, ≥ 0)

  Stored as GaussianOutput(mu=array([μ_★]), Sigma=array([[Σ_★]])).

Sensitivity (Proposition 8, with sigma -> tau):
  l  = kernel diagonal bound: k(x, x) ≤ l^2 for all x
  B  = label bound: |y| ≤ B
  σ_n^2 = observation noise variance

  Δ(τ)    = l * B * √n / (σ_n * √τ)
  ρ_∞(τ)  = log((τ + l^2) / τ)
  ν(τ)    = ρ_∞(τ)          [scalar output: m = 1, so ν = 1 × ρ_∞]

IMPORTANT: Sensitivity diverges at τ = 0 (unlike BLR which has finite sensitivity
at τ = 0). find_tau_star handles this correctly: at τ = 0 the sentinel returns
delta = 1 > any target delta, so the binary search expands the bracket.

RBF kernel note: k(x, x) = exp(0) = 1 for any x and any lengthscale.
Therefore l = 1 when using an RBF kernel regardless of lengthscale.

Public query point: x_★ is fixed and public (part of the algorithm specification).
This phase implements single-query GPR only. Multi-query releases require
privacy composition and are out of scope for Phase 2.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ndis_gaussian.interfaces import GaussianOutputAlgorithm
from ndis_gaussian.types import GaussianOutput, NDISSensitivity


# ---------------------------------------------------------------------------
# RBF kernel helper
# ---------------------------------------------------------------------------

class RBFKernel:
    """Radial Basis Function (squared-exponential) kernel.

    k(x, x') = exp(-||x - x'||^2 / (2 * lengthscale^2))

    Properties relevant to NDIS:
      - k(x, x) = 1 for all x (regardless of lengthscale)
      - Therefore the kernel diagonal bound is l = 1 (i.e., k(x,x) ≤ 1^2)
    """

    def __init__(self, lengthscale: float = 1.0) -> None:
        if lengthscale <= 0.0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}")
        self.lengthscale = float(lengthscale)
        # Kernel diagonal bound: k(x,x) = 1 ≤ l^2 = 1^2
        self.l_bound: float = 1.0

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate k(x1, x2)."""
        diff = x1 - x2
        return float(np.exp(-0.5 * np.dot(diff, diff) / (self.lengthscale ** 2)))

    def gram(self, X: np.ndarray) -> np.ndarray:
        """Compute the n×n Gram matrix K_{ij} = k(x_i, x_j) efficiently."""
        n = X.shape[0]
        ls2 = self.lengthscale ** 2
        # Pairwise squared distances via broadcasting
        # ||x_i - x_j||^2 = ||x_i||^2 - 2 x_i^T x_j + ||x_j||^2
        sq_norms = np.sum(X ** 2, axis=1)  # shape (n,)
        sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
        sq_dist = np.maximum(sq_dist, 0.0)  # numerical safety
        return np.exp(-0.5 * sq_dist / ls2)

    def k_star(self, X: np.ndarray, x_star: np.ndarray) -> np.ndarray:
        """Compute kernel vector (k_★)_i = k(x_★, x_i), shape (n,)."""
        diff = X - x_star[None, :]      # shape (n, d)
        sq_dist = np.sum(diff ** 2, axis=1)  # shape (n,)
        return np.exp(-0.5 * sq_dist / (self.lengthscale ** 2))

    def k_starstar(self, x_star: np.ndarray) -> float:
        """k(x_★, x_★) = 1 for the RBF kernel."""
        return 1.0


# ---------------------------------------------------------------------------
# GPR Gaussian-output algorithm
# ---------------------------------------------------------------------------

class GPRGaussianOutput(GaussianOutputAlgorithm):
    """Single-query GPR surrogate (Definition 8, Proposition 8).

    Implements Gaussian Process Regression as a Gaussian-output algorithm
    for a FIXED public query point x_★. The output is scalar (m = 1):

        q_GPR(D) = (μ_★, Σ_★)  where  μ_★ ∈ R,  Σ_★ ∈ R_{≥ 0}

    Stored as GaussianOutput(mu=array([μ_★]), Sigma=array([[Σ_★]])).

    Parameters
    ----------
    x_star : np.ndarray, shape (d,)
        Fixed public query point. MUST be public (not derived from the
        private dataset) to preserve the DP guarantee.
    kernel : RBFKernel or callable (x1, x2) -> float
        Positive-definite kernel. For NDIS sensitivity (Proposition 8),
        the kernel diagonal bound l must satisfy k(x, x) ≤ l^2 for all x.
        For the RBF kernel, l = 1.0 (exact equality).
    sigma_n2 : float
        Observation noise variance σ_n^2 > 0. Larger σ_n^2 reduces
        sensitivity (Δ ∝ 1/σ_n) at the cost of worse non-private utility.
    B_bound : float
        Label bound B: |y_i| ≤ B for all training records. Must be a
        PUBLIC upper bound (not computed from the private labels).
    l_bound : float
        Kernel diagonal bound l: k(x, x) ≤ l^2 for all x ∈ X.
        For the RBF kernel, l = 1.0 exactly.

    Sensitivity note (tau = 0 is undefined):
        Proposition 8's Δ(τ) = lB√n/(σ_n√τ) diverges at τ = 0.
        sensitivity() returns a sentinel NDISSensitivity with Delta=inf
        and rho_inf = nu = 0.0 at τ ≤ 0, which correctly maps to delta = 1.
        find_tau_star() handles this via the bracket-expansion pattern.

    This class is for SINGLE-QUERY GPR only. To release predictions for
    multiple query points, privacy composition must be applied explicitly.
    """

    def __init__(
        self,
        x_star: np.ndarray,
        kernel: RBFKernel | Callable,
        sigma_n2: float,
        B_bound: float,
        l_bound: float,
    ) -> None:
        if sigma_n2 <= 0.0:
            raise ValueError(f"sigma_n2 must be > 0, got {sigma_n2}")
        if B_bound <= 0.0:
            raise ValueError(f"B_bound must be > 0, got {B_bound}")
        if l_bound <= 0.0:
            raise ValueError(f"l_bound must be > 0, got {l_bound}")

        self.x_star = np.asarray(x_star, dtype=np.float64)
        self.kernel = kernel
        self.sigma_n2 = float(sigma_n2)
        self.B_bound = float(B_bound)
        self.l_bound = float(l_bound)

        self._n_train: int | None = None
        self._diagnostics: dict = {}

    # ------------------------------------------------------------------
    # GaussianOutputAlgorithm interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianOutput:
        """Fit GPR and return scalar Gaussian output at x_★ (Definition 8).

        Computes:
          A = K + σ_n^2 * I_n
          μ_★ = k_★^T A^{-1} y          (scalar posterior mean)
          Σ_★ = k_★★ - k_★^T A^{-1} k_★  (scalar posterior variance)

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
            Training labels. Must satisfy |y_i| ≤ B_bound.

        Returns
        -------
        GaussianOutput
            mu shape (1,), Sigma shape (1, 1).
        """
        n, d = X.shape
        self._n_train = n

        # Compute Gram matrix K ∈ R^{n×n}
        if isinstance(self.kernel, RBFKernel):
            K = self.kernel.gram(X)
            k_vec = self.kernel.k_star(X, self.x_star)
            k_ss = self.kernel.k_starstar(self.x_star)
        else:
            # Fallback: element-wise (slow, for generic kernels)
            K = np.array([[self.kernel(X[i], X[j]) for j in range(n)] for i in range(n)])
            k_vec = np.array([self.kernel(self.x_star, X[i]) for i in range(n)])
            k_ss = float(self.kernel(self.x_star, self.x_star))

        # A = K + σ_n^2 * I_n
        A = K + self.sigma_n2 * np.eye(n)

        # Solve linear systems (more numerically stable than explicit A^{-1})
        # alpha = A^{-1} y
        alpha = np.linalg.solve(A, y)
        # v = A^{-1} k_★
        v = np.linalg.solve(A, k_vec)

        # Posterior mean: μ_★ = k_★^T α  (scalar)
        mu_star = float(k_vec @ alpha)

        # Posterior variance: Σ_★ = k_★★ - k_★^T v  (scalar, clamped ≥ 0)
        Sigma_star = float(k_ss - k_vec @ v)
        Sigma_star = max(Sigma_star, 0.0)   # numerical safety: should be ≥ 0

        self._diagnostics = {
            "n_train": n,
            "d": d,
            "mu_star": mu_star,
            "Sigma_star": Sigma_star,
            "sigma_n2": self.sigma_n2,
        }

        return GaussianOutput(
            mu=np.array([mu_star]),
            Sigma=np.array([[Sigma_star]]),
        )

    def sensitivity(self, tau: float, public_meta: dict) -> NDISSensitivity:
        """Return (Δ, ρ_∞, ν) from Proposition 8 at covariance inflation τ.

        Formulae (Proposition 8, with sigma -> tau):
          l^2  = l_bound^2   (kernel diagonal bound)
          n    = public_meta["n"]
          B    = public_meta["B"]
          σ_n^2 = public_meta["sigma_n2"]

          Δ(τ)   = l * B * √n / (σ_n * √τ)
          ρ_∞(τ) = log((τ + l^2) / τ)
          ν(τ)   = ρ_∞(τ)         [scalar output m=1: ν = 1 × ρ_∞]

        Parameters
        ----------
        tau : float
            Covariance inflation (N = tau * I_1). Must be > 0 for finite
            sensitivity (Δ diverges at tau = 0). If tau ≤ 0, returns a
            sentinel: Delta=inf, rho_inf=0.0, nu=0.0, which maps to delta=1
            in delta_bar_LC and correctly signals that no privacy is guaranteed.
        public_meta : dict
            Must contain:
              "n"        — number of training records (int)
              "B"        — label bound: |y_i| ≤ B  (float)
              "l"        — kernel diagonal bound: k(x,x) ≤ l^2  (float)
              "sigma_n2" — observation noise variance σ_n^2 (float)

        Returns
        -------
        NDISSensitivity
            Note: nu = rho_inf (scalar output, m = 1).
            The 'd' parameter for delta_bar_LC must be set to 1 (output dim).
        """
        if tau <= 0.0:
            # Sentinel: sensitivity is infinite at tau=0; delta_bar_LC maps to 1.
            # rho_inf=0, nu=0 → delta_bar_cov=0; delta_bar_mean(eps, inf)=1.
            return NDISSensitivity(Delta=math.inf, rho_inf=0.0, nu=0.0, tau=tau)

        n = int(public_meta["n"])
        B = float(public_meta["B"])
        l = float(public_meta["l"])
        sigma_n2 = float(public_meta["sigma_n2"])

        l2 = l * l

        # Proposition 8 formulas
        Delta = l * B * math.sqrt(n) / (math.sqrt(sigma_n2) * math.sqrt(tau))
        rho_inf = math.log((tau + l2) / tau)   # = log(1 + l^2/tau) > 0
        nu = rho_inf                            # scalar output: m = 1

        return NDISSensitivity(Delta=Delta, rho_inf=rho_inf, nu=nu, tau=tau)

    # ------------------------------------------------------------------
    # Properties / diagnostics
    # ------------------------------------------------------------------

    @property
    def diagnostics(self) -> dict:
        """Diagnostic information from the last fit() call."""
        return dict(self._diagnostics)
