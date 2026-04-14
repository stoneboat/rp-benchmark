"""Core dataclasses for the NDIS Gaussian-output framework.

Paper references:
  Definition 3  — Gaussian-output algorithm (mu, Sigma pair)
  Definition 5  — Loewner-comparable NDIS sensitivity (Delta, rho_inf, nu, tau)
  Figure 5      — NDIS-Calibrated Gaussian Mechanism output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GaussianOutput:
    """The (mu, Sigma) pair produced by a fitted Gaussian-output algorithm.

    Paper: Definition 3. q(D) = (mu(D), Sigma(D)).
    """

    mu: np.ndarray     # shape (m,)    — posterior mean
    Sigma: np.ndarray  # shape (m, m), PSD — posterior covariance


@dataclass
class NDISSensitivity:
    """Loewner-comparable NDIS sensitivity bounds at a given tau.

    Paper: Definition 5. All three bounds are functions of tau (the additive
    covariance inflation level N = tau * I_m).

    Attributes
    ----------
    Delta : float
        ||m_N(D, D')||_2 <= Delta   (mean sensitivity bound)
    rho_inf : float
        max_i |ell_{N,i}| <= rho_inf  (per-dimension log-eigenvalue bound)
    nu : float
        |sum_i ell_{N,i}| <= nu       (total log-eigenvalue bound)
    tau : float
        Covariance inflation used to compute this sensitivity (NOT a std dev).
        N = tau * I_m.
    """

    Delta: float
    rho_inf: float
    nu: float
    tau: float  # covariance inflation level at which these bounds hold


@dataclass
class WrappedRelease:
    """Output of the NDIS-Calibrated Gaussian Mechanism (Figure 5).

    The private release theta_tilde is drawn from N(mu, Sigma + tau_star * I_m).

    IMPORTANT: tau_star is a covariance (units: variance), NOT a standard
    deviation. The additive noise matrix is tau_star * I_m, not tau_star^2 * I_m.
    """

    theta_tilde: np.ndarray      # shape (m,) — private parameter release
    mu: np.ndarray               # shape (m,) — pre-noise posterior mean
    Sigma_noised: np.ndarray     # shape (m, m) — Sigma(D) + tau_star * I_m
    tau_star: float              # calibrated covariance inflation (NOT a std dev)
    sensitivity: NDISSensitivity  # sensitivity bounds at tau_star
    diagnostics: dict[str, Any] = field(default_factory=dict)
