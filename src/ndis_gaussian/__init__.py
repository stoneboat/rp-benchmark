"""ndis_gaussian — NDIS Gaussian-output mechanism framework.

Public surface:
  GaussianOutput          — (mu, Sigma) pair from a fitted algorithm
  NDISSensitivity         — (Delta, rho_inf, nu, tau) sensitivity bounds
  WrappedRelease          — output of the NDIS-Calibrated Gaussian Mechanism
  GaussianOutputAlgorithm — ABC for plug-and-play Gaussian-output algorithms
  NDISGaussianWrapper     — the generic wrapper (Figure 5)
  ScalarGPRExactNDISWrapper — scalar-GPR exact covariance specialization
  BLRGaussianOutput       — Böhning-style BLR algorithm (Definition 7, Prop. 7)
  GPRGaussianOutput       — Single-query GPR algorithm (Definition 8, Prop. 8)
  RBFKernel               — RBF kernel helper for GPRGaussianOutput
"""

from ndis_gaussian.types import GaussianOutput, NDISSensitivity, WrappedRelease
from ndis_gaussian.interfaces import GaussianOutputAlgorithm
from ndis_gaussian.wrapper import NDISGaussianWrapper, ScalarGPRExactNDISWrapper
from ndis_gaussian.blr.model import BLRGaussianOutput
from ndis_gaussian.gpr.model import GPRGaussianOutput, RBFKernel

__all__ = [
    "GaussianOutput",
    "NDISSensitivity",
    "WrappedRelease",
    "GaussianOutputAlgorithm",
    "NDISGaussianWrapper",
    "ScalarGPRExactNDISWrapper",
    "BLRGaussianOutput",
    "GPRGaussianOutput",
    "RBFKernel",
]
