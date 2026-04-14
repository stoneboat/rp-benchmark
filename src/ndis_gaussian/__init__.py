"""ndis_gaussian — NDIS Gaussian-output mechanism framework.

Public surface:
  GaussianOutput       — (mu, Sigma) pair from a fitted algorithm
  NDISSensitivity      — (Delta, rho_inf, nu, tau) sensitivity bounds
  WrappedRelease       — output of the NDIS-Calibrated Gaussian Mechanism
  GaussianOutputAlgorithm — ABC for plug-and-play Gaussian-output algorithms
  NDISGaussianWrapper  — the generic wrapper (Figure 5)
  BLRGaussianOutput    — Böhning-style BLR algorithm (Definition 7, Proposition 7)
"""

from ndis_gaussian.types import GaussianOutput, NDISSensitivity, WrappedRelease
from ndis_gaussian.interfaces import GaussianOutputAlgorithm
from ndis_gaussian.wrapper import NDISGaussianWrapper
from ndis_gaussian.blr.model import BLRGaussianOutput

__all__ = [
    "GaussianOutput",
    "NDISSensitivity",
    "WrappedRelease",
    "GaussianOutputAlgorithm",
    "NDISGaussianWrapper",
    "BLRGaussianOutput",
]
