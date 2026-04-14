"""Abstract interface for Gaussian-output algorithms.

Paper: Definition 3 (Gaussian-output algorithm).

Any algorithm that wants to be wrapped by NDISGaussianWrapper must implement
GaussianOutputAlgorithm. The wrapper calls only fit() and sensitivity().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ndis_gaussian.types import GaussianOutput, NDISSensitivity


class GaussianOutputAlgorithm(ABC):
    """Interface for any Gaussian-output algorithm.

    Paper: Definition 3. Implementors supply:
      fit()        : (X, y) -> GaussianOutput   (mean and covariance of posterior)
      sensitivity(): tau -> NDISSensitivity      (Definition 5 bounds as f(tau))

    The wrapper (NDISGaussianWrapper) calls only these two methods, keeping
    BLR-specific logic fully separated from the generic wrapper core.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianOutput:
        """Fit on dataset (X, y) and return the Gaussian output (mu, Sigma).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
            Label convention depends on algorithm (e.g. {-1, +1} for BLR).

        Returns
        -------
        GaussianOutput
            mu shape (m,), Sigma shape (m, m), Sigma PSD.
        """

    @abstractmethod
    def sensitivity(self, tau: float, public_meta: dict) -> NDISSensitivity:
        """Return NDIS sensitivity bounds at covariance inflation level tau.

        Parameters
        ----------
        tau : float
            Covariance inflation level (N = tau * I_m). NOT a standard deviation.
            tau = 0 means no additive noise (only the algorithm's own covariance).
        public_meta : dict
            Public dataset metadata (n, d, l, etc.) as returned by
            DatasetAdapter.public_meta().

        Returns
        -------
        NDISSensitivity
            Delta, rho_inf, nu at this tau value.
        """
