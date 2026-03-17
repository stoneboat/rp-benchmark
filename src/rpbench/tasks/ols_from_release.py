"""OLS downstream task: fit OLS from ReleaseBundle, evaluate test MSE."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpbench.mechanisms.base import ReleaseBundle
from rpbench.tasks.base import Task


class OLSFromRelease(Task):
    """Ordinary Least Squares fitted from released X^T X and X^T y."""

    name = "OLSFromRelease"

    def __init__(self, rcond: float = 1e-6):
        self.rcond = rcond

    def _spectral_pinv_solve(self, xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
        """Solve a symmetric linear system via a truncated spectral pseudoinverse."""
        xtx_sym = 0.5 * (xtx + xtx.T)
        eigvals, eigvecs = np.linalg.eigh(xtx_sym)
        scale = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        cutoff = max(self.rcond * scale, 0.0)

        inv_eigvals = np.zeros_like(eigvals)
        keep = eigvals > cutoff
        inv_eigvals[keep] = 1.0 / eigvals[keep]
        return eigvecs @ (inv_eigvals * (eigvecs.T @ xty))

    def fit_from_release(self, release_bundle: ReleaseBundle, train_meta: dict) -> np.ndarray:
        xtx = release_bundle.xtx_hat
        xty = release_bundle.xty_hat
        assert xtx is not None
        assert xty is not None

        if (
            release_bundle.mechanism_name == "Mech_RP"
            and release_bundle.sketch_matrix is not None
            and "lambda_ridge" in release_bundle.calibration
        ):
            m_tilde = release_bundle.sketch_matrix
            d = xtx.shape[0]
            m_x = m_tilde[:d, :]
            m_y = m_tilde[d, :]
            beta_hat = self._spectral_pinv_solve(m_x @ m_x.T, m_x @ m_y)
        else:
            # Generic covariance-release decoder.
            beta_hat = self._spectral_pinv_solve(xtx, xty)

        return beta_hat

    def evaluate(self, fitted_obj: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        beta_hat = fitted_obj
        y_pred = X_test @ beta_hat
        test_mse = float(np.mean((y_test - y_pred) ** 2))
        return {"test_mse": test_mse}
