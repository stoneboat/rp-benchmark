"""OLS downstream task: fit OLS from ReleaseBundle, evaluate test MSE."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpbench.mechanisms.base import ReleaseBundle
from rpbench.tasks.base import Task


class OLSFromRelease(Task):
    """Ordinary Least Squares fitted from released X^T X and X^T y."""

    name = "OLSFromRelease"

    def fit_from_release(self, release_bundle: ReleaseBundle, train_meta: dict) -> np.ndarray:
        xtx = release_bundle.xtx_hat
        xty = release_bundle.xty_hat

        try:
            beta_hat = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            beta_hat, *_ = np.linalg.lstsq(xtx, xty, rcond=None)

        return beta_hat

    def evaluate(self, fitted_obj: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        beta_hat = fitted_obj
        y_pred = X_test @ beta_hat
        test_mse = float(np.mean((y_test - y_pred) ** 2))
        return {"test_mse": test_mse}
