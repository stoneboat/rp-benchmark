"""Downstream-task metrics."""

from __future__ import annotations

import numpy as np


def test_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error on the test set."""
    return float(np.mean((y_true - y_pred) ** 2))
