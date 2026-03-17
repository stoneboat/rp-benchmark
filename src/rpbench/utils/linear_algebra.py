"""Linear algebra helpers."""

from __future__ import annotations

import numpy as np


def gram_matrix(X: np.ndarray) -> np.ndarray:
    """Compute X^T X."""
    return X.T @ X


def augmented_data(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stack features and labels: [X | y] of shape (n, d+1)."""
    return np.column_stack([X, y])
