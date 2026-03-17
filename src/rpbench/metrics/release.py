"""Release-level metrics."""

from __future__ import annotations

import numpy as np


def relative_frobenius_xtx(xtx_true: np.ndarray, xtx_hat: np.ndarray) -> float:
    """Relative Frobenius error: ||X^T X - xtx_hat||_F / ||X^T X||_F."""
    num = np.linalg.norm(xtx_true - xtx_hat, "fro")
    denom = np.linalg.norm(xtx_true, "fro")
    if denom < 1e-15:
        return float("inf")
    return float(num / denom)
