"""Base class for downstream tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from rpbench.mechanisms.base import ReleaseBundle


class Task(ABC):
    """Common interface for downstream tasks that consume a ReleaseBundle."""

    name: str = "BaseTask"

    @abstractmethod
    def fit_from_release(self, release_bundle: ReleaseBundle, train_meta: dict) -> Any:
        """Fit a model from the private release. Returns a fitted object."""

    @abstractmethod
    def evaluate(self, fitted_obj: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Evaluate the fitted object on test data. Returns metric dict."""
