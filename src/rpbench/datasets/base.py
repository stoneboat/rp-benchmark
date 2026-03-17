"""Base class for dataset adapters and the DatasetBundle container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rpbench.config import SplitSpec, PreprocessSpec


@dataclass
class DatasetBundle:
    """Container returned by DatasetAdapter.load()."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Common interface for dataset loading + preprocessing."""

    name: str = "BaseDataset"

    @abstractmethod
    def load(self, split_spec: SplitSpec, preprocess_spec: PreprocessSpec) -> DatasetBundle:
        """Load, preprocess, split, and return the dataset."""

    @abstractmethod
    def public_meta(self, bundle: DatasetBundle) -> dict[str, Any]:
        """Return public metadata needed by mechanisms (n, d, l, etc.)."""
