"""Base classes for DP mechanisms and the shared ReleaseBundle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ReleaseBundle:
    """Shared return object for all mechanisms.

    Every mechanism must populate at least ``xtx_hat`` and ``xty_hat``
    so that downstream tasks and release-level metrics can consume the
    release uniformly.
    """

    mechanism_name: str
    release_kind: str  # e.g. "gram_blocks", "sketch"
    xtx_hat: np.ndarray | None = None
    xty_hat: np.ndarray | None = None
    sketch_matrix: np.ndarray | None = None
    beta_hat: np.ndarray | None = None
    calibration: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    runtime_sec: float = 0.0


class Mechanism(ABC):
    """Common interface for all DP mechanisms in the benchmark."""

    name: str = "BaseMechanism"

    @abstractmethod
    def calibrate(self, privacy_spec, public_meta: dict) -> None:
        """Set internal parameters from (epsilon, delta) and public metadata."""

    @abstractmethod
    def release(self, train_data: np.ndarray, seed: int) -> ReleaseBundle:
        """Produce a private release from training data.

        Parameters
        ----------
        train_data : np.ndarray
            Augmented data ``[X | y]`` of shape ``(n, d+1)``.
        seed : int
            RNG seed for this trial.
        """

    def diagnostics(self) -> dict[str, Any]:
        """Return calibration/runtime diagnostics."""
        return {}
