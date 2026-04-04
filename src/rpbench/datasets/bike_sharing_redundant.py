"""Bike Sharing with repeated training rows (redundant support).

Each *unique* training example from the canonical OpenML bike-sharing adapter is
stacked ``copies_per_row`` times with identical ``(x, y)``. The test split is
unchanged. This inflates ``n`` while leaving the least-squares solution on the
span of unique rows the same, and is meant to mimic regimes where weak directions
matter for prediction but many redundant rows stabilize subsampled / sketched
geometry (e.g. Poisson RP).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rpbench.config import PreprocessSpec, SplitSpec
from rpbench.datasets.base import DatasetBundle
from rpbench.datasets.bike_sharing import BikeSharingAdapter


def repeat_training_rows(bundle: DatasetBundle, copies_per_row: int) -> DatasetBundle:
    """Stack each training row ``copies_per_row`` times; leave test data unchanged."""
    if copies_per_row < 1:
        raise ValueError("copies_per_row must be >= 1")

    X_train = np.repeat(bundle.X_train, copies_per_row, axis=0)
    y_train = np.repeat(bundle.y_train, copies_per_row)

    meta: dict[str, Any] = dict(bundle.meta)
    meta["dataset"] = "bike_sharing_redundant"
    meta["redundant_copies_per_row"] = int(copies_per_row)
    meta["n_train_unique"] = int(bundle.X_train.shape[0])
    meta["n_train"] = int(X_train.shape[0])

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        meta=meta,
    )


class BikeSharingRedundantAdapter(BikeSharingAdapter):
    """Same as ``bike_sharing`` after split/preprocess, then duplicate each train row.

    Registry key: ``bike_sharing_redundant``.
    """

    name = "bike_sharing_redundant"

    def __init__(self, cache_dir: str = "data/cache", copies_per_row: int = 10) -> None:
        super().__init__(cache_dir=cache_dir)
        self.copies_per_row = int(copies_per_row)

    def load(self, split_spec: SplitSpec, preprocess_spec: PreprocessSpec) -> DatasetBundle:
        base = super().load(split_spec, preprocess_spec)
        return repeat_training_rows(base, self.copies_per_row)
