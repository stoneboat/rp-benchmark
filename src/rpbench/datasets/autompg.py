"""AutoMPG dataset adapter.

Source: UCI ML Repository via sklearn.datasets.fetch_openml (data_id=196).
Preprocessing contract: see docs/notes/preprocessing_contract.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rpbench.config import SplitSpec, PreprocessSpec
from rpbench.datasets.base import DatasetAdapter, DatasetBundle


class AutoMPGAdapter(DatasetAdapter):
    """Dataset adapter for the Auto MPG regression dataset."""

    name = "autompg"

    FEATURE_COLS = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model",
        "origin",
    ]
    TARGET_COL = "class"

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir

    def load(self, split_spec: SplitSpec, preprocess_spec: PreprocessSpec) -> DatasetBundle:
        cache = Path(self.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)

        fetch_kwargs = {
            "data_id": 196,
            "as_frame": True,
            "data_home": str(cache),
        }
        try:
            bunch = fetch_openml(parser="auto", **fetch_kwargs)
        except TypeError:
            # Older scikit-learn versions do not support the parser kwarg.
            bunch = fetch_openml(**fetch_kwargs)
        df = bunch.frame  # type: ignore[union-attr]

        # Drop rows with missing values
        df = df.dropna().reset_index(drop=True)

        X = df[self.FEATURE_COLS].values.astype(np.float64)
        y = df[self.TARGET_COL].values.astype(np.float64)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=split_spec.train_fraction,
            random_state=split_spec.seed,
            shuffle=True,
        )

        # Scaling: fit on train only
        if preprocess_spec.scale_x:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Label scaling: standardize target
        y_mean = y_train.mean()
        y_std = y_train.std()
        if y_std > 0:
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

        # Row clipping (L2 norm)
        if preprocess_spec.clip_x:
            norms = np.linalg.norm(X_train, axis=1)
            C_X = np.percentile(norms, 95)
            mask_train = norms <= C_X
            X_train = X_train[mask_train]
            y_train = y_train[mask_train]

            norms_test = np.linalg.norm(X_test, axis=1)
            mask_test = norms_test <= C_X
            X_test = X_test[mask_test]
            y_test = y_test[mask_test]
        else:
            C_X = float(np.max(np.linalg.norm(X_train, axis=1)))

        # Label clipping
        if preprocess_spec.clip_y:
            C_Y = np.percentile(np.abs(y_train), 95)
            y_train = np.clip(y_train, -C_Y, C_Y)
            y_test = np.clip(y_test, -C_Y, C_Y)
        else:
            C_Y = float(np.max(np.abs(y_train)))

        meta = {
            "dataset": self.name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "d": X_train.shape[1],
            "C_X": float(C_X),
            "C_Y": float(C_Y),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
        }

        return DatasetBundle(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            meta=meta,
        )

    def public_meta(self, bundle: DatasetBundle) -> dict[str, Any]:
        A_train = np.column_stack([bundle.X_train, bundle.y_train])
        l = float(np.max(np.linalg.norm(A_train, axis=1)))
        return {
            "n": bundle.meta["n_train"],
            "d": bundle.meta["d"],
            "d_aug": bundle.meta["d"] + 1,
            "l": l,
            "C_X": bundle.meta["C_X"],
            "C_Y": bundle.meta["C_Y"],
        }
