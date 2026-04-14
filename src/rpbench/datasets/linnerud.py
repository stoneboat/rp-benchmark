"""Linnerud dataset adapter for scalar-target GPR demos.

Source: sklearn.datasets.load_linnerud()

Linnerud has 20 samples, 3 exercise features, and 3 physiological targets.
The current GPR NDIS path is scalar-output and fixed-query, so this adapter
selects exactly one target column via ``target_index`` and returns one-dimensional
``y_train`` / ``y_test`` arrays.

Default target_index=0 selects the first sklearn target, "Weight".
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rpbench.config import SplitSpec, PreprocessSpec
from rpbench.datasets.base import DatasetAdapter, DatasetBundle


class LinnerudAdapter(DatasetAdapter):
    """Dataset adapter for sklearn's Linnerud regression dataset.

    The adapter is intentionally scalar-target: use ``target_index`` to choose
    one of sklearn's target columns. It normalizes the selected target by the
    training maximum absolute value, matching the standalone Tecator GPR path's
    simple bounded-target convention. This gives C_Y = 1.0 after normalization
    unless the selected training target is identically zero.
    """

    name = "linnerud"

    def __init__(self, target_index: int = 0) -> None:
        if target_index < 0:
            raise ValueError(f"target_index must be non-negative, got {target_index}")
        self.target_index = int(target_index)

    @staticmethod
    def _clip_rows_to_l2_bound(X: np.ndarray, bound: float) -> np.ndarray:
        """Rescale rows so that ||x_i||_2 <= bound."""
        if bound <= 0:
            raise ValueError("clip_x_bound must be > 0 when clip_x is enabled")
        norms = np.linalg.norm(X, axis=1)
        scales = np.ones_like(norms)
        mask = norms > bound
        scales[mask] = bound / norms[mask]
        return X * scales[:, np.newaxis]

    def load(self, split_spec: SplitSpec, preprocess_spec: PreprocessSpec) -> DatasetBundle:
        bunch = load_linnerud()
        X_all = np.asarray(bunch.data, dtype=np.float64)
        y_all_multi = np.asarray(bunch.target, dtype=np.float64)

        target_names = [str(name) for name in bunch.target_names]
        feature_names = [str(name) for name in bunch.feature_names]
        if self.target_index >= y_all_multi.shape[1]:
            raise ValueError(
                f"target_index={self.target_index} out of range for Linnerud "
                f"targets {target_names}"
            )

        target_name = target_names[self.target_index]
        y_all = y_all_multi[:, self.target_index]

        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            train_size=split_spec.train_fraction,
            random_state=split_spec.seed,
            shuffle=True,
        )

        if preprocess_spec.scale_x:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if preprocess_spec.clip_x:
            C_X = float(preprocess_spec.clip_x_bound)
            X_train = self._clip_rows_to_l2_bound(X_train, C_X)
            X_test = self._clip_rows_to_l2_bound(X_test, C_X)
        else:
            C_X = float(np.max(np.linalg.norm(X_train, axis=1)))

        y_scale = float(np.max(np.abs(y_train)))
        if y_scale <= 0.0:
            raise ValueError("Linnerud selected target is constant zero on the training split")
        y_train = y_train / y_scale
        y_test = y_test / y_scale

        if preprocess_spec.clip_y:
            C_Y = float(preprocess_spec.clip_y_bound)
            if C_Y <= 0:
                raise ValueError("clip_y_bound must be > 0 when clip_y is enabled")
            y_train = np.clip(y_train, -C_Y, C_Y)
            y_test = np.clip(y_test, -C_Y, C_Y)
            C_Y = min(C_Y, float(np.max(np.abs(y_train))))
        else:
            C_Y = float(np.max(np.abs(y_train)))

        meta = {
            "dataset": self.name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "d": X_train.shape[1],
            "feature_columns": feature_names,
            "target_names": target_names,
            "target_index": self.target_index,
            "target_name": target_name,
            "split_seed": int(split_spec.seed),
            "train_fraction": float(split_spec.train_fraction),
            "scale_x": bool(preprocess_spec.scale_x),
            "clip_x": bool(preprocess_spec.clip_x),
            "clip_y": bool(preprocess_spec.clip_y),
            "C_X": float(C_X),
            "C_Y": float(C_Y),
            "y_scale": y_scale,
        }

        return DatasetBundle(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            meta=meta,
        )

    def public_meta(self, bundle: DatasetBundle) -> dict[str, Any]:
        return {
            "n": bundle.meta["n_train"],
            "d": bundle.meta["d"],
            "l": 1.0,  # RBF kernel diagonal bound for the standalone GPR path.
            "B": bundle.meta["C_Y"],
            "C_X": bundle.meta["C_X"],
            "C_Y": bundle.meta["C_Y"],
            "target_index": bundle.meta["target_index"],
            "target_name": bundle.meta["target_name"],
        }
