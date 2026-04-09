"""Tecator adapter matching the committed GaussMix linear-regression setup.

Reference implementation:
  - GaussMix: ``utils_linear_mixing.py``, ``dataset_name == 'tecator'``
  - shared post-branch normalization at the end of ``GetDataset(...)``

This adapter intentionally bypasses the benchmark's generic preprocessing
contract and instead reproduces the GaussMix Tecator path:

- fetch OpenML dataset ``name='Tecator', version=1``
- preserve the OpenML feature order
- random permutation split, first 80% train and remainder test
- divide ``y`` by ``max(abs(y_train))``
- divide ``X`` by the maximum training-row L2 norm
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml

from rpbench.config import PreprocessSpec, SplitSpec
from rpbench.datasets.base import DatasetAdapter, DatasetBundle


class TecatorAdapter(DatasetAdapter):
    """Dataset adapter for the GaussMix-style Tecator regression task."""

    name = "tecator"

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir

    def _fetch_tecator(self):
        cache = Path(self.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {
            "name": "Tecator",
            "version": 1,
            "as_frame": True,
            "data_home": str(cache),
        }
        try:
            return fetch_openml(parser="auto", **kwargs)
        except TypeError:
            kwargs.pop("parser", None)
            return fetch_openml(**kwargs)

    @staticmethod
    def _openml_provenance(bunch: Any) -> dict[str, Any]:
        details = getattr(bunch, "details", None)
        details_out: dict[str, str] = {}
        if isinstance(details, dict):
            for k, v in details.items():
                details_out[str(k)] = "" if v is None else str(v)
        elif details is not None:
            details_out["_raw"] = str(details)

        return {
            "name": "Tecator",
            "version": 1,
            "details": details_out,
        }

    @staticmethod
    def _normalize_like_gaussmix(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        y_scale = float(np.max(np.abs(y_train)))
        if y_scale <= 0:
            raise ValueError("tecator requires non-constant training targets")
        y_train = y_train / y_scale
        y_test = y_test / y_scale

        x_scale = float(np.max(np.linalg.norm(X_train, axis=1)))
        if x_scale <= 0:
            raise ValueError("tecator requires non-zero training feature norm")
        X_train = X_train / x_scale
        X_test = X_test / x_scale

        return X_train, y_train, X_test, y_test, x_scale, y_scale

    def load(self, split_spec: SplitSpec, preprocess_spec: PreprocessSpec) -> DatasetBundle:
        del preprocess_spec  # This adapter follows the GaussMix Tecator pipeline exactly.

        bunch = self._fetch_tecator()
        X_df = bunch.data  # type: ignore[union-attr]
        y_raw = bunch.target  # type: ignore[union-attr]

        feature_columns = [str(c) for c in X_df.columns]
        X = X_df.to_numpy(dtype=np.float64)
        y = np.asarray(y_raw, dtype=np.float64)
        target_name = getattr(y_raw, "name", None) or "target"

        rng = np.random.RandomState(split_spec.seed)
        p = rng.permutation(len(y))
        X = X[p]
        y = y[p]

        train_size = int(split_spec.train_fraction * len(y))
        if train_size <= 0 or train_size >= len(y):
            raise ValueError("tecator requires 0 < train_size < n")

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]

        X_train, y_train, X_test, y_test, x_scale, y_scale = self._normalize_like_gaussmix(
            X_train, y_train, X_test, y_test
        )

        C_X = float(np.max(np.linalg.norm(X_train, axis=1)))
        C_Y = float(np.max(np.abs(y_train)))

        meta = {
            "dataset": self.name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "d": X_train.shape[1],
            "feature_columns": feature_columns,
            "target_column": str(target_name),
            "split_seed": int(split_spec.seed),
            "train_fraction": float(split_spec.train_fraction),
            "x_scale": x_scale,
            "y_scale": y_scale,
            "C_X": C_X,
            "C_Y": C_Y,
            "openml": self._openml_provenance(bunch),
        }

        return DatasetBundle(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            meta=meta,
        )

    def public_meta(self, bundle: DatasetBundle) -> dict[str, Any]:
        l = float(np.sqrt(bundle.meta["C_X"] ** 2 + bundle.meta["C_Y"] ** 2))
        return {
            "n": bundle.meta["n_train"],
            "d": bundle.meta["d"],
            "d_aug": bundle.meta["d"] + 1,
            "l": l,
            "C_X": bundle.meta["C_X"],
            "C_Y": bundle.meta["C_Y"],
            "n_test": bundle.meta["n_test"],
        }
