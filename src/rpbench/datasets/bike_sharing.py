"""Bike Sharing dataset adapter.

**Canonical OpenML identity:** `data_id=42713` (Bike Sharing Demand — hourly,
OpenML). We load only this dataset so runs are reproducible; row counts,
columns, and eigen-spectra do not depend on a multi-candidate fallback chain.

`DatasetBundle.meta["openml"]` stores `data_id` and a stringified copy of
sklearn's `bunch.details` for provenance.

Numeric features only; target is the first matching column in
`TARGET_CANDIDATES` (e.g. `cnt` or `count`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rpbench.config import PreprocessSpec, SplitSpec
from rpbench.datasets.base import DatasetAdapter, DatasetBundle


class BikeSharingAdapter(DatasetAdapter):
    """Dataset adapter for Bike Sharing demand regression."""

    name = "bike_sharing"

    #: Single OpenML dataset id (no fallback). See module docstring.
    OPENML_DATA_ID = 42713

    TARGET_CANDIDATES = ["cnt", "count", "casual", "registered", "class"]

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir

    @staticmethod
    def _clip_rows_to_l2_bound(X: np.ndarray, bound: float) -> np.ndarray:
        if bound <= 0:
            raise ValueError("clip_x_bound must be > 0 when clip_x is enabled")
        norms = np.linalg.norm(X, axis=1)
        scales = np.ones_like(norms)
        mask = norms > bound
        scales[mask] = bound / norms[mask]
        return X * scales[:, np.newaxis]

    def _fetch_bike_sharing(self, cache_dir: str):
        """Fetch the single canonical OpenML dataset; see OPENML_DATA_ID."""
        kwargs: dict[str, Any] = {
            "data_id": self.OPENML_DATA_ID,
            "as_frame": True,
            "data_home": cache_dir,
        }
        try:
            return fetch_openml(parser="auto", **kwargs)
        except TypeError:
            kwargs.pop("parser", None)
            return fetch_openml(**kwargs)

    @staticmethod
    def _openml_provenance(bunch: Any, data_id: int) -> dict[str, Any]:
        """JSON-friendly OpenML provenance from sklearn fetch_openml Bunch."""
        details = getattr(bunch, "details", None)
        details_out: dict[str, str] = {}
        if isinstance(details, dict):
            for k, v in details.items():
                details_out[str(k)] = "" if v is None else str(v)
        elif details is not None:
            details_out["_raw"] = str(details)

        return {
            "data_id": data_id,
            "details": details_out,
        }

    def load(self, split_spec: SplitSpec, preprocess_spec: PreprocessSpec) -> DatasetBundle:
        cache = Path(self.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)

        bunch = self._fetch_bike_sharing(str(cache))
        df = bunch.frame  # type: ignore[union-attr]
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.dropna().reset_index(drop=True)

        target_col = None
        for c in self.TARGET_CANDIDATES:
            if c in df.columns:
                target_col = c
                break
        if target_col is None:
            raise ValueError(f"Could not locate Bike Sharing target column in {list(df.columns)}")

        # Keep numeric predictors only (simple/robust Stage-1 choice).
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if target_col not in numeric_df.columns:
            numeric_df[target_col] = df[target_col].astype(np.float64)

        X_df = numeric_df.drop(columns=[target_col])
        y = numeric_df[target_col].values.astype(np.float64)
        X = X_df.values.astype(np.float64)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=split_spec.train_fraction,
            random_state=split_spec.seed,
            shuffle=True,
        )

        if preprocess_spec.scale_x:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        y_mean = y_train.mean()
        y_std = y_train.std()
        if y_std > 0:
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

        if preprocess_spec.clip_x:
            C_X = float(preprocess_spec.clip_x_bound)
            X_train = self._clip_rows_to_l2_bound(X_train, C_X)
            X_test = self._clip_rows_to_l2_bound(X_test, C_X)
        else:
            C_X = float(np.max(np.linalg.norm(X_train, axis=1)))

        if preprocess_spec.clip_y:
            C_Y = float(preprocess_spec.clip_y_bound)
            if C_Y <= 0:
                raise ValueError("clip_y_bound must be > 0 when clip_y is enabled")
            y_train = np.clip(y_train, -C_Y, C_Y)
            y_test = np.clip(y_test, -C_Y, C_Y)
        else:
            C_Y = float(np.max(np.abs(y_train)))

        meta = {
            "dataset": self.name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "d": X_train.shape[1],
            "feature_columns": list(X_df.columns),
            "target_column": target_col,
            "C_X": float(C_X),
            "C_Y": float(C_Y),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "openml": self._openml_provenance(bunch, self.OPENML_DATA_ID),
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
        }

