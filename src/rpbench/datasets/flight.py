"""Flight dataset adapter.

Source: nycflights13 flights data via the Rdatasets mirror.
  URL: https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/flights.csv

This dataset is used in the full-version paper (Section 7.2 / Experiment 1) as
having 2 features and 327,346 records.  The corresponding data reference points to
https://rpubs.com/salmaeng/linear_regression ("Applying linear regression to
study flights delay"), which uses the nycflights13 R package.

Feature note: the paper's design matrix X has two columns — dep_delay and a
constant intercept column (1.0).  This adapter exposes dep_delay as the sole
feature column (d = 1) so that StandardScaler can be applied without
zero-variance issues.  The constant intercept is therefore not included; the
record count (327,346 after dropna) still matches the paper exactly.

Preprocessing contract: see docs/notes/preprocessing_contract.md.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rpbench.config import PreprocessSpec, SplitSpec
from rpbench.datasets.base import DatasetAdapter, DatasetBundle

_RDATASETS_URL = (
    "https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/flights.csv"
)
_CACHE_FILENAME = "nycflights13_flights.csv"


class FlightAdapter(DatasetAdapter):
    """Dataset adapter for the nycflights13 flight-delay regression task.

    Target: arr_delay (arrival delay in minutes).
    Feature: dep_delay (departure delay in minutes), standardised.
    Records: 327,346 (all 2013 NYC-departure flights with non-null delays).
    """

    name = "flight"

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cached_csv_path(self) -> Path:
        return Path(self.cache_dir) / _CACHE_FILENAME

    def _fetch_csv(self) -> Path:
        cache = Path(self.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        dest = self._cached_csv_path()
        if not dest.exists():
            urllib.request.urlretrieve(_RDATASETS_URL, dest)
        return dest

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

    # ------------------------------------------------------------------
    # DatasetAdapter interface
    # ------------------------------------------------------------------

    def load(self, split_spec: SplitSpec, preprocess_spec: PreprocessSpec) -> DatasetBundle:
        csv_path = self._fetch_csv()
        df = pd.read_csv(csv_path, index_col=0)

        # Keep only the two relevant columns and drop rows where either is NA.
        df = df[["dep_delay", "arr_delay"]].dropna().reset_index(drop=True)

        X = df[["dep_delay"]].values.astype(np.float64)   # shape (n, 1)
        y = df["arr_delay"].values.astype(np.float64)

        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=split_spec.train_fraction,
            random_state=split_spec.seed,
            shuffle=True,
        )

        # Scale X (fit on train only)
        if preprocess_spec.scale_x:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Standardise y
        y_mean = float(y_train.mean())
        y_std = float(y_train.std())
        if y_std > 0:
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

        # Clip X rows to L2 bound
        if preprocess_spec.clip_x:
            C_X = float(preprocess_spec.clip_x_bound)
            X_train = self._clip_rows_to_l2_bound(X_train, C_X)
            X_test = self._clip_rows_to_l2_bound(X_test, C_X)
        else:
            C_X = float(np.max(np.linalg.norm(X_train, axis=1)))

        # Clip y values to absolute-value bound
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
            "feature_columns": ["dep_delay"],
            "target_column": "arr_delay",
            "C_X": C_X,
            "C_Y": C_Y,
            "y_mean": y_mean,
            "y_std": y_std,
            "source_url": _RDATASETS_URL,
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
