"""Kaggle Bike Sharing Demand adapter matching the GaussMix setup.

Ground truth: `docs/gaussmix_bike_sharing_report.md`.

This adapter intentionally does not reuse the generic repo preprocessing path.
Instead it reproduces the GaussMix Bike feature engineering, split, and
normalization:

- load local `bike_train.csv`
- one-hot encode `season` and `weather`
- derive `hour`, `day`, `month`, `year` from `datetime`
- map years `{2011, 2012}` to `{0, 1}`
- drop `datetime`, `casual`, `registered`, and original `season` / `weather`
- random permutation split, first 80% train and rest test
- divide y by `max(abs(y_train))`
- divide X by the maximum training-row L2 norm
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rpbench.config import PreprocessSpec, SplitSpec
from rpbench.datasets.base import DatasetAdapter, DatasetBundle


class BikeSharingKaggleAdapter(DatasetAdapter):
    """Dataset adapter for the Kaggle Bike Sharing Demand regression task."""

    name = "bike_sharing_kaggle"
    DEFAULT_CSV_PATH = "data/raw/bike_sharing_kaggle/bike_train.csv"

    def __init__(self, csv_path: str | None = None):
        self.csv_path = csv_path or self.DEFAULT_CSV_PATH

    def _load_csv(self) -> pd.DataFrame:
        csv_path = Path(self.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                "bike_sharing_kaggle requires a local Kaggle Bike Sharing Demand CSV. "
                f"Expected file at '{csv_path}'. "
                "Place Kaggle's bike_train.csv there, or pass dataset_params.csv_path."
            )
        return pd.read_csv(csv_path)

    @staticmethod
    def _normalize_like_gaussmix(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        y_scale = float(np.max(np.abs(y_train)))
        if y_scale <= 0:
            raise ValueError("bike_sharing_kaggle requires non-constant training targets")
        y_train = y_train / y_scale
        y_test = y_test / y_scale

        x_scale = float(np.max(np.linalg.norm(X_train, axis=1)))
        if x_scale <= 0:
            raise ValueError("bike_sharing_kaggle requires non-zero training feature norm")
        X_train = X_train / x_scale
        X_test = X_test / x_scale

        return X_train, y_train, X_test, y_test, x_scale, y_scale

    def load(self, split_spec: SplitSpec, preprocess_spec: PreprocessSpec) -> DatasetBundle:
        del preprocess_spec  # This adapter follows the GaussMix-specific pipeline exactly.

        df = self._load_csv()

        season = pd.get_dummies(df["season"], prefix="season")
        weather = pd.get_dummies(df["weather"], prefix="weather")
        df = pd.concat([df, season, weather], axis=1)

        dt = pd.DatetimeIndex(df["datetime"])
        df["hour"] = dt.hour
        df["day"] = dt.dayofweek
        df["month"] = dt.month
        df["year"] = dt.year.map({2011: 0, 2012: 1})

        df = df.drop(columns=["datetime", "casual", "registered", "season", "weather"])

        feature_columns = [c for c in df.columns if c != "count"]
        X = df[feature_columns].values.astype(np.float64)
        y = df["count"].values.astype(np.float64)

        rng = np.random.RandomState(split_spec.seed)
        p = rng.permutation(len(y))
        X = X[p]
        y = y[p]

        train_size = int(split_spec.train_fraction * len(y))
        if train_size <= 0 or train_size >= len(y):
            raise ValueError("bike_sharing_kaggle requires 0 < train_size < n")

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
            "target_column": "count",
            "csv_path": str(Path(self.csv_path)),
            "split_seed": int(split_spec.seed),
            "train_fraction": float(split_spec.train_fraction),
            "x_scale": x_scale,
            "y_scale": y_scale,
            "C_X": C_X,
            "C_Y": C_Y,
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
