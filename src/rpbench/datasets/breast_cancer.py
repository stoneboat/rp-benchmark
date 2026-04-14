"""Wisconsin Diagnostic Breast Cancer dataset adapter.

Source: sklearn.datasets.load_breast_cancer()
Task:   Binary classification (malignant vs. benign)
Labels: y in {-1, +1}  (malignant = +1, benign = -1)

Key differences from regression adapters (e.g. AutoMPGAdapter):
  1. Labels are {-1, +1}, NOT standardized continuous values.
  2. public_meta() returns l = C_X (feature row norm only).
     Regression adapters return l = sqrt(C_X^2 + C_Y^2) (augmented row norm).
     BLR sensitivity (Proposition 7) uses ||x_i||_2, NOT ||[x_i; y_i]||_2.
  3. No "d_aug" key in public_meta (BLR does not augment [X | y] rows).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rpbench.config import SplitSpec, PreprocessSpec
from rpbench.datasets.base import DatasetAdapter, DatasetBundle


class BreastCancerAdapter(DatasetAdapter):
    """Dataset adapter for Wisconsin Diagnostic Breast Cancer (binary classification).

    Labels: y in {-1, +1} with malignant = +1, benign = -1.

    public_meta() returns l = C_X (max train feature row norm), NOT the
    augmented row norm. This is required for BLR NDIS sensitivity (Prop. 7).
    """

    name = "breast_cancer"

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
        """Load, preprocess, and split the Breast Cancer dataset.

        Preprocessing steps:
          1. Load sklearn breast cancer (569 samples, 30 features).
          2. Train/test split (stratified by label).
          3. StandardScaler on X, fit on train only.
          4. Clip X rows to preprocess_spec.clip_x_bound if clip_x is True.
          5. Map sklearn labels {0, 1} -> {-1, +1} (malignant=1 -> +1, benign=0 -> -1).
          6. preprocess_spec.clip_y is IGNORED (binary labels, no clipping needed).

        Returns
        -------
        DatasetBundle
            meta includes: n_train, n_test, d, C_X, dataset name.
            Note: C_Y is NOT included (binary labels have no meaningful C_Y).
        """
        bunch = load_breast_cancer()
        X_all = bunch.data.astype(np.float64)
        # sklearn: 0=malignant, 1=benign -> map to +1/-1 respectively
        # We want malignant=+1, benign=-1 (positive class = malignant)
        y_raw = bunch.target.astype(np.float64)  # 0=malignant, 1=benign
        y_all = np.where(y_raw == 0, 1.0, -1.0)  # malignant -> +1, benign -> -1

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            train_size=split_spec.train_fraction,
            random_state=split_spec.seed,
            shuffle=True,
            stratify=y_all,
        )

        # Standardize features: fit on train only
        if preprocess_spec.scale_x:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Clip feature rows to C_X bound
        if preprocess_spec.clip_x:
            C_X = float(preprocess_spec.clip_x_bound)
            X_train = self._clip_rows_to_l2_bound(X_train, C_X)
            X_test = self._clip_rows_to_l2_bound(X_test, C_X)
        else:
            C_X = float(np.max(np.linalg.norm(X_train, axis=1)))

        meta = {
            "dataset": self.name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "d": X_train.shape[1],
            "C_X": C_X,
            # NOTE: C_Y intentionally omitted — binary labels, no numeric bound needed.
        }

        return DatasetBundle(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            meta=meta,
        )

    def public_meta(self, bundle: DatasetBundle) -> dict[str, Any]:
        """Return public metadata for the BLR mechanism.

        Returns l = C_X (feature row-norm bound ONLY).

        IMPORTANT: This is different from regression adapters that return
        l = sqrt(C_X^2 + C_Y^2). BLR Proposition 7 uses only the feature
        norm because labels are not model inputs in logistic regression.

        There is NO "d_aug" key — BLR does not augment rows with labels.
        The NDISGaussianWrapper.calibrate() checks for "d_aug" and raises an
        error if it is present, preventing accidental cross-use of adapters.
        """
        C_X = float(bundle.meta["C_X"])
        return {
            "n": bundle.meta["n_train"],
            "d": bundle.meta["d"],
            "l": C_X,         # row-norm bound: l = C_X (features only, NOT augmented)
            "C_X": C_X,
            # Intentionally: no "d_aug", no "C_Y"
        }
