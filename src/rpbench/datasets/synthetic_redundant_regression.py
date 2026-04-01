"""Synthetic redundant-cluster regression dataset.

Designed for regimes where Poisson row subsampling preserves Gram geometry:
tall n, redundant replicates of prototypes, mild spectrum, small within-cluster noise.

Construction (train split uses the same preprocessing contract as other adapters):
  x_{j,t} = z_j + cluster_noise * xi_{j,t},  j=1..m, t=1..k,  n = m*k.
Labels: y = X @ beta + label_noise * N(0,1).

Dataset meta also compares Poisson row thinning vs a Blocki12-style JL sketch (Gaussian
``M``, ``(MA)^T MA / r`` on augmented ``A = [X|y]``) via relative Gram operator-norm error;
the JL diagnostic omits spectral regularization ``w`` (privacy-specific).
"""

from __future__ import annotations

import secrets
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rpbench.config import PreprocessSpec, SplitSpec
from rpbench.datasets.base import DatasetAdapter, DatasetBundle
from rpbench.utils.linear_algebra import augmented_data


def gram_spectrum_stats(X: np.ndarray) -> dict[str, float]:
    """Spectral stats of X^T X (symmetric PSD)."""
    xtx = X.T @ X
    w = np.linalg.eigvalsh(xtx)
    w_min = float(w[0])
    w_max = float(w[-1])
    cond = float(w_max / w_min) if w_min > 1e-15 else float("inf")
    return {
        "xtx_eig_min": w_min,
        "xtx_eig_max": w_max,
        "xtx_condition_number": cond,
    }


def max_row_leverage(X: np.ndarray) -> float:
    """Max diagonal leverage h_i = x_i^T (X^T X)^{-1} x_i."""
    d = X.shape[1]
    if X.shape[0] <= d:
        return float("nan")
    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    z = X @ xtx_inv
    h = np.einsum("ij,ij->i", z, X)
    return float(np.max(h))


def poisson_gram_relative_opnorm_stats(
    X: np.ndarray,
    q: float,
    rng: np.random.Generator,
    n_trials: int = 30,
) -> dict[str, float]:
    """Mean/std/quantiles of || q^{-1} X_S^T X_S - X^T X ||_2 / ||X^T X||_2 under row-wise Bernoulli(q)."""
    G = X.T @ X
    norm_g = np.linalg.norm(G, ord=2)
    if norm_g < 1e-15:
        return {"mean": float("nan"), "std": float("nan"), "p50": float("nan"), "p90": float("nan"), "n_used": 0.0}

    n, d = X.shape
    errs: list[float] = []
    for _ in range(n_trials):
        mask = rng.random(n) < q
        if int(mask.sum()) < max(1, d // 2):
            continue
        xs = X[mask]
        g_hat = (1.0 / q) * (xs.T @ xs)
        err = np.linalg.norm(g_hat - G, ord=2) / norm_g
        errs.append(float(err))

    if not errs:
        return {"mean": float("nan"), "std": float("nan"), "p50": float("nan"), "p90": float("nan"), "n_used": 0.0}

    arr = np.array(errs)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "n_used": float(len(errs)),
    }


def blocki12_jl_gram_relative_opnorm_stats(
    A: np.ndarray,
    r: int,
    rng: np.random.Generator,
    n_trials: int = 30,
) -> dict[str, float]:
    """Relative operator-norm Gram error for the JL sketch in Blocki et al. (2012), Alg. 3.

    Uses augmented ``A`` (same layout as ``Blocki12_JL.release``): ``M`` iid ``N(0,1)``
    with shape ``(r, n)``, ``MA = M @ A``, ``G_hat = (MA)^T MA / r``, compared to ``G = A^T A``. Omits the
    spectral regularization step (``w``), so this isolates JL projection variance.
    """
    G = A.T @ A
    norm_g = np.linalg.norm(G, ord=2)
    if norm_g < 1e-15:
        return {"mean": float("nan"), "std": float("nan"), "p50": float("nan"), "p90": float("nan"), "n_used": 0.0}

    n = A.shape[0]
    if r <= 0 or n < 1:
        return {"mean": float("nan"), "std": float("nan"), "p50": float("nan"), "p90": float("nan"), "n_used": 0.0}

    errs: list[float] = []
    for _ in range(n_trials):
        M = rng.standard_normal((r, n))
        MA = M @ A
        g_hat = (MA.T @ MA) / r
        err = np.linalg.norm(g_hat - G, ord=2) / norm_g
        errs.append(float(err))

    arr = np.array(errs)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "n_used": float(len(errs)),
    }


class SyntheticRedundantRegressionAdapter(DatasetAdapter):
    """Redundant-cluster synthetic regression; registry key: ``synthetic_redundant_regression``.

    If ``generation_seed`` is omitted (``None``), a cryptographically drawn integer seed is used;
    ``meta["synthetic_params"]["generation_seed"]`` records the seed actually used.
    """

    name = "synthetic_redundant_regression"

    def __init__(
        self,
        feature_dim: int = 10,
        n_prototypes: int = 100,
        copies_per_prototype: int = 30,
        cluster_noise: float = 0.1,
        label_noise: float = 0.1,
        generation_seed: int | None = None,
        prototype_spectrum_scale: float = 1.0,
        prototype_spectrum_decay: float = 0.92,
        poisson_diag_qs: list[float] | None = None,
        poisson_diag_trials: int = 30,
        poisson_diag_seed: int = 999,
        blocki12_jl_diag_rs: list[int] | None = None,
        blocki12_jl_diag_trials: int = 30,
        blocki12_jl_diag_seed: int = 10001,
    ) -> None:
        self.feature_dim = feature_dim
        self.n_prototypes = n_prototypes
        self.copies_per_prototype = copies_per_prototype
        self.cluster_noise = cluster_noise
        self.label_noise = label_noise
        self.generation_seed = generation_seed
        self.prototype_spectrum_scale = prototype_spectrum_scale
        self.prototype_spectrum_decay = prototype_spectrum_decay
        self.poisson_diag_qs = poisson_diag_qs if poisson_diag_qs is not None else [0.2, 0.8]
        self.poisson_diag_trials = poisson_diag_trials
        self.poisson_diag_seed = poisson_diag_seed
        self.blocki12_jl_diag_rs = blocki12_jl_diag_rs if blocki12_jl_diag_rs is not None else [24, 50]
        self.blocki12_jl_diag_trials = blocki12_jl_diag_trials
        self.blocki12_jl_diag_seed = blocki12_jl_diag_seed

    @staticmethod
    def _clip_rows_to_l2_bound(X: np.ndarray, bound: float) -> np.ndarray:
        if bound <= 0:
            raise ValueError("clip_x_bound must be > 0 when clip_x is enabled")
        norms = np.linalg.norm(X, axis=1)
        scales = np.ones_like(norms)
        mask = norms > bound
        scales[mask] = bound / norms[mask]
        return X * scales[:, np.newaxis]

    def _generate_xy(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        m = self.n_prototypes
        k = self.copies_per_prototype
        d = self.feature_dim
        n = m * k

        u, _ = np.linalg.qr(rng.standard_normal((d, d)))
        spectrum = np.array(
            [self.prototype_spectrum_scale * (self.prototype_spectrum_decay**j) for j in range(d)]
        )
        sigma = u @ np.diag(spectrum) @ u.T
        z = rng.multivariate_normal(np.zeros(d), sigma, size=m)

        x_rows = np.zeros((n, d))
        for j in range(m):
            block = z[j] + self.cluster_noise * rng.standard_normal((k, d))
            x_rows[j * k : (j + 1) * k] = block

        perm = rng.permutation(n)
        x_rows = x_rows[perm]

        beta = rng.standard_normal(d)
        beta = beta / (np.linalg.norm(beta) + 1e-15)
        y = x_rows @ beta + self.label_noise * rng.standard_normal(n)
        return x_rows, y

    def load(self, split_spec: SplitSpec, preprocess_spec: PreprocessSpec) -> DatasetBundle:
        seed_used = (
            secrets.randbits(63) if self.generation_seed is None else int(self.generation_seed)
        )
        gen = np.random.default_rng(seed_used)
        x_raw, y_raw = self._generate_xy(gen)

        x_train, x_test, y_train, y_test = train_test_split(
            x_raw,
            y_raw,
            train_size=split_spec.train_fraction,
            random_state=split_spec.seed,
            shuffle=True,
        )

        if preprocess_spec.scale_x:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

        y_mean = y_train.mean()
        y_std = y_train.std()
        if y_std > 0:
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

        if preprocess_spec.clip_x:
            c_x = float(preprocess_spec.clip_x_bound)
            x_train = self._clip_rows_to_l2_bound(x_train, c_x)
            x_test = self._clip_rows_to_l2_bound(x_test, c_x)
        else:
            c_x = float(np.max(np.linalg.norm(x_train, axis=1)))

        if preprocess_spec.clip_y:
            c_y = float(preprocess_spec.clip_y_bound)
            if c_y <= 0:
                raise ValueError("clip_y_bound must be > 0 when clip_y is enabled")
            y_train = np.clip(y_train, -c_y, c_y)
            y_test = np.clip(y_test, -c_y, c_y)
        else:
            c_y = float(np.max(np.abs(y_train)))

        geom = gram_spectrum_stats(x_train)
        geom["max_leverage"] = max_row_leverage(x_train)

        diag_rng = np.random.default_rng(int(self.poisson_diag_seed))
        poisson_sub: dict[str, Any] = {}
        for q in self.poisson_diag_qs:
            poisson_sub[str(q)] = poisson_gram_relative_opnorm_stats(
                x_train, float(q), diag_rng, n_trials=self.poisson_diag_trials
            )

        A_train = augmented_data(x_train, y_train)
        jl_rng = np.random.default_rng(int(self.blocki12_jl_diag_seed))
        blocki12_jl_sub: dict[str, Any] = {}
        for r in self.blocki12_jl_diag_rs:
            blocki12_jl_sub[str(int(r))] = blocki12_jl_gram_relative_opnorm_stats(
                A_train, int(r), jl_rng, n_trials=self.blocki12_jl_diag_trials
            )

        meta: dict[str, Any] = {
            "dataset": self.name,
            "n_train": len(x_train),
            "n_test": len(x_test),
            "d": x_train.shape[1],
            "C_X": float(c_x),
            "C_Y": float(c_y),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "synthetic_params": {
                "feature_dim": self.feature_dim,
                "n_prototypes": self.n_prototypes,
                "copies_per_prototype": self.copies_per_prototype,
                "cluster_noise": self.cluster_noise,
                "label_noise": self.label_noise,
                "generation_seed": seed_used,
                "generation_seed_from_config": self.generation_seed,
                "prototype_spectrum_scale": self.prototype_spectrum_scale,
                "prototype_spectrum_decay": self.prototype_spectrum_decay,
            },
            "geometry_diagnostics": geom,
            "poisson_subsampling_diagnostics": poisson_sub,
            "blocki12_jl_diagnostics": blocki12_jl_sub,
        }

        return DatasetBundle(
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
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
