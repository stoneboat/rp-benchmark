"""Synthetic redundant-cluster regression dataset.

Designed for regimes where Poisson row subsampling preserves Gram geometry while
the downstream response can remain sensitive to ridge-heavy calibration.

Construction (train split uses the same preprocessing contract as other adapters):
  x_{j,t} = z_j + cluster_noise * xi_{j,t},  j=1..m, t=1..k,  n = m*k.
Labels: y = X @ beta + label_noise * N(0,1), where ``beta`` can be aligned to
weak eigendirections of the prototype covariance.

Dataset meta also compares Poisson row thinning vs a Blocki12-style JL sketch
(Gaussian ``M``, ``(MA)^T MA / r`` on augmented ``A = [X|y]``) via relative
Gram operator-norm error; the JL diagnostic omits spectral regularization ``w``
(privacy-specific).
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


def _summarize_counts(counts: np.ndarray) -> dict[str, float]:
    arr = np.asarray(counts, dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "min": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "frac_zero": float("nan"),
        }
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "frac_zero": float(np.mean(arr == 0.0)),
    }


def _inverse_sqrt_psd(matrix: np.ndarray, relative_floor: float = 1e-10) -> tuple[np.ndarray, float]:
    eigvals, eigvecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    eig_max = float(np.max(eigvals)) if eigvals.size else 0.0
    floor = max(relative_floor * max(eig_max, 1.0), 1e-12)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, floor))) @ eigvecs.T
    return inv_sqrt, float(floor)


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
    prototype_ids: np.ndarray | None = None,
    n_trials: int = 30,
) -> dict[str, float]:
    """Diagnostics for Poisson row thinning under row-wise Bernoulli(q)."""
    G = X.T @ X
    norm_g = np.linalg.norm(G, ord=2)
    if norm_g < 1e-15:
        return {
            "relative_opnorm_mean": float("nan"),
            "relative_opnorm_std": float("nan"),
            "relative_opnorm_p50": float("nan"),
            "relative_opnorm_p90": float("nan"),
            "whitened_relative_opnorm_mean": float("nan"),
            "whitened_relative_opnorm_std": float("nan"),
            "whitened_relative_opnorm_p50": float("nan"),
            "whitened_relative_opnorm_p90": float("nan"),
            "n_used": 0.0,
        }

    g_inv_sqrt, whitening_floor = _inverse_sqrt_psd(G)

    n, d = X.shape
    errs: list[float] = []
    whitened_errs: list[float] = []
    retained_count_summaries: list[dict[str, float]] = []
    for _ in range(n_trials):
        mask = rng.random(n) < q
        if int(mask.sum()) < max(1, d // 2):
            continue
        xs = X[mask]
        g_hat = (1.0 / q) * (xs.T @ xs)
        diff = g_hat - G
        err = np.linalg.norm(diff, ord=2) / norm_g
        errs.append(float(err))
        whitened = g_inv_sqrt @ diff @ g_inv_sqrt
        whitened_errs.append(float(np.linalg.norm(whitened, ord=2)))
        if prototype_ids is not None:
            retained = np.bincount(prototype_ids[mask], minlength=int(np.max(prototype_ids)) + 1)
            retained_count_summaries.append(_summarize_counts(retained))

    if not errs:
        return {
            "relative_opnorm_mean": float("nan"),
            "relative_opnorm_std": float("nan"),
            "relative_opnorm_p50": float("nan"),
            "relative_opnorm_p90": float("nan"),
            "whitened_relative_opnorm_mean": float("nan"),
            "whitened_relative_opnorm_std": float("nan"),
            "whitened_relative_opnorm_p50": float("nan"),
            "whitened_relative_opnorm_p90": float("nan"),
            "n_used": 0.0,
            "whitening_floor": whitening_floor,
        }

    arr = np.array(errs)
    whitened_arr = np.array(whitened_errs)
    stats = {
        "relative_opnorm_mean": float(arr.mean()),
        "relative_opnorm_std": float(arr.std()),
        "relative_opnorm_p50": float(np.percentile(arr, 50)),
        "relative_opnorm_p90": float(np.percentile(arr, 90)),
        "whitened_relative_opnorm_mean": float(whitened_arr.mean()),
        "whitened_relative_opnorm_std": float(whitened_arr.std()),
        "whitened_relative_opnorm_p50": float(np.percentile(whitened_arr, 50)),
        "whitened_relative_opnorm_p90": float(np.percentile(whitened_arr, 90)),
        "n_used": float(len(errs)),
        "whitening_floor": whitening_floor,
    }
    if prototype_ids is not None and retained_count_summaries:
        keys = retained_count_summaries[0].keys()
        stats["prototype_retention_per_trial"] = {
            key: float(np.mean([trial[key] for trial in retained_count_summaries])) for key in keys
        }
    return stats


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
        prototype_eigenvalues: list[float] | None = None,
        beta_mode: str = "random",
        beta_strength: float = 1.0,
        beta_midpoint: float = 0.7,
        beta_bandwidth: float = 0.18,
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
        self.prototype_eigenvalues = prototype_eigenvalues
        self.beta_mode = beta_mode
        self.beta_strength = beta_strength
        self.beta_midpoint = beta_midpoint
        self.beta_bandwidth = beta_bandwidth
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

    def _prototype_spectrum(self) -> np.ndarray:
        if self.prototype_eigenvalues is not None:
            spectrum = np.asarray(self.prototype_eigenvalues, dtype=float)
            if spectrum.shape != (self.feature_dim,):
                raise ValueError(
                    "prototype_eigenvalues must have length equal to feature_dim"
                )
            if np.any(spectrum <= 0):
                raise ValueError("prototype_eigenvalues must be strictly positive")
            return spectrum

        return np.array(
            [self.prototype_spectrum_scale * (self.prototype_spectrum_decay**j) for j in range(self.feature_dim)],
            dtype=float,
        )

    def _sample_beta(
        self,
        rng: np.random.Generator,
        eigvecs: np.ndarray,
        eigvals: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        d = eigvals.shape[0]
        if self.beta_mode == "random":
            beta = rng.standard_normal(d)
        else:
            idx = np.linspace(0.0, 1.0, d)
            scaled_eigs = eigvals / max(float(np.max(eigvals)), 1e-12)
            if self.beta_mode == "weak":
                weights = np.power(np.maximum(scaled_eigs, 1e-12), -self.beta_strength)
            elif self.beta_mode == "mid_weak":
                bandwidth = max(self.beta_bandwidth, 1e-6)
                bump = np.exp(-0.5 * ((idx - self.beta_midpoint) / bandwidth) ** 2)
                weights = bump * np.power(np.maximum(scaled_eigs, 1e-12), -self.beta_strength)
            else:
                raise ValueError(
                    "beta_mode must be one of {'random', 'weak', 'mid_weak'}"
                )
            coeffs = rng.standard_normal(d) * weights
            beta = eigvecs @ coeffs

        beta = beta / (np.linalg.norm(beta) + 1e-15)
        coeffs = eigvecs.T @ beta
        weak_start = d // 2
        weak_mass = float(np.sum(coeffs[weak_start:] ** 2))
        mid_start = d // 3
        mid_mass = float(np.sum(coeffs[mid_start:] ** 2))
        return beta, {
            "beta_mode": self.beta_mode,
            "beta_strength": float(self.beta_strength),
            "beta_midpoint": float(self.beta_midpoint),
            "beta_bandwidth": float(self.beta_bandwidth),
            "beta_weak_half_mass": weak_mass,
            "beta_mid_to_weak_mass": mid_mass,
        }

    def _generate_xy(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        m = self.n_prototypes
        k = self.copies_per_prototype
        d = self.feature_dim
        n = m * k

        u, _ = np.linalg.qr(rng.standard_normal((d, d)))
        spectrum = self._prototype_spectrum()
        sigma = u @ np.diag(spectrum) @ u.T
        z = rng.multivariate_normal(np.zeros(d), sigma, size=m)

        x_rows = np.zeros((n, d))
        prototype_ids = np.repeat(np.arange(m, dtype=int), k)
        for j in range(m):
            block = z[j] + self.cluster_noise * rng.standard_normal((k, d))
            x_rows[j * k : (j + 1) * k] = block

        perm = rng.permutation(n)
        x_rows = x_rows[perm]
        prototype_ids = prototype_ids[perm]

        beta, beta_meta = self._sample_beta(rng, u, spectrum)
        y = x_rows @ beta + self.label_noise * rng.standard_normal(n)
        generation_meta: dict[str, Any] = {
            "prototype_eigenvalues": spectrum.tolist(),
            "beta": beta.tolist(),
            **beta_meta,
        }
        return x_rows, y, prototype_ids, generation_meta

    def load(self, split_spec: SplitSpec, preprocess_spec: PreprocessSpec) -> DatasetBundle:
        seed_used = (
            secrets.randbits(63) if self.generation_seed is None else int(self.generation_seed)
        )
        gen = np.random.default_rng(seed_used)
        x_raw, y_raw, prototype_ids_raw, generation_meta = self._generate_xy(gen)

        x_train, x_test, y_train, y_test, prototype_ids_train, prototype_ids_test = train_test_split(
            x_raw,
            y_raw,
            prototype_ids_raw,
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
        train_counts = np.bincount(prototype_ids_train, minlength=self.n_prototypes)
        geom["train_copies_per_prototype_mean"] = float(train_counts.mean())
        geom["train_copies_per_prototype_min"] = float(train_counts.min())
        geom["train_copies_per_prototype_max"] = float(train_counts.max())
        geom["q_times_copies_per_prototype"] = {
            str(q): float(q * self.copies_per_prototype) for q in self.poisson_diag_qs
        }
        geom["q_times_train_copies_per_prototype_mean"] = {
            str(q): float(q * train_counts.mean()) for q in self.poisson_diag_qs
        }

        diag_rng = np.random.default_rng(int(self.poisson_diag_seed))
        poisson_sub: dict[str, Any] = {}
        for q in self.poisson_diag_qs:
            poisson_sub[str(q)] = poisson_gram_relative_opnorm_stats(
                x_train,
                float(q),
                diag_rng,
                prototype_ids=prototype_ids_train,
                n_trials=self.poisson_diag_trials,
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
                "prototype_eigenvalues": generation_meta["prototype_eigenvalues"],
                "beta_mode": generation_meta["beta_mode"],
                "beta_strength": generation_meta["beta_strength"],
                "beta_midpoint": generation_meta["beta_midpoint"],
                "beta_bandwidth": generation_meta["beta_bandwidth"],
                "beta_weak_half_mass": generation_meta["beta_weak_half_mass"],
                "beta_mid_to_weak_mass": generation_meta["beta_mid_to_weak_mass"],
            },
            "geometry_diagnostics": geom,
            "prototype_retention_diagnostics": {
                "train_copy_count_distribution": _summarize_counts(train_counts),
                "expected_retained_copies_per_prototype": {
                    str(q): float(q * np.mean(train_counts)) for q in self.poisson_diag_qs
                },
                "test_copy_count_distribution": _summarize_counts(
                    np.bincount(prototype_ids_test, minlength=self.n_prototypes)
                ),
            },
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
