"""Native ModifiedGaussMix mechanism for augmented linear-regression data.

Paper ground truth:
  - "The Gaussian Mixing Mechanism: Renyi Differential Privacy via Gaussian
    Sketches" (Algorithm 1: ``ModifiedGaussMix``)
  - Algorithm 2, Line 1 for the ``gamma`` calibration step in linear
    regression.

Implementation tie-breaker:
  - Reference GaussMix linear-regression implementation files
    ``utils_linear_mixing.py`` and ``Code_LinearRegression.py``.

This module intentionally ports only the release mechanism into the benchmark's
existing ``Mechanism -> ReleaseBundle`` abstraction. It does not port the
GaussMix repo's experiment loop, dataset loaders, or downstream regression
solver as the primary interface.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import optimize

from rpbench.mechanisms.base import Mechanism, ReleaseBundle


def _modified_gaussmix_objective_full(
    alpha: float,
    r: int,
    sigma_matrix: float,
    delta: float,
    aug_row_bound_sq: float,
) -> float:
    """GaussMix repo calibration objective for Algorithm 2, Line 1.

    This is the native benchmark adaptation of ``objective_func_full`` from the
    GaussMix repo. The paper's Algorithm 2 calibrates the noise level used
    before the release step; the repo solves a quantity named
    ``sigma_matrix`` and uses it directly in the release code.

    In this benchmark, ``public_meta["l"]`` is already the augmented row bound
    for ``A = [X | y]``, so the squared bound here is exactly ``l^2`` with no
    extra ``+ 1`` term.
    """

    if alpha <= 1.0 or alpha >= sigma_matrix / aug_row_bound_sq:
        return np.inf

    term1 = (r * alpha) / (2.0 * (alpha - 1.0)) * np.log(
        1.0 - aug_row_bound_sq / sigma_matrix
    )
    term2 = -(r / (2.0 * (alpha - 1.0))) * np.log(
        1.0 - (alpha * aug_row_bound_sq) / sigma_matrix
    )
    term3 = (
        np.log(3.0 / delta)
        + (alpha - 1.0) * np.log(1.0 - 1.0 / alpha)
        - np.log(alpha)
    ) / (alpha - 1.0)
    term4 = np.sqrt(2.0 * np.log(3.75 / delta)) / (sigma_matrix / np.sqrt(r))
    return float(term1 + term2 + term3 + term4)


def _solve_modified_gaussmix_sigma_matrix(
    epsilon: float,
    delta: float,
    r: int,
    aug_row_bound_sq: float,
) -> float:
    """Solve the GaussMix target quantity used before the release step.

    This follows the repo's ``solve_sigma_renyi_full`` helper, but expressed in
    the benchmark's augmented-row-bound notation ``aug_row_bound_sq = l^2``.
    """

    gaussian_var = 2.0 * aug_row_bound_sq * np.log(1.25 / delta) / (epsilon ** 2)
    left = max(gaussian_var / 500000.0, aug_row_bound_sq + 1e-6)
    right = max(500000.0 * gaussian_var, left * 2.0)
    best_sigma_matrix = right

    while right - left > 1e-6:
        mid_sigma_matrix = (left + right) / 2.0
        result = optimize.minimize_scalar(
            _modified_gaussmix_objective_full,
            bounds=(1.0 + 1e-5, mid_sigma_matrix - 1e-5),
            args=(r, mid_sigma_matrix, delta, aug_row_bound_sq),
            method="bounded",
        )
        if result.success and result.fun < epsilon:
            best_sigma_matrix = mid_sigma_matrix
            right = mid_sigma_matrix
        else:
            left = mid_sigma_matrix

    return float(best_sigma_matrix)


class MechModifiedGaussMix(Mechanism):
    """Algorithm 1 (ModifiedGaussMix) on augmented data ``[X | y]``.

    Parameters
    ----------
    r : int
        Released sketch dimension.

    Notes
    -----
    The mechanism is applied once to the augmented matrix ``A = [X | y]``.
    The released object is the privatized sketch ``A_tilde`` itself, from which
    ``xtx_hat`` and ``xty_hat`` are derived as downstream-consumable sufficient
    statistics.
    """

    name = "Mech_Modified_GaussMix"

    def __init__(self, r: int):
        self.r = int(r)
        self._calibrated = False
        self._cal: dict[str, Any] = {}

    def calibrate(self, privacy_spec, public_meta: dict) -> None:
        epsilon = float(privacy_spec.epsilon)
        delta = float(privacy_spec.delta)
        r = self.r
        l = float(public_meta["l"])
        d_aug = int(public_meta["d_aug"])

        if r <= 0:
            raise ValueError("Mech_Modified_GaussMix calibration requires r > 0")
        if epsilon <= 0.0:
            raise ValueError("Mech_Modified_GaussMix calibration requires epsilon > 0")
        if not (0.0 < delta < 1.0):
            raise ValueError("Mech_Modified_GaussMix calibration requires delta in (0, 1)")
        if l <= 0.0:
            raise ValueError("Mech_Modified_GaussMix calibration requires l > 0")

        aug_row_bound_sq = l ** 2
        tau = float(np.sqrt(2.0 * np.log(3.0 / delta)))
        sigma_matrix = _solve_modified_gaussmix_sigma_matrix(
            epsilon, delta, r, aug_row_bound_sq
        )
        eta = sigma_matrix / np.sqrt(r)

        self._cal = {
            "epsilon": epsilon,
            "delta": delta,
            "r": r,
            "d_aug": d_aug,
            "l": l,
            "aug_row_bound_sq": aug_row_bound_sq,
            "tau": tau,
            "sigma_matrix": sigma_matrix,
            "eta": float(eta),
            "calibration_kind": "modified_gaussmix_full_dp",
        }
        self._calibrated = True

    def release(self, train_data: np.ndarray, seed: int) -> ReleaseBundle:
        assert self._calibrated, "Must call calibrate() first"

        t_start = time.time()
        rng = np.random.RandomState(seed)

        A = train_data
        n, d_aug = A.shape
        d = d_aug - 1

        r = self._cal["r"]
        sigma_matrix = self._cal["sigma_matrix"]
        tau = self._cal["tau"]
        eta = self._cal["eta"]

        # The augmented matrix A = [X | y] is sketched once, matching the
        # paper's joint-data treatment and the repo's shared-sketch release.
        S = rng.standard_normal((r, n))
        Z = rng.standard_normal((r, d_aug))

        lambda_min_aug = float(np.linalg.eigvalsh(A.T @ A).min())
        branch = "simple_noisy_sketch"
        z = None
        private_shift = 0.0
        lambda_tilde = 0.0
        effective_noise_variance = sigma_matrix

        if sigma_matrix > tau:
            branch = "instance_adaptive"
            z = float(rng.standard_normal())

            # Paper notation: Algorithm 2 sets eta = sigma_matrix / sqrt(r).
            # Repo tie-breaker: the private eigenvalue shift is implemented via
            # ``sqrt(sigma_matrix / sqrt(r)) * (tau - z)``. We follow that concrete
            # release rule here rather than re-deriving an alternative scaling.
            private_shift = float(np.sqrt(eta) * (tau - z))
            lambda_tilde = max(lambda_min_aug - private_shift, 0.0)
            effective_noise_variance = max(sigma_matrix - lambda_tilde, 0.0)

        effective_noise_std = float(np.sqrt(effective_noise_variance))
        sketch_matrix = S @ A + effective_noise_std * Z

        x_tilde = sketch_matrix[:, :d]
        y_tilde = sketch_matrix[:, d]
        xtx_hat = x_tilde.T @ x_tilde
        xty_hat = x_tilde.T @ y_tilde

        runtime = time.time() - t_start

        diagnostics = {
            "branch": branch,
            "epsilon": self._cal["epsilon"],
            "delta": self._cal["delta"],
            "r": r,
            "tau": tau,
            "sigma_matrix": sigma_matrix,
            "eta": float(eta),
            "lambda_min_aug": lambda_min_aug,
            "private_eigenvalue_shift": private_shift,
            "lambda_tilde": lambda_tilde,
            "effective_noise_variance": float(effective_noise_variance),
            "effective_noise_std": effective_noise_std,
            "gaussian_test_draw": z,
        }

        return ReleaseBundle(
            mechanism_name=self.name,
            release_kind="sketch",
            xtx_hat=xtx_hat,
            xty_hat=xty_hat,
            sketch_matrix=sketch_matrix,
            calibration=dict(self._cal),
            diagnostics=diagnostics,
            runtime_sec=runtime,
        )

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._cal)
