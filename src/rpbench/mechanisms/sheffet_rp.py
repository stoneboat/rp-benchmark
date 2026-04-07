"""Sheffet-style random projection mechanism for private linear regression.

This is a minimal native benchmark mechanism derived from the Sheffet-style
Gaussian projection baseline used in the GaussMix linear-regression experiments.
It intentionally only implements the release rule inside the existing benchmark
abstraction:

- input: augmented training data ``D = [X | y]``
- output: private projected sufficient statistics in ``ReleaseBundle``

It does not port GaussMix's dataset handling, experiment script structure, or
plotting/evaluation logic.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import optimize

from rpbench.mechanisms.base import Mechanism, ReleaseBundle


def _solve_improved_sheffet_noise_var(
    epsilon: float,
    delta: float,
    r: int,
    row_bound_sq: float,
) -> float:
    """Calibrate the improved Sheffet/GaussMix sigma parameter.

    This is the minimal native adaptation of the ``solve_sigma_renyi`` helper
    used by the GaussMix linear-regression script for the improved "our's
    noise" Sheffet variant.
    """

    sigma_dp = 2.0 * row_bound_sq * np.log(1.25 / delta) / (epsilon ** 2)
    c_term = 1.0 + row_bound_sq

    def objective(alpha: float, sigma: float) -> float:
        if alpha <= 1.0 or alpha >= sigma / c_term:
            return np.inf
        term1 = (r * alpha) / (2.0 * (alpha - 1.0)) * np.log(1.0 - c_term / sigma)
        term2 = -(r / (2.0 * (alpha - 1.0))) * np.log(1.0 - (alpha * c_term) / sigma)
        term3 = (
            np.log(1.0 / delta)
            + (alpha - 1.0) * np.log(1.0 - 1.0 / alpha)
            - np.log(alpha)
        ) / (alpha - 1.0)
        return float(term1 + term2 + term3)

    left = max(sigma_dp / 30000.0, c_term + 1e-6)
    right = max(30000.0 * sigma_dp, left * 2.0)
    best_sigma = right

    while right - left > 1e-6:
        mid_sigma = (left + right) / 2.0
        upper_alpha = mid_sigma / c_term - 1e-5
        if upper_alpha <= 1.0 + 1e-5:
            left = mid_sigma
            continue
        result = optimize.minimize_scalar(
            objective,
            bounds=(1.0 + 1e-5, upper_alpha),
            args=(mid_sigma,),
            method="bounded",
        )
        if result.success and result.fun < (epsilon / 2.0):
            best_sigma = mid_sigma
            right = mid_sigma
        else:
            left = mid_sigma

    return float(best_sigma)


def _sheffet_release_bundle(
    *,
    mechanism_name: str,
    release_kind: str,
    train_data: np.ndarray,
    seed: int,
    r: int,
    noise_var: float,
    laplace_scale: float,
    threshold_core: float,
    threshold_offset: float,
    calibration: dict[str, Any],
) -> ReleaseBundle:
    """Shared Sheffet-style release path used by baseline and improved variants."""

    t_start = time.time()
    rng = np.random.RandomState(seed)

    d_aug = train_data.shape[1]
    d = d_aug - 1
    X = train_data[:, :d]
    y = train_data[:, d]

    lambda_min_aug = float(np.linalg.eigvalsh(train_data.T @ train_data).min())
    threshold_noise = float(rng.laplace(loc=0.0, scale=laplace_scale))
    threshold_rhs = threshold_core + threshold_noise + threshold_offset
    released_clean_sketch = bool(lambda_min_aug > threshold_rhs)

    S = rng.standard_normal((r, X.shape[0]))
    x_proj = S @ X
    y_proj = S @ y

    if released_clean_sketch:
        x_priv = x_proj
        y_priv = y_proj
    else:
        x_priv = x_proj + np.sqrt(noise_var) * rng.standard_normal((r, d))
        y_priv = y_proj + np.sqrt(noise_var) * rng.standard_normal(r)

    xtx_hat = x_priv.T @ x_priv
    xty_hat = x_priv.T @ y_priv
    sketch_matrix = np.vstack([x_priv.T, y_priv[np.newaxis, :]])

    runtime = time.time() - t_start

    return ReleaseBundle(
        mechanism_name=mechanism_name,
        release_kind=release_kind,
        xtx_hat=xtx_hat,
        xty_hat=xty_hat,
        sketch_matrix=sketch_matrix,
        calibration=dict(calibration),
        diagnostics={
            "lambda_min_aug": lambda_min_aug,
            "threshold_noise": threshold_noise,
            "threshold_core": float(threshold_core),
            "threshold_rhs": threshold_rhs,
            "released_clean_sketch": released_clean_sketch,
            "noise_var": float(noise_var),
            "laplace_scale": float(laplace_scale),
        },
        runtime_sec=runtime,
    )


class MechSheffetRP(Mechanism):
    """Sheffet-style Gaussian RP release.

    Parameters
    ----------
    r : int
        Projection dimension.

    Notes
    -----
    The mechanism uses the benchmark's public row bound
    ``l = sqrt(C_X^2 + C_Y^2)`` and release privacy parameters ``(epsilon,delta)``.
    It privately checks whether the augmented Gram matrix is sufficiently
    well-conditioned to release an unnoised Gaussian sketch; otherwise it adds
    Gaussian noise to the projected design/response pair.
    """

    name = "Mech_Sheffet_RP"

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
            raise ValueError("Mech_Sheffet_RP calibration requires r > 0")
        if epsilon <= 0.0:
            raise ValueError("Mech_Sheffet_RP calibration requires epsilon > 0")
        if not (0.0 < delta < 1.0):
            raise ValueError("Mech_Sheffet_RP calibration requires delta in (0, 1)")
        if l <= 0.0:
            raise ValueError("Mech_Sheffet_RP calibration requires l > 0")

        row_bound_sq = l ** 2
        log_8_over_delta = float(np.log(8.0 / delta))
        log_1_over_delta = float(np.log(1.0 / delta))

        # Sheffet-style noise/threshold calibration, expressed in the benchmark's
        # public row-bound notation.
        noise_var = (
            4.0
            * row_bound_sq
            * (np.sqrt(2.0 * r * log_8_over_delta) + log_8_over_delta)
            / epsilon
        )
        laplace_scale = 4.0 * row_bound_sq / epsilon
        threshold_offset = 4.0 * row_bound_sq * log_1_over_delta / epsilon

        self._cal = {
            "epsilon": epsilon,
            "delta": delta,
            "r": r,
            "d_aug": d_aug,
            "l": l,
            "row_bound_sq": row_bound_sq,
            "noise_var": float(noise_var),
            "laplace_scale": float(laplace_scale),
            "threshold_offset": float(threshold_offset),
        }
        self._calibrated = True

    def release(self, train_data: np.ndarray, seed: int) -> ReleaseBundle:
        assert self._calibrated, "Must call calibrate() first"
        return _sheffet_release_bundle(
            mechanism_name=self.name,
            release_kind="sketch",
            train_data=train_data,
            seed=seed,
            r=self._cal["r"],
            noise_var=self._cal["noise_var"],
            laplace_scale=self._cal["laplace_scale"],
            threshold_core=self._cal["noise_var"],
            threshold_offset=self._cal["threshold_offset"],
            calibration=self._cal,
        )

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._cal)


class MechImprovedSheffetRP(Mechanism):
    """Improved Sheffet-style RP release using GaussMix's sigma calibration."""

    name = "Mech_Improved_Sheffet_RP"

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
            raise ValueError("Mech_Improved_Sheffet_RP calibration requires r > 0")
        if epsilon <= 0.0:
            raise ValueError("Mech_Improved_Sheffet_RP calibration requires epsilon > 0")
        if not (0.0 < delta < 1.0):
            raise ValueError("Mech_Improved_Sheffet_RP calibration requires delta in (0, 1)")
        if l <= 0.0:
            raise ValueError("Mech_Improved_Sheffet_RP calibration requires l > 0")

        row_bound_sq = l ** 2
        noise_var = _solve_improved_sheffet_noise_var(epsilon, delta, r, row_bound_sq)
        laplace_scale = 4.0 * row_bound_sq / epsilon
        threshold_offset = 4.0 * row_bound_sq * np.log(1.0 / delta) / epsilon

        self._cal = {
            "epsilon": epsilon,
            "delta": delta,
            "r": r,
            "d_aug": d_aug,
            "l": l,
            "row_bound_sq": row_bound_sq,
            "noise_var": float(noise_var),
            "laplace_scale": float(laplace_scale),
            "threshold_offset": float(threshold_offset),
            "threshold_core": float(noise_var / 2.0),
            "calibration_kind": "gaussmix_improved_sheffet",
        }
        self._calibrated = True

    def release(self, train_data: np.ndarray, seed: int) -> ReleaseBundle:
        assert self._calibrated, "Must call calibrate() first"
        return _sheffet_release_bundle(
            mechanism_name=self.name,
            release_kind="sketch",
            train_data=train_data,
            seed=seed,
            r=self._cal["r"],
            noise_var=self._cal["noise_var"],
            laplace_scale=self._cal["laplace_scale"],
            threshold_core=self._cal["threshold_core"],
            threshold_offset=self._cal["threshold_offset"],
            calibration=self._cal,
        )

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._cal)
