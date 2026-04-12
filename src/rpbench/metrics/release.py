"""Release-level metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rpbench.mechanisms.base import ReleaseBundle


def relative_frobenius_xtx(xtx_true: np.ndarray, xtx_hat: np.ndarray) -> float:
    """Relative Frobenius error: ||X^T X - xtx_hat||_F / ||X^T X||_F."""
    num = np.linalg.norm(xtx_true - xtx_hat, "fro")
    denom = np.linalg.norm(xtx_true, "fro")
    if denom < 1e-15:
        return float("inf")
    return float(num / denom)


def normalized_debiased_xtx(release_bundle: "ReleaseBundle") -> np.ndarray:
    """Return an X^T X diagnostic on the original covariance scale.

    This does not mutate the release or affect downstream OLS. It only maps raw
    Gaussian-sketch sufficient statistics to the scale used by the covariance
    diagnostic.
    """
    xtx_hat = release_bundle.xtx_hat
    if xtx_hat is None:
        raise ValueError("ReleaseBundle.xtx_hat is required for covariance metrics")

    mech = release_bundle.mechanism_name
    cal = release_bundle.calibration or {}
    diag = release_bundle.diagnostics or {}

    if mech in {"Mech_Sheffet_RP", "Mech_Improved_Sheffet_RP"}:
        r = int(cal.get("r", diag.get("r", 0)))
        if r <= 0:
            raise ValueError(f"{mech} requires positive r for normalized covariance metric")
        xtx = xtx_hat / float(r)
        if not bool(diag.get("released_clean_sketch", False)):
            noise_var = float(diag.get("noise_var", cal.get("noise_var", 0.0)))
            xtx = xtx - noise_var * np.eye(xtx.shape[0])
        return xtx

    if mech == "Mech_Modified_GaussMix":
        r = int(cal.get("r", diag.get("r", 0)))
        if r <= 0:
            raise ValueError(f"{mech} requires positive r for normalized covariance metric")
        noise_var = float(
            diag.get("effective_noise_variance", cal.get("sigma_matrix", 0.0))
        )
        return xtx_hat / float(r) - noise_var * np.eye(xtx_hat.shape[0])

    return xtx_hat


def relative_frobenius_xtx_normalized(
    xtx_true: np.ndarray,
    release_bundle: "ReleaseBundle",
) -> float:
    """Relative Frobenius error after diagnostic-only covariance normalization."""
    return relative_frobenius_xtx(xtx_true, normalized_debiased_xtx(release_bundle))
