"""Mech_RP — NDIS-based random projection DP mechanism.

Ground truth: NDIS paper (arxiv 2309.01243), Figure 1 (MRP mechanism).
Privacy analysis: Lemma 4 and Proposition 4 (delta-curve as a function of
leverage).  Utility: Theorem 4 (unbiased Gram estimator).

Core privacy-curve functions (_compute_gamma_delta, compute_IS) are copied
from the old NDIS repo (NDIS/src/analysis/RP_privacy_analysis_advanced.py).
The binary search (compute_leverage_upper_bound) is copied from
NDIS/src/RP_mechanisms/optim_RP_DP.py.
"""

from __future__ import annotations

import time
from typing import Any

import mpmath as mp
import numpy as np

from rpbench.mechanisms.base import Mechanism, ReleaseBundle


# ---------------------------------------------------------------------------
# Privacy curve — copied from NDIS/src/analysis/RP_privacy_analysis_advanced.py
# ---------------------------------------------------------------------------

def _compute_gamma_delta(s, t0, rho, eps, dps=80):
    r"""Compute the RP delta-curve value.

    Implements Lemma 4 / Proposition 4 of the NDIS paper:

        delta = (Gamma(s, t0/2) - exp(eps) * Gamma(s, rho*t0/2)) / Gamma(s)

    where Gamma(s, x) is the upper incomplete gamma function.

    Uses mpmath for high-precision evaluation (``dps`` decimal places).
    """
    mp.mp.dps = dps

    s = mp.mpf(s)
    t0 = mp.mpf(t0)
    rho = mp.mpf(rho)
    eps = mp.mpf(eps)

    x1 = t0 / 2
    x2 = rho * t0 / 2

    G1 = mp.gammainc(s, x1, mp.inf)   # upper incomplete gamma Gamma(s, x1)
    G2 = mp.gammainc(s, x2, mp.inf)   # upper incomplete gamma Gamma(s, x2)

    delta = (G1 - mp.exp(eps) * G2) / mp.gamma(s)

    if delta < 0:
        delta = mp.mpf("0")
    elif delta > 1:
        delta = mp.mpf("1")

    return float(delta)


def compute_IS(epsilon: float, leverage: float, r: int) -> float:
    r"""Evaluate delta^{gRP}_r(epsilon; p) from Proposition 4.

    Parameters
    ----------
    epsilon : float
        Privacy parameter.
    leverage : float
        Leverage score p in (0, 1).
    r : int
        Sketch dimension (number of random projections).
    """
    s = r / 2.0
    rho = 1.0 / (1.0 - leverage)
    t0 = 2.0 * (epsilon + (r / 2.0) * np.log(rho)) / (rho - 1.0)
    return _compute_gamma_delta(s, t0, rho, epsilon)


# ---------------------------------------------------------------------------
# Calibration helpers — adapted from NDIS/src/RP_mechanisms/optim_RP_DP.py
# ---------------------------------------------------------------------------

def compute_leverage_upper_bound(
    epsilon: float,
    delta: float,
    r: int,
    low: float = 1e-10,
    high: float = 1 - 1e-8,
    tol: float = 1e-8,
) -> float:
    """Binary search for p* such that delta^{gRP}_r(eps; p*) = delta.

    Implements Figure 1, Step 1 of the NDIS paper.
    """
    if compute_IS(epsilon, low, r) > delta:
        raise ValueError(
            f"delta^{{gRP}}(eps={epsilon}, p={low}, r={r}) > delta={delta}; "
            "lower bound too high"
        )
    if compute_IS(epsilon, high, r) < delta:
        raise ValueError(
            f"delta^{{gRP}}(eps={epsilon}, p={high}, r={r}) < delta={delta}; "
            "upper bound too low"
        )

    while high - low > tol:
        mid = (low + high) / 2.0
        if compute_IS(epsilon, mid, r) > delta:
            high = mid
        else:
            low = mid

    return (low + high) / 2.0


def compute_largest_l2(D: np.ndarray) -> float:
    """Return max row L2 norm of D."""
    return float(np.max(np.linalg.norm(D, axis=1)))


# ---------------------------------------------------------------------------
# MechRP — the benchmark mechanism class
# ---------------------------------------------------------------------------

class MechRP(Mechanism):
    """NDIS-based Random Projection mechanism (MRP, Figure 1 of the paper).

    Operates on augmented data D = [X | y] of shape (n, d_aug).
    """

    name = "Mech_RP"

    def __init__(self, r: int):
        self.r = r
        self._calibrated = False
        self._cal: dict[str, Any] = {}

    def calibrate(self, privacy_spec, public_meta: dict) -> None:
        epsilon = privacy_spec.epsilon
        delta = privacy_spec.delta
        l = public_meta["l"]
        d_aug = public_meta["d_aug"]
        r = self.r

        if r <= 0:
            raise ValueError("Mech_RP calibration requires r > 0")

        p_star = compute_leverage_upper_bound(epsilon, delta, r)
        lambda_ridge = l ** 2 / p_star

        self._cal = {
            "epsilon": epsilon,
            "delta": delta,
            "r": r,
            "d_aug": d_aug,
            "l": l,
            "p_star": p_star,
            "lambda_ridge": lambda_ridge,
            "sigma": np.sqrt(lambda_ridge),
        }
        self._calibrated = True

    def release(self, train_data: np.ndarray, seed: int) -> ReleaseBundle:
        assert self._calibrated, "Must call calibrate() first"
        t_start = time.time()

        rng = np.random.RandomState(seed)
        D = train_data  # (n, d_aug)
        n, d_aug = D.shape
        r = self._cal["r"]
        sigma = self._cal["sigma"]
        lambda_ridge = self._cal["lambda_ridge"]

        # Figure 1, Step 2-3: augment and project
        # D_bar = [D; sigma * I_{d_aug}]  -> shape (n + d_aug, d_aug)
        # M_tilde = D_bar^T G  = D^T G_1 + sigma * G_2
        G1 = rng.standard_normal((n, r))       # (n, r)
        G2 = rng.standard_normal((d_aug, r))   # (d_aug, r)
        M_tilde = D.T @ G1 + sigma * G2        # (d_aug, r)

        # Theorem 4: unbiased Gram estimator
        Sigma_hat = (M_tilde @ M_tilde.T) / r - lambda_ridge * np.eye(d_aug)

        d = d_aug - 1
        xtx_hat = Sigma_hat[:d, :d]
        xty_hat = Sigma_hat[:d, d]

        runtime = time.time() - t_start

        return ReleaseBundle(
            mechanism_name=self.name,
            release_kind="gram_blocks",
            xtx_hat=xtx_hat,
            xty_hat=xty_hat,
            sketch_matrix=M_tilde,
            calibration=dict(self._cal),
            diagnostics={"sigma": sigma, "lambda_ridge": lambda_ridge},
            runtime_sec=runtime,
        )

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._cal)
