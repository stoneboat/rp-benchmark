"""Mech_RP — NDIS-based random projection DP mechanism.

Ground truth: full-version NDIS paper, Figure 1 (MRP mechanism).
Privacy analysis: Lemma 4 and Proposition 4 (delta-curve as a function of
leverage).  Utility: Theorem 4 (unbiased Gram estimator).

Core privacy-curve functions (_compute_gamma_delta, compute_IS) are copied
from the old NDIS repo (NDIS/src/analysis/RP_privacy_analysis_advanced.py).
The binary search (compute_leverage_upper_bound) is copied from
NDIS/src/RP_mechanisms/optim_RP_DP.py.

MechRPPTR implements the PTR-based wrapper M_RP^PTR from Figure 4 /
Theorem 6 / Proposition 5 (Section 5.3).
"""

from __future__ import annotations

import time
from typing import Any

import mpmath as mp
import numpy as np
from scipy.stats import norm as _norm

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
    if r <= 0:
        raise ValueError("compute_IS requires r > 0")
    if epsilon < 0:
        raise ValueError("compute_IS requires epsilon >= 0")
    if not (0.0 <= leverage < 1.0):
        raise ValueError("compute_IS requires leverage in [0, 1)")
    if leverage == 0.0:
        return 0.0

    s = r / 2.0
    one_minus_p = 1.0 - leverage
    rho = 1.0 / one_minus_p
    log_rho = -np.log1p(-leverage)
    rho_minus_1 = leverage / one_minus_p
    t0 = 2.0 * (epsilon + (r / 2.0) * log_rho) / rho_minus_1
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
    if r <= 0:
        raise ValueError("compute_leverage_upper_bound requires r > 0")
    if epsilon < 0:
        raise ValueError("compute_leverage_upper_bound requires epsilon >= 0")
    if not (0.0 < delta < 1.0):
        raise ValueError("compute_leverage_upper_bound requires delta in (0, 1)")
    if not (0.0 <= low < high < 1.0):
        raise ValueError("compute_leverage_upper_bound requires 0 <= low < high < 1")
    if tol <= 0:
        raise ValueError("compute_leverage_upper_bound requires tol > 0")

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
        if epsilon < 0:
            raise ValueError("Mech_RP calibration requires epsilon >= 0")
        if not (0.0 < delta < 1.0):
            raise ValueError("Mech_RP calibration requires delta in (0, 1)")
        if l <= 0:
            raise ValueError("Mech_RP calibration requires l > 0")

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


class MechRPPois(Mechanism):
    """Poisson-subsampled wrapper for Mech_RP (Figure 3, Section 5.3)."""

    name = "Mech_RP_Pois"

    def __init__(self, r: int, q: float):
        self.r = r
        self.q = q
        self._inner = MechRP(r=r)
        self._calibrated = False
        self._cal: dict[str, Any] = {}

    def calibrate(self, privacy_spec, public_meta: dict) -> None:
        epsilon = privacy_spec.epsilon
        delta = privacy_spec.delta

        if self.r <= 0:
            raise ValueError("Mech_RP_Pois calibration requires r > 0")
        if epsilon < 0:
            raise ValueError("Mech_RP_Pois calibration requires epsilon >= 0")
        if not (0.0 < delta < 1.0):
            raise ValueError("Mech_RP_Pois calibration requires delta in (0, 1)")
        if not (delta < self.q <= 1.0):
            raise ValueError("Mech_RP_Pois calibration requires delta < q <= 1")

        epsilon_0 = float(np.log1p(np.expm1(epsilon) / self.q))
        delta_0 = float(delta / self.q)

        inner_privacy = type(privacy_spec)(
            epsilon=epsilon_0,
            delta=delta_0,
            adjacency=privacy_spec.adjacency,
        )
        self._inner.calibrate(inner_privacy, public_meta)
        self._cal = {
            "epsilon": epsilon,
            "delta": delta,
            "q": self.q,
            "epsilon_0": epsilon_0,
            "delta_0": delta_0,
            "r": self.r,
            "inner_calibration": self._inner.diagnostics(),
        }
        self._calibrated = True

    def release(self, train_data: np.ndarray, seed: int) -> ReleaseBundle:
        assert self._calibrated, "Must call calibrate() first"
        t_start = time.time()

        rng = np.random.RandomState(seed)
        n = train_data.shape[0]
        mask = rng.uniform(size=n) < self.q
        d_sub = train_data[mask]

        inner_seed = int(rng.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        rb = self._inner.release(d_sub, inner_seed)
        rb.mechanism_name = self.name
        rb.calibration = dict(self._cal)
        rb.diagnostics = {
            "q": self.q,
            "subsample_size": int(mask.sum()),
            "subsample_rate_realized": float(mask.mean()) if n > 0 else 0.0,
            "inner_seed": inner_seed,
            **rb.diagnostics,
        }
        rb.runtime_sec = time.time() - t_start
        return rb

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._cal)


# ---------------------------------------------------------------------------
# PTR helpers — support for MechRPPTR (Figure 4, Section 5.3)
# ---------------------------------------------------------------------------

def _gaussian_mech_delta(epsilon_T: float, l: float, tau: float) -> float:
    r"""Gaussian mechanism privacy profile for the eigenvalue test.

    Implements Figure 4, Step 1:
        δ_T(ε_T) = Φ(l²/(2τ) − ε_T·τ/l²) − e^{ε_T}·Φ(−l²/(2τ) − ε_T·τ/l²)

    The test adds N(0, τ²) noise to λ_min(D^T D), which has sensitivity l²
    (removing one row changes λ_min by at most ||v||² ≤ l²).  This is exactly
    the standard Gaussian mechanism privacy profile with sensitivity Δ = l²
    and noise scale σ = τ.
    """
    a = l ** 2 / (2.0 * tau)
    b = epsilon_T * tau / l ** 2
    cdf_pos = _norm.cdf(a - b)
    cdf_neg = _norm.cdf(-a - b)
    # Avoid overflow: when epsilon_T is large, exp(epsilon_T)*cdf_neg can be
    # 0*inf = nan.  Use log-space: second term = exp(epsilon_T + log(cdf_neg)).
    if cdf_neg <= 0.0:
        return float(np.clip(cdf_pos, 0.0, 1.0))
    log_second = epsilon_T + float(np.log(cdf_neg))
    if log_second > 700.0:  # exp would overflow; whole expression ≤ 0 → 0
        return 0.0
    val = cdf_pos - float(np.exp(log_second))
    return float(np.clip(val, 0.0, 1.0))


def _find_epsilon_T(
    delta_T: float,
    l: float,
    tau: float,
    tol: float = 1e-8,
) -> float:
    """Binary search for smallest ε_T ≥ 0 s.t. δ_T(ε_T) ≤ delta_T.

    Implements Figure 4, Step 1 (ε_T calibration).
    """
    if delta_T <= 0.0:
        raise ValueError("_find_epsilon_T requires delta_T > 0")
    if _gaussian_mech_delta(0.0, l, tau) <= delta_T:
        return 0.0
    # Find upper bound where the profile drops below delta_T
    high = 1.0
    while _gaussian_mech_delta(high, l, tau) > delta_T:
        high *= 2.0
    low = 0.0
    while high - low > tol:
        mid = (low + high) / 2.0
        if _gaussian_mech_delta(mid, l, tau) <= delta_T:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# MechRPPTR — PTR-based wrapper (Figure 4, Section 5.3)
# ---------------------------------------------------------------------------

class MechRPPTR(Mechanism):
    """PTR-based wrapper for Mech_RP (Figure 4 / Theorem 6 / Section 5.3).

    Adaptively reduces ridge regularization when λ_min(D^T D) is large,
    improving utility on well-conditioned datasets while remaining (ε,δ)-DP.

    Parameters
    ----------
    r : int
        Sketch dimension.
    tau : float
        Eigenvalue noise scale (τ > 0).  Controls how much Gaussian noise is
        added to λ_min to derive the lower bound λ_lb.
    delta_r : float
        Privacy budget allocated to the RP release step (δ_R).
    delta_t : float
        Privacy budget allocated to the eigenvalue test step (δ_T).
    delta_ptr : float
        Privacy budget for the PTR failure probability (δ_ptr).

    Notes
    -----
    The three delta parameters must sum exactly to the overall δ supplied
    via privacy_spec.delta (validated in calibrate() up to tolerance 1e-10).
    For a delta_rule of "1e-6", a valid split is e.g.
    delta_r=5e-7, delta_t=3e-7, delta_ptr=2e-7.
    """

    name = "Mech_RP_PTR"

    def __init__(
        self,
        r: int,
        tau: float,
        delta_r: float,
        delta_t: float,
        delta_ptr: float,
    ):
        self.r = int(r)
        self.tau = float(tau)
        self.delta_r = float(delta_r)
        self.delta_t = float(delta_t)
        self.delta_ptr = float(delta_ptr)
        self._calibrated = False
        self._cal: dict[str, Any] = {}

    def calibrate(self, privacy_spec, public_meta: dict) -> None:
        epsilon = privacy_spec.epsilon
        delta = privacy_spec.delta
        l = public_meta["l"]
        d_aug = public_meta["d_aug"]
        r = self.r
        tau = self.tau
        delta_r = self.delta_r
        delta_t = self.delta_t
        delta_ptr = self.delta_ptr

        # --- Validate inputs ---
        if r <= 0:
            raise ValueError("MechRPPTR requires r > 0")
        if epsilon < 0:
            raise ValueError("MechRPPTR requires epsilon >= 0")
        if not (0.0 < delta < 1.0):
            raise ValueError("MechRPPTR requires delta in (0, 1)")
        if l <= 0:
            raise ValueError("MechRPPTR requires l > 0")
        if tau <= 0:
            raise ValueError("MechRPPTR requires tau > 0")
        if not (delta_r > 0 and delta_t > 0 and delta_ptr > 0):
            raise ValueError("MechRPPTR requires delta_r, delta_t, delta_ptr > 0")
        if abs(delta_r + delta_t + delta_ptr - delta) > 1e-10:
            raise ValueError(
                f"MechRPPTR requires delta_r + delta_t + delta_ptr == delta; "
                f"got {delta_r + delta_t + delta_ptr} vs {delta}"
            )

        # --- Figure 4, Step 1: find epsilon_T via Gaussian mechanism bound ---
        # Sensitivity of λ_min(D^T D) to one-row removal is ≤ l²; noise = τ.
        epsilon_T = _find_epsilon_T(delta_t, l, tau)
        epsilon_R = max(epsilon - epsilon_T, 0.0)

        # --- Figure 4, Step 2: find p* for the PTR-side RP release ---
        # Reuse existing calibration helper: δ_r^{gRP}(ε_R; p*) ≤ δ_R.
        p_star_ptr = compute_leverage_upper_bound(epsilon_R, delta_r, r)

        # Baseline MRP ridge for Proposition 5 comparison uses the full budget
        # (epsilon, delta), not the PTR-side split (epsilon_R, delta_R).
        p_star_rp = compute_leverage_upper_bound(epsilon, delta, r)
        lambda_rp = l ** 2 / p_star_rp

        # Pre-compute α = τ · Φ^{-1}(1 − δ_ptr)  [used in release()]
        alpha = tau * float(_norm.ppf(1.0 - delta_ptr))

        self._cal = {
            "epsilon": epsilon,
            "delta": delta,
            "epsilon_T": epsilon_T,
            "epsilon_R": epsilon_R,
            "delta_R": delta_r,
            "delta_T": delta_t,
            "delta_ptr": delta_ptr,
            "tau": tau,
            "r": r,
            "d_aug": d_aug,
            "l": l,
            "p_star": p_star_ptr,
            "p_star_ptr": p_star_ptr,
            "p_star_rp": p_star_rp,
            "lambda_rp": lambda_rp,
            "alpha": alpha,
        }
        self._calibrated = True

    def release(self, train_data: np.ndarray, seed: int) -> ReleaseBundle:
        assert self._calibrated, "Must call calibrate() first"
        t_start = time.time()

        rng = np.random.RandomState(seed)
        D = train_data  # (n, d_aug)
        n, d_aug = D.shape

        r = self._cal["r"]
        l = self._cal["l"]
        p_star_ptr = self._cal["p_star_ptr"]
        p_star_rp = self._cal["p_star_rp"]
        lambda_rp = self._cal["lambda_rp"]
        alpha = self._cal["alpha"]
        tau = self._cal["tau"]

        # --- Figure 4, Step 3: private eigenvalue lower bound ---
        # Compute λ_min of the raw Gram matrix D^T D.
        lambda_min_raw = float(np.linalg.eigvalsh(D.T @ D).min())

        # η ~ N(0, τ²) drawn first so RNG state is deterministic before sketch.
        eta = float(rng.normal(0.0, tau))

        # λ_lb = max(λ_min_raw + η − α, 0)
        lambda_lb = max(lambda_min_raw + eta - alpha, 0.0)

        # --- Figure 4, Step 4: PTR ridge ---
        # λ_ptr = max(l²/p*_ptr − λ_lb, 0)
        lambda_ptr = max(l ** 2 / p_star_ptr - lambda_lb, 0.0)
        sigma_ptr = float(np.sqrt(lambda_ptr))

        # Proposition 5 compares λ_ptr against the baseline Mech_RP ridge
        # calibrated under the full (epsilon, delta) budget.
        prop5_threshold = alpha + (l ** 2 / p_star_ptr) - lambda_rp
        prop5_condition_met = bool(lambda_min_raw > prop5_threshold)
        lambda_ptr_lt_lambda_rp = bool(lambda_ptr < lambda_rp)

        # --- Figure 4, Step 5: Gaussian RP on augmented database ---
        # D̄ = [D; √λ_ptr · I_{d_aug}]
        # M̃ = D̄^T G = D^T G_1 + σ_ptr · G_2   (same formula as MechRP)
        G1 = rng.standard_normal((n, r))       # (n, r)
        G2 = rng.standard_normal((d_aug, r))   # (d_aug, r)
        M_tilde = D.T @ G1 + sigma_ptr * G2   # (d_aug, r)

        # Theorem 4 (unbiased Gram estimator), adapted for λ_ptr:
        Sigma_hat = (M_tilde @ M_tilde.T) / r - lambda_ptr * np.eye(d_aug)

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
            diagnostics={
                "epsilon": self._cal["epsilon"],
                "delta": self._cal["delta"],
                "epsilon_T": self._cal["epsilon_T"],
                "epsilon_R": self._cal["epsilon_R"],
                "delta_R": self._cal["delta_R"],
                "delta_T": self._cal["delta_T"],
                "delta_ptr": self._cal["delta_ptr"],
                "tau": tau,
                "p_star": p_star_ptr,
                "p_star_ptr": p_star_ptr,
                "p_star_rp": p_star_rp,
                "lambda_min_raw": lambda_min_raw,
                "alpha": alpha,
                "eta": eta,
                "lambda_lb": lambda_lb,
                "lambda_ptr": lambda_ptr,
                "lambda_rp": lambda_rp,
                "prop5_threshold": prop5_threshold,
                "prop5_condition_met": prop5_condition_met,
                "lambda_ptr_lt_lambda_rp": lambda_ptr_lt_lambda_rp,
            },
            runtime_sec=runtime,
        )

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._cal)
