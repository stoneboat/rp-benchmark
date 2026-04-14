"""NDIS privacy calibration: delta upper bounds and tau* binary search.

Paper references (updated Definition 6):
  delta_bar_mean  — Proposition 1 / Definition 6 (mean-shift upper bound)
  delta_bar_cov   — Lemma 2 / Definition 6 (covariance-shift upper bound,
                    tighter form using envelope functionals A_* and B_*)
  delta_bar_LC    — Definition 6 (Loewner-comparable combined upper bound)
  find_tau_star   — Figure 5, Step 1 (binary search for calibrated tau*)

Key math (updated Definition 6, from Lemma 2 proof):
  c(rho_inf) = exp(rho_inf) - 1
  k          = 1 / (2 * c(rho_inf))

  phi_s(ell) = s*ell - 0.5 * log(1 + 2s*(1 - exp(-ell)))
  psi_u(ell) = -u*ell - 0.5 * log(1 - 2u*(exp(ell) - 1))

  A_*(s; rho_inf, nu, d) = sup{ sum_i phi_s(ell_i) : 0<=ell_i<=rho_inf, sum<=nu }
  B_*(u; rho_inf, nu, d) = sup{ sum_i psi_u(ell_i) : 0<=ell_i<=rho_inf, sum<=nu }

  Both phi_s and psi_u are convex and non-decreasing in ell >= 0, so the sup
  is achieved at the concentrated vertex:
    k* = min(d, floor(nu / rho_inf)) coordinates at rho_inf
    r  = nu - k* * rho_inf           one coordinate at r (if r > 0)

  delta_bar_cov(eps; rho_inf, nu, d):
    = 0                                              if eps >= nu/2
    = inf_{s>=0, u in (0,k)} [
          exp(-2*eps*s + A_*(s))
        - exp(eps) * (1 - exp(2*eps*u + B_*(u)))
      ]                                              if eps < nu/2

  delta_bar_LC(eps; Delta, rho_inf, nu, d):
    = inf_{eps1 + eps2 = eps, eps1,eps2 >= 0} [
          delta_bar_mean(eps1; Delta)
        + exp(eps1) * delta_bar_cov(eps2; rho_inf, nu, d)
      ]
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

if TYPE_CHECKING:
    from ndis_gaussian.interfaces import GaussianOutputAlgorithm


# ---------------------------------------------------------------------------
# Mean-shift upper bound
# ---------------------------------------------------------------------------

def delta_bar_mean(epsilon: float, Delta: float) -> float:
    """IS upper bound for the mean-shift contribution (Proposition 1 / Def. 6).

    delta_bar_mean(eps; Delta) =
        Phi(Delta/2 - eps/Delta) - e^eps * Phi(-Delta/2 - eps/Delta)  Delta > 0
        0                                                              Delta = 0

    where Phi is the standard normal CDF.
    """
    if Delta <= 0.0:
        return 0.0
    a = Delta / 2.0 - epsilon / Delta
    b = -Delta / 2.0 - epsilon / Delta
    return float(norm.cdf(a) - math.exp(epsilon) * norm.cdf(b))


# ---------------------------------------------------------------------------
# Internal helpers for the envelope functionals A_* and B_*
# ---------------------------------------------------------------------------

def _phi_s(s: float, ell: float) -> float:
    """Per-dimension contribution to A_*: phi_s(ell) = s*ell - 0.5*log(1+2s*(1-exp(-ell))).

    Derived from E[exp(-s*w_i*chi^2_1)] = (1+2s*w_i)^{-1/2} where w_i = 1 - exp(-ell_i).
    phi_s is convex and non-decreasing in ell >= 0.
    """
    if ell <= 0.0 or s <= 0.0:
        return 0.0
    return s * ell - 0.5 * math.log1p(2.0 * s * (1.0 - math.exp(-ell)))


def _psi_u(u: float, ell: float) -> float:
    """Per-dimension contribution to B_*: psi_u(ell) = -u*ell - 0.5*log(1-2u*(exp(ell)-1)).

    Derived from E[exp(u*w_i'*chi^2_1)] = (1-2u*w_i')^{-1/2} where w_i' = exp(ell_i)-1.
    Requires u < 1/(2*(exp(ell)-1)); the caller is responsible for u < k.
    psi_u is convex and non-decreasing in ell >= 0 for u in (0, k).
    """
    if ell <= 0.0 or u <= 0.0:
        return 0.0
    inner = 1.0 - 2.0 * u * (math.exp(ell) - 1.0)
    if inner <= 0.0:
        return math.inf  # outside the valid domain; caller should not reach here
    return -u * ell - 0.5 * math.log(inner)


def _A_star(s: float, rho_inf: float, nu: float, d: int) -> float:
    """Envelope functional A_*(s; rho_inf, nu, d) = sup sum_i phi_s(ell_i).

    Since phi_s is convex and non-decreasing, the sup is at the concentrated vertex:
      k* = min(d, floor(nu/rho_inf)) coordinates at rho_inf
      r  = nu - k* * rho_inf         one coordinate at r (if r > tol)
    """
    if rho_inf <= 0.0 or nu <= 0.0 or s <= 0.0:
        return 0.0
    # number of dimensions saturated at rho_inf
    k_star = min(d, int(nu / rho_inf + 1e-9))
    r = nu - k_star * rho_inf
    result = k_star * _phi_s(s, rho_inf)
    if r > 1e-12:
        result += _phi_s(s, r)
    return result


def _B_star(u: float, rho_inf: float, nu: float, d: int) -> float:
    """Envelope functional B_*(u; rho_inf, nu, d) = sup sum_i psi_u(ell_i).

    Same vertex structure as A_*.
    """
    if rho_inf <= 0.0 or nu <= 0.0 or u <= 0.0:
        return 0.0
    k_star = min(d, int(nu / rho_inf + 1e-9))
    r = nu - k_star * rho_inf
    result = k_star * _psi_u(u, rho_inf)
    if r > 1e-12:
        result += _psi_u(u, r)
    return result


# ---------------------------------------------------------------------------
# Covariance-shift upper bound (updated Definition 6)
# ---------------------------------------------------------------------------

_EXP_CAP = 700.0  # avoid math range error in math.exp


def _cov_objective(s: float, u: float, epsilon: float,
                   rho_inf: float, nu: float, d: int) -> float:
    """The 2D objective for delta_bar_cov at a specific (s, u).

    Returns math.inf when the exponent of the first term overflows — such
    points are never the infimum (the grid search/Nelder-Mead will ignore them).
    """
    A = _A_star(s, rho_inf, nu, d)
    B = _B_star(u, rho_inf, nu, d)
    arg1 = -2.0 * epsilon * s + A
    if arg1 > _EXP_CAP:
        return math.inf   # too large; not the infimum
    first = math.exp(arg1)
    arg2 = 2.0 * epsilon * u + B
    if arg2 > _EXP_CAP:
        return math.inf   # second_exp overflows; conservatively treat as inf
    second_exp = math.exp(arg2)
    return first - math.exp(epsilon) * (1.0 - second_exp)


def delta_bar_cov(
    epsilon: float,
    rho_inf: float,
    nu: float,
    d: int,
) -> float:
    """IS upper bound for the covariance-shift contribution (updated Definition 6).

    Uses the tighter envelope functionals A_* and B_* derived from Lemma 2:

      delta_bar_cov(eps; rho_inf, nu, d) =
        0,                                                     eps >= nu/2
        inf_{s>=0, u in (0,k)} [
            exp(-2*eps*s + A_*(s; rho_inf, nu, d))
          - exp(eps) * (1 - exp(2*eps*u + B_*(u; rho_inf, nu, d)))
        ],                                                     eps < nu/2

    where:
      phi_s(ell) = s*ell - 0.5*log(1+2s*(1-exp(-ell)))
      psi_u(ell) = -u*ell - 0.5*log(1-2u*(exp(ell)-1))
      A_*, B_*   = sup of sum_i phi_s(ell_i), sum_i psi_u(ell_i) over
                   {0<=ell_i<=rho_inf, sum ell_i <= nu}
      k          = 1/(2*c(rho_inf)),  c = exp(rho_inf) - 1

    Parameters
    ----------
    d : int
        Gaussian output dimension. Explicit to handle the rho_inf -> 0 edge
        case (where nu/rho_inf = d is indeterminate).

    Notes
    -----
    The 2D infimum is computed via a coarse grid search followed by local
    refinement using scipy.optimize.minimize (Nelder-Mead).
    """
    if rho_inf <= 0.0 or nu <= 0.0 or epsilon >= nu / 2.0:
        return 0.0

    c = math.exp(rho_inf) - 1.0
    k = 1.0 / (2.0 * c)  # upper bound on u (exclusive)

    # Grid search over (s, u) to find a good starting point.
    # s: the optimal s satisfies A_*'(s) = 2*eps; for large d this can be
    #    O(nu/eps) so we search up to a generous s_max.
    s_max = max(50.0, 10.0 * nu / max(epsilon, 1e-6))
    n_grid = 40
    s_vals = np.linspace(0.0, s_max, n_grid + 1)
    # u must stay strictly inside (0, k); use 99% of k as ceiling
    u_vals = np.linspace(0.0, 0.99 * k, n_grid + 1)[1:]  # exclude u=0

    best_val = math.inf
    best_s, best_u = 0.0, k * 0.01

    for s in s_vals:
        for u in u_vals:
            val = _cov_objective(s, u, epsilon, rho_inf, nu, d)
            if val < best_val:
                best_val = val
                best_s, best_u = s, u

    # Local refinement from the best grid point, parameterizing u via logit
    # to keep it in (0, k) and s via exp to keep it >= 0.
    def _neg_obj_transformed(params: np.ndarray) -> float:
        t_s, t_u = params
        if t_s > math.log(_EXP_CAP):  # cap s to avoid downstream overflow
            return math.inf
        s = math.exp(t_s)                         # s = exp(t_s) >= 0
        # u = k * sigmoid(t_u) to stay in (0, k)
        u = k / (1.0 + math.exp(-t_u))
        return _cov_objective(s, u, epsilon, rho_inf, nu, d)

    # Convert best grid point to transformed coordinates
    t_s0 = math.log(best_s + 1e-6)
    t_u0 = math.log(best_u / (k - best_u + 1e-10))

    try:
        from scipy.optimize import minimize
        res = minimize(
            _neg_obj_transformed,
            x0=np.array([t_s0, t_u0]),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000},
        )
        refined_val = res.fun
    except Exception:
        refined_val = best_val

    result = min(best_val, refined_val)

    # Clamp: delta must lie in [0, 1].
    return float(np.clip(result, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Loewner-comparable combined upper bound
# ---------------------------------------------------------------------------

def delta_bar_LC(
    epsilon: float,
    Delta: float,
    rho_inf: float,
    nu: float,
    d: int,
) -> float:
    """Combined IS upper bound (Definition 6).

    delta_bar_LC(eps; Delta, rho_inf, nu, d) =
        inf_{eps1 + eps2 = eps, eps1,eps2 >= 0}
            [delta_bar_mean(eps1; Delta) + exp(eps1) * delta_bar_cov(eps2; rho_inf, nu, d)]

    1D line search over eps1 in [0, eps] using scipy.optimize.minimize_scalar.
    """
    if epsilon < 0.0:
        return 1.0

    def _combined(eps1: float) -> float:
        eps2 = epsilon - eps1
        if eps2 < 0.0:
            return math.inf
        dm = delta_bar_mean(eps1, Delta)
        dc = delta_bar_cov(eps2, rho_inf, nu, d)
        return dm + math.exp(eps1) * dc

    result = minimize_scalar(
        _combined,
        bounds=(0.0, epsilon),
        method="bounded",
        options={"xatol": 1e-10},
    )
    val = float(result.fun)
    return float(np.clip(val, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Binary search for tau*
# ---------------------------------------------------------------------------

def find_tau_star(
    epsilon: float,
    delta: float,
    algorithm: "GaussianOutputAlgorithm",
    public_meta: dict,
    tau_lo: float = 0.0,
    tau_hi_init: float = 1.0,
    tol: float = 1e-8,
    max_doublings: int = 60,
) -> float:
    """Binary search for the smallest tau* >= 0 s.t. delta_bar_LC(...) <= delta.

    Paper: Figure 5, Step 1. Returns tau* (covariance inflation, NOT std dev).

    The upper bracket is doubled until delta_bar_LC(tau_hi) <= delta (same
    bracket-expansion pattern as _find_epsilon_T in rp_ndis.py).

    Parameters
    ----------
    epsilon, delta : float
        Target privacy parameters.
    algorithm : GaussianOutputAlgorithm
        Must implement sensitivity(tau, public_meta) -> NDISSensitivity.
    public_meta : dict
        Public dataset metadata; passed directly to algorithm.sensitivity().
    tau_lo : float
        Starting lower bracket (default 0).
    tau_hi_init : float
        Initial upper bracket; doubled until condition is satisfied.
    tol : float
        Convergence tolerance on tau.
    max_doublings : int
        Maximum bracket doublings before raising RuntimeError.
    """
    d = int(public_meta["d"])

    def _delta_at_tau(tau: float) -> float:
        sens = algorithm.sensitivity(tau, public_meta)
        return delta_bar_LC(epsilon, sens.Delta, sens.rho_inf, sens.nu, d)

    # Verify lower bracket: at tau_lo, delta_bar_LC should exceed delta
    # (otherwise tau* = tau_lo and no noise is needed — return 0).
    if _delta_at_tau(tau_lo) <= delta:
        return tau_lo

    # Expand upper bracket until it satisfies the condition
    tau_hi = tau_hi_init
    for _ in range(max_doublings):
        if _delta_at_tau(tau_hi) <= delta:
            break
        tau_hi *= 2.0
    else:
        raise RuntimeError(
            f"find_tau_star: could not find tau_hi satisfying delta_bar_LC <= {delta} "
            f"after {max_doublings} doublings (last tau_hi={tau_hi:.3e}). "
            "Check that epsilon > 0 and delta > 0 are achievable."
        )

    # Binary search within [tau_lo, tau_hi]
    while tau_hi - tau_lo > tol:
        tau_mid = (tau_lo + tau_hi) / 2.0
        if _delta_at_tau(tau_mid) <= delta:
            tau_hi = tau_mid
        else:
            tau_lo = tau_mid

    return tau_hi
