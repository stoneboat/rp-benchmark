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

  phi_s(ell) = -s*ell - 0.5 * log(1 - 2s*(exp(ell) - 1))
  psi_u(ell) = u*ell - 0.5 * log(1 + 2u*(1 - exp(-ell)))

  A_*(s; rho_inf, nu, d) = sup{ sum_i phi_s(ell_i) : 0<=ell_i<=rho_inf, sum<=nu }
  B_*(u; rho_inf, nu, d) = sup{ sum_i psi_u(ell_i) : 0<=ell_i<=rho_inf, sum<=nu }

  Both phi_s and psi_u are convex and non-decreasing in ell >= 0, so the sup
  is achieved at the concentrated vertex:
    b  = min(nu, d * rho_inf)
    k* = floor(b / rho_inf) coordinates at rho_inf
    r  = b - k* * rho_inf   one coordinate at r (if k* < d and r > 0)

  delta_bar_cov(eps; rho_inf, nu, d):
    = inf_{s in [0,k), u>=0} [
          exp(-2*eps*s + A_*(s))
        - exp(eps) * (1 - exp(2*eps*u + B_*(u)))
      ]

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
from scipy.special import erfcx, log_ndtr
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
# Exact scalar covariance shift (specialized GPR path only)
# ---------------------------------------------------------------------------

_LOG_TWO = math.log(2.0)
_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_PHI_ONE = math.exp(-0.5) / math.sqrt(2.0 * math.pi)
_MIN_SUBNORMAL = float(np.nextafter(0.0, 1.0))
_LOG_MIN_SUBNORMAL = math.log(_MIN_SUBNORMAL)
# SciPy's special functions are highly accurate but do not promise directed
# rounding.  Reserve a small absolute allowance so this privacy path does not
# mistake an ordinary several-ULP approximation for a certified upper value.
_BINARY64_ABS_SLACK = 16.0 * np.finfo(float).eps


def _scalar_covariance_derivative_upper(ell: float, t0: float) -> float:
    """Rigorous fallback from d(delta)/d(ell) = u*phi(u) <= phi(1).

    The sharper endpoint value is used when ``u=sqrt(t0) >= 1``: in that
    regime u(s) decreases and u(s)*phi(u(s)) increases on ``[0, ell]``.
    This path is used only when the exact binary64 tail calculation is below
    range or loses its near-zero tail difference.
    """
    if ell <= 0.0:
        return 0.0

    if math.isfinite(t0) and t0 >= 1.0:
        log_upper = (
            math.log(ell)
            + 0.5 * math.log(t0)
            - 0.5 * t0
            - _LOG_SQRT_TWO_PI
        )
        if log_upper <= _LOG_MIN_SUBNORMAL:
            return _MIN_SUBNORMAL
        upper = math.exp(log_upper)
    else:
        upper = ell * _PHI_ONE

    if upper <= 0.0:
        return _MIN_SUBNORMAL
    if upper >= 1.0:
        return 1.0
    return float(np.nextafter(upper, math.inf))


def _delta_cov_scalar_exact(epsilon: float, ell: float) -> float:
    """Exact hard-direction scalar Gaussian covariance divergence.

    This evaluates the hockey-stick divergence between ``N(0, exp(ell))``
    and ``N(0, 1)`` at privacy budget ``epsilon``.  For ``ell > 0``, let

      t0 = (ell + 2*epsilon) / (exp(ell) - 1).

    The exact value is ``2*sf(sqrt(t0)) -
    2*exp(epsilon)*sf(sqrt(exp(ell)*t0))``.  The implementation avoids both
    ``exp(ell)`` and subtraction of the two tails.  If ``A`` is the first
    tail and ``B`` the epsilon-weighted second tail, then

      log(B/A) = -ell/2
                 + log(erfcx(sqrt(exp(ell)*t0/2)))
                 - log(erfcx(sqrt(t0/2))).

    The scalar envelope needs no numerical maximization.  Differentiating the
    hockey-stick integral over its likelihood-ratio region gives

      d delta / d ell = sqrt(t0) * phi(sqrt(t0)) > 0.

    Hence its worst case on ``0 <= ell <= rho_inf`` is exactly ``ell=rho_inf``.
    At binary64 underflow/cancellation extremes, a derivative-based upper bound
    is returned conservatively. Ordinary finite regimes evaluate the exact
    formula and add the documented binary64 allowance used by this privacy path.
    """
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError(f"epsilon must be finite and >= 0, got {epsilon}")
    if not math.isfinite(ell) or ell < 0.0:
        raise ValueError(f"ell must be finite and >= 0, got {ell}")
    if ell == 0.0:
        return 0.0

    denominator = -math.expm1(-ell)  # 1 - exp(-ell), accurate near zero
    numerator = ell + 2.0 * epsilon
    if not math.isfinite(numerator):
        return _scalar_covariance_derivative_upper(ell, math.nan)

    tau_t0 = numerator / denominator
    t0 = tau_t0 * math.exp(-ell)
    if not math.isfinite(tau_t0) or not math.isfinite(t0):
        return _scalar_covariance_derivative_upper(ell, t0)

    sqrt_t0 = math.sqrt(t0)
    log_first_tail = _LOG_TWO + float(log_ndtr(-sqrt_t0))
    # The divergence is bounded above by the first tail.  If that tail is
    # smaller than binary64's minimum positive value, returning the minimum
    # subnormal is conservative and avoids reporting an underflowed zero.
    if log_first_tail <= _LOG_MIN_SUBNORMAL:
        return _MIN_SUBNORMAL

    erfcx_small = float(erfcx(math.sqrt(t0 / 2.0)))
    erfcx_large = float(erfcx(math.sqrt(tau_t0 / 2.0)))
    if erfcx_small <= 0.0 or not math.isfinite(erfcx_small):
        return _scalar_covariance_derivative_upper(ell, t0)

    if erfcx_large == 0.0:
        log_tail_ratio = -math.inf
    elif erfcx_large < 0.0 or not math.isfinite(erfcx_large):
        return _scalar_covariance_derivative_upper(ell, t0)
    else:
        # log1p retains the erfcx ratio when the two thresholds are close.
        relative_change = (erfcx_large - erfcx_small) / erfcx_small
        if relative_change <= -1.0:
            log_erfcx_ratio = -math.inf
        else:
            log_erfcx_ratio = math.log1p(relative_change)
        log_tail_ratio = -0.5 * ell + log_erfcx_ratio

    # Mathematically log_tail_ratio < 0 for ell > 0.  Very close to identical
    # covariance, binary64 may round it to zero; use the proven derivative
    # upper bound instead of turning that cancellation into an invalid zero.
    if log_tail_ratio >= 0.0 or -log_tail_ratio < 1e-10:
        return _scalar_covariance_derivative_upper(ell, t0)

    gap = -math.expm1(log_tail_ratio)
    value = math.exp(log_first_tail) * gap
    if not math.isfinite(value) or value <= 0.0:
        return _scalar_covariance_derivative_upper(ell, t0)
    if value >= 1.0:
        return 1.0
    return float(min(1.0, value + _BINARY64_ABS_SLACK))


def _delta_mean_scalar_stable_upper(epsilon: float, Delta: float) -> float:
    """Stable scalar Gaussian mean divergence for the specialized path.

    This is algebraically the same quantity as :func:`delta_bar_mean`, but it
    avoids ``exp(epsilon)`` and catastrophic tail subtraction.  A conservative
    binary64 allowance is added because ``log_ndtr`` and ``erfcx`` are not
    directed-rounding routines.  The generic baseline deliberately continues
    to call the existing helper unchanged.
    """
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError(f"epsilon must be finite and >= 0, got {epsilon}")
    if math.isnan(Delta) or Delta < 0.0:
        raise ValueError(f"Delta must be >= 0, got {Delta}")
    if Delta == 0.0:
        return 0.0
    if math.isinf(Delta):
        return 1.0

    center = epsilon / Delta
    a = center - Delta / 2.0
    b = center + Delta / 2.0

    if a >= 0.0:
        erfcx_a = float(erfcx(a / math.sqrt(2.0)))
        erfcx_b = float(erfcx(b / math.sqrt(2.0)))
        if erfcx_a <= 0.0 or not math.isfinite(erfcx_a):
            return _BINARY64_ABS_SLACK
        log_first_tail = -_LOG_TWO - 0.5 * a * a + math.log(erfcx_a)
        if log_first_tail <= _LOG_MIN_SUBNORMAL:
            return _BINARY64_ABS_SLACK
        if erfcx_b == 0.0:
            log_tail_ratio = -math.inf
        elif erfcx_b < 0.0 or not math.isfinite(erfcx_b):
            return float(min(1.0, math.exp(log_first_tail) + _BINARY64_ABS_SLACK))
        else:
            relative_change = (erfcx_b - erfcx_a) / erfcx_a
            log_tail_ratio = (
                -math.inf
                if relative_change <= -1.0
                else math.log1p(relative_change)
            )
    else:
        log_first_tail = float(log_ndtr(-a))
        log_second_tail = epsilon + float(log_ndtr(-b))
        log_tail_ratio = log_second_tail - log_first_tail

    # The first tail alone is always an upper bound if binary64 loses the
    # positive difference between the two exact terms.
    if log_tail_ratio >= 0.0:
        value = math.exp(log_first_tail)
    else:
        value = math.exp(log_first_tail) * (-math.expm1(log_tail_ratio))
    if not math.isfinite(value) or value <= 0.0:
        return _BINARY64_ABS_SLACK
    return float(min(1.0, value + _BINARY64_ABS_SLACK))


# ---------------------------------------------------------------------------
# Internal helpers for the envelope functionals A_* and B_*
# ---------------------------------------------------------------------------

def _phi_s(s: float, ell: float) -> float:
    """Per-dimension contribution to A_*.

    phi_s(ell) = -s*ell - 0.5*log(1-2s*(exp(ell)-1)), with
    0 <= s < 1/(2*(exp(rho_inf)-1)). The caller enforces the bound on s.
    """
    if ell <= 0.0 or s <= 0.0:
        return 0.0
    log_arg = -2.0 * s * math.expm1(ell)
    if log_arg <= -1.0:
        return math.inf
    return -s * ell - 0.5 * math.log1p(log_arg)


def _psi_u(u: float, ell: float) -> float:
    """Per-dimension contribution to B_*.

    psi_u(ell) = u*ell - 0.5*log(1+2u*(1-exp(-ell))), with u >= 0.
    """
    if ell <= 0.0 or u <= 0.0:
        return 0.0
    return u * ell - 0.5 * math.log1p(2.0 * u * (-math.expm1(-ell)))


def _A_star(s: float, rho_inf: float, nu: float, d: int) -> float:
    """Envelope functional A_*(s; rho_inf, nu, d) = sup sum_i phi_s(ell_i).

    Since phi_s is convex and non-decreasing, the sup is at the concentrated vertex:
      b  = min(nu, d*rho_inf)
      k* = floor(b/rho_inf) coordinates at rho_inf
      r  = b - k* * rho_inf one coordinate at r (if k* < d and r > 0)
    """
    if rho_inf <= 0.0 or nu <= 0.0 or s <= 0.0:
        return 0.0
    budget = min(nu, d * rho_inf)
    quotient, r = divmod(budget, rho_inf)
    k_star = min(d, int(quotient))
    r = min(rho_inf, r)
    result = k_star * _phi_s(s, rho_inf)
    if k_star < d and r > 0.0:
        result += _phi_s(s, r)
    return result


def _B_star(u: float, rho_inf: float, nu: float, d: int) -> float:
    """Envelope functional B_*(u; rho_inf, nu, d) = sup sum_i psi_u(ell_i).

    Same vertex structure as A_*.
    """
    if rho_inf <= 0.0 or nu <= 0.0 or u <= 0.0:
        return 0.0
    budget = min(nu, d * rho_inf)
    quotient, r = divmod(budget, rho_inf)
    k_star = min(d, int(quotient))
    r = min(rho_inf, r)
    result = k_star * _psi_u(u, rho_inf)
    if k_star < d and r > 0.0:
        result += _psi_u(u, r)
    return result


# ---------------------------------------------------------------------------
# Covariance-shift upper bound (updated Definition 6)
# ---------------------------------------------------------------------------

_EXP_CAP = 700.0  # avoid math range error in math.exp


def _cov_objective(s: float, u: float, epsilon: float,
                   rho_inf: float, nu: float, d: int) -> float:
    """The 2D objective for delta_bar_cov at a specific (s, u).

    Returns math.inf when either exponential term would overflow. Such points
    are never the infimum (the grid search/Nelder-Mead will ignore them).
    """
    A = _A_star(s, rho_inf, nu, d)
    B = _B_star(u, rho_inf, nu, d)
    arg1 = -2.0 * epsilon * s + A
    if arg1 > _EXP_CAP:
        return math.inf   # too large; not the infimum
    first = math.exp(arg1)
    arg2 = 2.0 * epsilon * u + B
    if arg2 > _EXP_CAP:
        return math.inf
    if arg2 == 0.0:
        return first
    # -exp(epsilon) * (1 - exp(arg2)) = exp(epsilon) * expm1(arg2).
    # Work in the log domain to avoid both cancellation at small arg2 and
    # exp(epsilon) overflow when their product is still representable.
    second_log = epsilon + math.log(math.expm1(arg2))
    if second_log > _EXP_CAP:
        return math.inf
    return first + math.exp(second_log)


def delta_bar_cov(
    epsilon: float,
    rho_inf: float,
    nu: float,
    d: int,
) -> float:
    """IS upper bound for the covariance-shift contribution (updated Definition 6).

    Uses the tighter envelope functionals A_* and B_* derived from Lemma 2:

      delta_bar_cov(eps; rho_inf, nu, d) =
        inf_{s in [0,k), u>=0} [
            exp(-2*eps*s + A_*(s; rho_inf, nu, d))
          - exp(eps) * (1 - exp(2*eps*u + B_*(u; rho_inf, nu, d)))
        ]

    where:
      phi_s(ell) = -s*ell - 0.5*log(1-2s*(exp(ell)-1))
      psi_u(ell) = u*ell - 0.5*log(1+2u*(1-exp(-ell)))
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
    refinement using scipy.optimize.minimize (Nelder-Mead). The refinement
    parameterizes s with a scaled sigmoid and u with an exponential so that
    the mathematical domains remain s in (0, k) and u in (0, infinity).

    Large rho_inf guard: when rho_inf > 700, exp(rho_inf) approaches binary64
    overflow. In this regime k = 1/(2c) -> 0, the s-domain collapses, and the
    infimum approaches its value at s=u=0, namely 1. We return 1.0 directly.
    """
    if rho_inf <= 0.0 or nu <= 0.0:
        return 0.0

    # Guard against exp(rho_inf) overflow: when rho_inf > 700, k -> 0 and
    # the objective approaches 1 (no useful covariance-shift bound).
    if rho_inf > 700.0:
        return 1.0

    c = math.expm1(rho_inf)
    k = 1.0 / (2.0 * c)  # upper bound on s (exclusive)

    # Grid search over (s, u) to find a good starting point.
    # u is unbounded, so use a generous grid range before unconstrained local
    # refinement in log coordinates.
    u_max = max(50.0, 10.0 * nu / max(epsilon, 1e-6))
    n_grid = 40
    # Keep the coarse grid safely inside the open s < k boundary. Local
    # refinement can approach the boundary more closely through the sigmoid.
    s_vals = np.linspace(0.0, 0.99 * k, n_grid + 1)
    u_vals = np.linspace(0.0, u_max, n_grid + 1)

    best_val = math.inf
    best_s, best_u = 0.0, 0.0

    for s in s_vals:
        for u in u_vals:
            val = _cov_objective(s, u, epsilon, rho_inf, nu, d)
            if val < best_val:
                best_val = val
                best_s, best_u = s, u

    # A zero produced by exponential underflow is already the smallest value
    # representable after the final [0, 1] clamp, so refinement cannot improve it.
    if best_val <= 0.0:
        return 0.0

    # Local refinement from the best grid point, parameterizing s via logit
    # to keep it in (0, k) and u via exp to keep it >= 0.
    def _neg_obj_transformed(params: np.ndarray) -> float:
        t_s, t_u = params
        if t_s >= 0.0:
            z = math.exp(-t_s)
            s_fraction = 1.0 / (1.0 + z)
        else:
            z = math.exp(t_s)
            s_fraction = z / (1.0 + z)
        # Floating-point sigmoid can round to one for large t_s. nextafter
        # preserves the strict mathematical boundary in that case.
        s = min(k * s_fraction, float(np.nextafter(k, 0.0)))
        if t_u > math.log(np.finfo(float).max):
            return math.inf
        u = math.exp(t_u)
        return _cov_objective(s, u, epsilon, rho_inf, nu, d)

    # Convert best grid point to transformed coordinates
    s_fraction0 = min(max(best_s / k, 1e-12), 1.0 - 1e-12)
    t_s0 = math.log(s_fraction0 / (1.0 - s_fraction0))
    t_u0 = math.log(best_u + 1e-6)

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
# Scalar-GPR exact covariance specialization
# ---------------------------------------------------------------------------

def scalar_gpr_privacy_certificate(
    epsilon: float,
    Delta: float,
    rho_inf: float,
) -> dict[str, float]:
    """Return the optimized covariance-first scalar-GPR privacy certificate.

    Only the covariance term is specialized.  The composition requested for
    scalar GPR is retained verbatim:

      inf_{eps_cov + eps_mean = epsilon} [
          delta_cov_exact(eps_cov; rho_inf)
        + exp(eps_cov) * delta_bar_mean(eps_mean; Delta)
      ].

    The exact scalar covariance envelope is evaluated at ``rho_inf`` because
    ``_delta_cov_scalar_exact`` is nondecreasing in ``ell`` by the derivative
    proof in its docstring.  Both privacy-budget endpoints are evaluated
    explicitly in addition to the bounded scalar optimization.
    """
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError(f"epsilon must be finite and >= 0, got {epsilon}")
    if math.isnan(Delta) or Delta < 0.0:
        raise ValueError(f"Delta must be >= 0, got {Delta}")
    if not math.isfinite(rho_inf) or rho_inf < 0.0:
        raise ValueError(f"rho_inf must be finite and >= 0, got {rho_inf}")

    def _components(epsilon_cov: float) -> tuple[float, float, float, float]:
        epsilon_mean = epsilon - epsilon_cov
        delta_covariance = _delta_cov_scalar_exact(epsilon_cov, rho_inf)
        delta_mean_raw = _delta_mean_scalar_stable_upper(epsilon_mean, Delta)
        try:
            delta_mean_contribution = math.exp(epsilon_cov) * delta_mean_raw
        except OverflowError:
            delta_mean_contribution = math.inf if delta_mean_raw > 0.0 else 0.0
        total = delta_covariance + delta_mean_contribution
        return total, delta_covariance, delta_mean_raw, delta_mean_contribution

    candidate_splits = [0.0, epsilon]
    if epsilon > 0.0:
        result = minimize_scalar(
            lambda eps_cov: _components(float(eps_cov))[0],
            bounds=(0.0, epsilon),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 1000},
        )
        if math.isfinite(float(result.x)):
            candidate_splits.append(float(result.x))

    epsilon_cov = min(candidate_splits, key=lambda split: _components(split)[0])
    total, delta_covariance, delta_mean_raw, delta_mean_contribution = _components(
        epsilon_cov
    )
    epsilon_mean = epsilon - epsilon_cov

    return {
        "epsilon_cov": float(epsilon_cov),
        "epsilon_mean": float(epsilon_mean),
        "delta_covariance": float(delta_covariance),
        "delta_mean_raw": float(delta_mean_raw),
        "delta_mean_contribution": float(delta_mean_contribution),
        "delta_total": float(min(1.0, total)),
    }


def delta_bar_LC_scalar_gpr(
    epsilon: float,
    Delta: float,
    rho_inf: float,
) -> float:
    """Specialized scalar-GPR bound with exact covariance-only NDIS."""
    return scalar_gpr_privacy_certificate(epsilon, Delta, rho_inf)["delta_total"]


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


def find_tau_star_scalar_gpr(
    epsilon: float,
    delta: float,
    algorithm: "GaussianOutputAlgorithm",
    public_meta: dict,
    tau_lo: float = 0.0,
    tau_hi_init: float = 1.0,
    tol: float = 1e-8,
    max_doublings: int = 60,
) -> float:
    """Calibrate scalar GPR using exact covariance NDIS and the mean bound.

    This is deliberately separate from :func:`find_tau_star`: the latter
    remains the corrected generic Definition-6 calibration.  The specialized
    path is valid only for scalar output and expects GPR's scalar sensitivity
    identity ``nu == rho_inf``.
    """
    d = int(public_meta["d"])
    if d != 1:
        raise ValueError(
            "find_tau_star_scalar_gpr requires scalar Gaussian output (d=1), "
            f"got d={d}"
        )

    def _delta_at_tau(tau: float) -> float:
        sens = algorithm.sensitivity(tau, public_meta)
        if not math.isclose(sens.nu, sens.rho_inf, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(
                "scalar GPR exact calibration requires nu == rho_inf; "
                f"got nu={sens.nu}, rho_inf={sens.rho_inf} at tau={tau}"
            )
        return delta_bar_LC_scalar_gpr(
            epsilon=epsilon,
            Delta=sens.Delta,
            rho_inf=sens.rho_inf,
        )

    if _delta_at_tau(tau_lo) <= delta:
        return tau_lo

    tau_hi = tau_hi_init
    for _ in range(max_doublings):
        if _delta_at_tau(tau_hi) <= delta:
            break
        tau_hi *= 2.0
    else:
        raise RuntimeError(
            "find_tau_star_scalar_gpr: could not find tau_hi satisfying the "
            f"specialized bound <= {delta} after {max_doublings} doublings "
            f"(last tau_hi={tau_hi:.3e})."
        )

    while tau_hi - tau_lo > tol:
        tau_mid = (tau_lo + tau_hi) / 2.0
        if _delta_at_tau(tau_mid) <= delta:
            tau_hi = tau_mid
        else:
            tau_lo = tau_mid

    return tau_hi
