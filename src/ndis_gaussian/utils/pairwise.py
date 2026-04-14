"""Exact pairwise IS divergence computation — Phase B placeholder.

This module will implement exact computation of the hockey-stick divergence
delta_{M(D), M(D')}(epsilon) for specific neighboring pairs (D, D'), using
the generalized-chi^2 CDF approach from Theorem 8 / Definition 9 of the
NDIS paper.

The calibration-gap experiment (Phase B) will use these functions to compare
the generic wrapper's certified delta_bar_LC(tau*) against the exact pairwise
delta for specific neighbors, quantifying the conservatism of the certification.

Status: NOT IMPLEMENTED — deferred to Phase B.
"""


def exact_pairwise_delta(
    epsilon: float,
    mu1: "np.ndarray",
    Sigma1: "np.ndarray",
    mu2: "np.ndarray",
    Sigma2: "np.ndarray",
) -> float:
    """Compute delta_{X,Y}(epsilon) exactly for X~N(mu1,Sigma1), Y~N(mu2,Sigma2).

    Uses the generalized-chi^2 CDF representation from NDIS Theorem 1 and
    Definition 9. Requires a CDF oracle for the generalized-chi^2 distribution.

    Phase B implementation note: see Das (2025) [ref 19] or Davies (1973) [ref 20]
    for numerical methods for the generalized-chi^2 CDF.

    Status: NOT IMPLEMENTED.
    """
    raise NotImplementedError(
        "exact_pairwise_delta is not yet implemented. "
        "This is a Phase B placeholder for the calibration-gap experiment."
    )
