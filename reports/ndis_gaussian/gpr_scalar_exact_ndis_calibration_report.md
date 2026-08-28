# Scalar-GPR exact-NDIS covariance calibration

Rerun date: 2026-08-27. Target privacy parameters: `(epsilon, delta) = (1, 1e-5)`.

## Outcome

The application-specific scalar exact-NDIS path certifies a smaller covariance inflation than the corrected generic Definition-6 wrapper while preserving the corrected generic implementation as the default baseline.

| GPR calibration | `sigma_std` | covariance inflation `tau_star` | abs. error mean | abs. error std (`ddof=0`) | privacy status |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical buggy | 0.8213393551 | 0.6745983362 | 0.7562774987 | 0.5289865769 | Invalid after hard-direction correction |
| Corrected generic | 3.9022548673 | 15.2275930494 | 2.2830794601 | 1.7258868658 | Valid generic Definition-6 bound |
| Scalar exact NDIS | 2.9406870828 | 8.6476405188 | 1.7367306289 | 1.3395429766 | Valid specialized scalar certificate |

All three rows above use the published legacy seeds `42, 43, 44, 45, 46`. The historical row is retained only as a correction record; it is not a valid private baseline. The scalar result should be compared with the corrected generic result when comparing two valid certificates.

For the currently checked-in seeds `2734335219, 2905568749, 2703185305, 3085492084, 1659915345`, the scalar exact result is `1.7585650501 +/- 0.9743238509`. The corrected generic result on those seeds is `2.3114286897 +/- 1.1523317203`. Population standard deviation (`ddof=0`) is used throughout.

## Proposed updated Table 1

The correction and scalar specialization imply the following paper-facing table. BLR uses the corrected generic wrapper; scalar GPR uses the explicitly labeled application-specific exact-NDIS covariance calibration.

| Alg. | Dataset | Metric | Non-private | Private |
| --- | --- | --- | ---: | ---: |
| BLR | Breast Cancer | Acc. ↑ | 0.9912 | 0.6982 +/- 0.1663 |
| GPR | Linnerud | Abs. err. ↓ | 0.4718 | 1.7367 +/- 1.3395 |

Results average over five runs. The calibrated noise standard deviations are `sigma_star = 4.1229` for BLR's corrected generic calibration and `sigma_star = 2.9407` for GPR's scalar exact-NDIS calibration. Their covariance inflations are `tau_star = 16.9984054044` and `8.6476405188`, respectively. For transparency, the corrected generic GPR baseline remains `sigma_star = 3.9023` and `2.2831 +/- 1.7259` absolute error.

## Exact scalar covariance term

For the hard covariance direction, let

```text
P_tau = N(0, tau),  Q = N(0, 1),  tau = exp(ell) >= 1,
t0 = (ell + 2 epsilon) / (tau - 1).
```

The implementation evaluates

```text
delta_cov_exact(epsilon; ell)
  = 2 sf(sqrt(t0)) - 2 exp(epsilon) sf(sqrt(tau t0)),
```

with the continuous value zero at `ell = 0`. It reproduces the correction diagnostic:

```text
analytic delta_cov_exact(1; 1) = 0.10655957800717815987...
conservative binary64 helper value = 0.10655957800718163
delta_bar_cov(1, rho_inf=1, nu=1, d=1) = 0.9099902959890521
```

No `epsilon >= nu/2` cutoff is used for the hard direction.

The hard direction also covers the reverse ordered-neighbor direction. Lemma 3 of the full-version paper proves covariance-order asymmetry: for equal means and `Sigma_max >= Sigma_min` in Loewner order, the more-spread-to-less-spread divergence is at least the reverse divergence for every `epsilon >= 0`. Here `N(0, exp(ell))` is the more-spread scalar Gaussian, so the exact hard-direction value upper-bounds both adjacency orientations. A focused regression additionally checks this dominance across scalar grids; the reverse divergence becomes exactly zero once `epsilon >= ell/2`, while the hard direction generally does not.

### Proof that the scalar-envelope endpoint is worst case

Let `u = sqrt(t0)` and let `A_tau` be the positive likelihood-ratio region. Then

```text
delta(tau) = integral over A_tau of (p_tau - exp(epsilon) q).
```

The integrand is zero at the moving boundary, so its boundary derivative vanishes. Differentiating under the integral gives

```text
d delta / d tau
  = (1 / (2 tau)) E[(Z^2 - 1) 1{|Z| > u}]
  = u phi(u) / tau.
```

Therefore

```text
d delta / d ell = u phi(u) > 0
```

for every finite `ell > 0` and `epsilon >= 0`. Together with continuity at zero, this proves that the rigorous maximum over `0 <= ell <= rho_inf` occurs at `ell = rho_inf`. The implementation therefore does not rely on a local envelope optimizer.

## Specialized composition and certificate

Only the scalar covariance term is replaced. The requested covariance-first composition is

```text
inf over epsilon_cov + epsilon_mean = epsilon of
  delta_cov_exact(epsilon_cov; rho_inf)
  + exp(epsilon_cov) delta_mean(epsilon_mean; Delta).
```

At the calibrated `tau_star = 8.647640518844128`, the public GPR sensitivity and optimized certificate are:

| Quantity | Value |
| --- | ---: |
| `epsilon_cov` | 0.8284878615692024 |
| `epsilon_mean` | 0.1715121384307976 |
| covariance contribution | 9.189187939330935e-6 |
| raw mean-divergence upper value | 3.540890291287905e-7 |
| weighted mean contribution | 8.108120187556392e-7 |
| total certified upper bound | 9.999999958086574e-6 |
| `Delta` | 0.04301413337968837 |
| `rho_inf = nu` | 0.10942686828854059 |
| maximum scalar variance ratio `exp(rho_inf)` | 1.1156384794003513 |

At `tau_star - 1e-8 = 8.647640508844127`, the implemented predicate is `1.0000000056896125e-5`, which exceeds the target. This confirms that the passing upper bracket is near-minimal at the configured binary-search tolerance.

A separate 100-digit recomputation of the unpadded analytic terms at the reported split gives total `9.999999946398633223e-6`. The binary64 certificate above is deliberately larger because it includes the numerical allowance, and both values remain below the target.

The submitted PDF and this task specify covariance-first composition. The repository's existing generic `delta_bar_LC` retains its historical mean-first ordering. The tests therefore establish the exact-versus-generic substitution pointwise under a common covariance-first ordering, and separately regression-check that the specialized result is also no worse than the existing generic workflow on representative GPR sensitivity tuples. The generic function itself was not changed by this specialization.

## Numerical evaluation

The scalar helper uses `expm1` near `ell = 0`, `log_ndtr` for the first Gaussian tail, and an `erfcx` tail ratio with `expm1` for the final positive difference. It never forms `exp(epsilon)` or `exp(ell)` in the exact covariance calculation. Both privacy-split endpoints are evaluated explicitly. In binary64 underflow or near-identical-covariance cancellation regimes, the implementation returns a mathematically justified upper bound based on

```text
d delta / d ell = u phi(u) <= phi(1),
```

rather than an invalid underflowed zero. Because SciPy's special functions are not directed-rounding routines, ordinary-regime covariance and mean evaluations add an explicit `16 * machine_epsilon` absolute binary64 allowance before use in the certificate. An independent 90-100-digit audit covered 7,252 covariance points and 7,210 mean points, including randomized and extreme inputs, with zero underestimates after this allowance. This is a broadly validated conservative binary64 safeguard, not a claim of formal interval arithmetic. High-precision reference values are also retained as focused regressions.

## Code paths changed

- `src/ndis_gaussian/calibration.py`: added the stable exact scalar covariance helper, a stable specialized mean evaluator, the covariance-first scalar certificate, and a separate scalar-GPR binary search. The corrected generic covariance, generic LC objective, and generic finder remain available.
- `src/ndis_gaussian/wrapper.py`: added `ScalarGPRExactNDISWrapper`; release sampling is inherited unchanged, and `NDISGaussianWrapper` remains generic by default.
- `src/ndis_gaussian/__init__.py`: exported the specialized wrapper.
- `scripts/run_gpr_ndis_demo.py`: added explicit `generic` / `scalar_gpr_exact` routing and JSONL certificate diagnostics. Missing mode still means generic.
- `configs/gpr_demo/linnerud_scalar_exact_legacy_seeds.yaml`: added a dedicated exact-mode config with legacy seeds `42-46`; the checked-in generic config is unchanged.
- `tests/test_gpr_ndis_smoke.py`: added focused exact-formula, quadrature, envelope, upper-bound, composition, calibration, and numerical-extreme regressions.

This is an application-specific direct NDIS calibration for scalar GPR. It does not change or "fix" Definition 6, and it does not implement the optional full joint mean-and-variance scalar optimization.

## Verification

Focused scalar tests:

```bash
PYTHONPATH=src /tmp/python-venv/rp_benchmark_venv/bin/python -m pytest -q -W error \
  tests/test_gpr_ndis_smoke.py -k scalar_exact
```

Result: `23 passed, 34 deselected in 2.97s`.

Full repository suite:

```bash
PYTHONPATH=src /tmp/python-venv/rp_benchmark_venv/bin/python -m pytest -q
```

Result: `133 passed in 215.98s`.

The focused tests compare five exact values against direct integration of the positive Gaussian-density difference, include the `0.10655957800717819` correction diagnostic, retain high-precision one-sided references, check hard-direction dominance and exact covariance against the corrected generic bound, exercise the proved monotonicity on broad grids, test large-`epsilon` stability, and verify both the passing calibration and the immediately lower failing predicate.

## Reproduction and raw outputs

Published legacy seeds:

```bash
/tmp/python-venv/rp_benchmark_venv/bin/python scripts/run_gpr_ndis_demo.py \
  --config configs/gpr_demo/linnerud_scalar_exact_legacy_seeds.yaml \
  --output-root data/outputs/gpr_scalar_exact_ndis/bell_py311/legacy_seeds
```

Execution output: `data/outputs/gpr_scalar_exact_ndis/bell_py311/legacy_seeds/gpr_ndis_results_20260827-203857.jsonl`.

Tracked raw artifact: `reports/ndis_gaussian/raw/gpr_scalar_exact_legacy_seeds_20260827-203857.jsonl`.

Current checked-in seeds:

```bash
/tmp/python-venv/rp_benchmark_venv/bin/python scripts/run_gpr_ndis_demo.py \
  --config configs/gpr_demo/linnerud.yaml \
  --calibration-mode scalar_gpr_exact \
  --output-root data/outputs/gpr_scalar_exact_ndis/bell_py311/current_seeds
```

Execution output: `data/outputs/gpr_scalar_exact_ndis/bell_py311/current_seeds/gpr_ndis_results_20260827-203857.jsonl`.

Tracked raw artifact: `reports/ndis_gaussian/raw/gpr_scalar_exact_current_seeds_20260827-203857.jsonl`.

The corrected generic legacy result can be reproduced with the dedicated legacy config plus `--calibration-mode generic`. Historical and corrected-generic raw paths, the two new scalar paths, full-precision metrics, sensitivities, and certificate components are recorded in `reports/ndis_gaussian/gpr_scalar_exact_ndis_before_after.csv`.

Execution environment: Purdue Bell host `bell-a153.rcac.purdue.edu`; CPU-only AMD EPYC 7662 node; Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0, and pytest 9.1.1. No GPU was used.

## Rebuttal interpretation

The reviewer identified a real implementation error in the generic covariance calibration. The corrected generic result remains reported and available. That correction exposed substantial conservatism in the generic analytic covariance envelope. Because this GPR release has scalar Gaussian output, exact one-dimensional covariance-shift NDIS yields a tighter application-specific certificate while retaining a rigorous worst-case scalar envelope and the requested mean/covariance composition.
