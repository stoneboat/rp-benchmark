# NDIS hard-covariance calibration correction

Rerun date: 2026-08-27. Privacy parameters: `(epsilon, delta) = (1, 1e-5)`.

## Outcome

Both published noise standard deviations change materially after correcting the hard covariance direction.

| Application | Repository before `sigma_std` | Corrected `sigma_std` | Absolute change | Relative change |
| --- | ---: | ---: | ---: | ---: |
| BLR, Breast Cancer | 3.1001567441 | 4.1229122480 | +1.0227555038 | +32.9904% |
| GPR, Linnerud | 0.8213393551 | 3.9022548673 | +3.0809155122 | +375.1087% |

The repository represents covariance inflation as `tau_star` and reports its square root as `sigma_std`. The paper rounds the repository-before values to `3.1002` and `0.8213`; the changes above use the underlying full-precision JSONL values. Relative to the literal rounded paper numbers, the changes are +1.0227122480 (+32.9886%) for BLR and +3.0809548673 (+375.1315%) for GPR. These paper values are `sigma_std`, not `tau_star`. The corresponding covariance inflations change from `9.6109718382` to `16.9984054044` for BLR (+76.8646%) and from `0.6745983362` to `15.2275930494` for GPR (+2157.2829%).

Using the same seeds as the published rows, the Section 6 results change as follows. Standard deviations use the existing population convention (`ddof=0`).

| Application | Metric | Before | Corrected | Non-private |
| --- | --- | ---: | ---: | ---: |
| BLR, Breast Cancer | Accuracy | 0.7649122807 +/- 0.1228571326 | 0.6982456140 +/- 0.1663061660 | 0.9912280702 |
| GPR, Linnerud, legacy seeds 42-46 | Absolute error | 0.7562774987 +/- 0.5289865769 | 2.2830794601 +/- 1.7258868658 | 0.4718045189 |

Therefore Table 1's private BLR and GPR entries and its two calibrated-`sigma` values must be corrected. The non-private entries do not change. The `sigma` description should also distinguish the noise standard deviation from the covariance inflation `tau_star`.

## Code correction

- `src/ndis_gaussian/calibration.py:76-97`: `_phi_s` now implements `-s*ell - 0.5*log(1 - 2*s*(exp(ell)-1))`; `_psi_u` now implements `u*ell - 0.5*log(1 + 2*u*(1-exp(-ell)))`. `expm1` and `log1p` retain accuracy near zero, and `_phi_s` returns infinity at or beyond its singular boundary.
- `src/ndis_gaussian/calibration.py:100-134`: `_A_star` and `_B_star` retain the envelope `0 <= ell_i <= rho_inf`, `sum ell_i <= nu`, cap its usable budget at `min(nu, d*rho_inf)`, and use the same concentrated vertex without creating an extra coordinate when `nu > d*rho_inf`.
- `src/ndis_gaussian/calibration.py:144-168`: `_cov_objective` evaluates the unchanged Definition 6 objective with `expm1` and a combined-log overflow guard, including an exact `u=0` path for large valid `epsilon` values.
- `src/ndis_gaussian/calibration.py:171-285`: `delta_bar_cov` no longer returns zero when `epsilon >= nu/2`. It keeps only zero covariance-sensitivity cases. The optimizer now enforces `0 <= s < k(rho_inf)` with a scaled, numerically safe sigmoid and lets `u >= 0` remain unbounded through a log/exponential parameterization. The coarse grid includes both allowed zero endpoints; `nextafter` keeps refined `s` strictly below the open boundary.
- `tests/test_blr_ndis_smoke.py:170-253`: focused regressions cover the hard one-dimensional direction, large-`epsilon` safety, exact formulas, zero sensitivity, envelope constraints, the open boundary, and both optimizer domains.

For `N(0,e)` versus `N(0,1)` at `epsilon=1`, the exact hard-direction hockey-stick divergence is `0.10655957800717819`. The corrected upper bound is:

```text
delta_bar_cov(1, rho_inf=1, nu=1, d=1) = 0.9099902959890521
```

It is finite, strictly positive, and above the exact divergence. The old result was zero because the removed cutoff applies only to the reverse/easy covariance direction.

## Verification

Targeted regressions:

```bash
/tmp/python-venv/rp_benchmark_venv/bin/python -m pytest -q -W error \
  tests/test_blr_ndis_smoke.py \
  -k 'delta_bar_cov_one_dimensional_hard_direction or delta_bar_cov_large_epsilon_is_finite or delta_bar_cov_zero_sensitivity or covariance_envelope_scalar_formulas or covariance_envelope_constraints or delta_bar_cov_optimizer_domains'
```

Result: `8 passed, 33 deselected in 0.73s`.

Full repository suite:

```bash
/tmp/python-venv/rp_benchmark_venv/bin/python -m pytest -q
```

Result: `110 passed in 207.74s`.

## Section 6 rerun

The existing Bell environment created according to `scripts/local_scripts/cluster_install_bell.sh` was already present, so it was reused without reinstalling dependencies. The exact corrected current-config commands were:

```bash
/tmp/python-venv/rp_benchmark_venv/bin/python scripts/run_blr_ndis_demo.py \
  --config configs/blr_demo/breast_cancer.yaml \
  --output-root data/outputs/ndis_covariance_correction/bell_py311/verified_final_after/blr

/tmp/python-venv/rp_benchmark_venv/bin/python scripts/run_gpr_ndis_demo.py \
  --config configs/gpr_demo/linnerud.yaml \
  --output-root data/outputs/ndis_covariance_correction/bell_py311/verified_final_after/gpr
```

Both commands wrote six records, with five successful private runs and no error records. They retained the checked-in splits, preprocessing, model hyperparameters, metrics, and seeds. The raw before/after paths and exact values are recorded in `covariance_calibration_before_after.csv`.

To reconcile the published GPR row, the additional corrected run was:

```bash
/tmp/python-venv/rp_benchmark_venv/bin/python scripts/run_gpr_ndis_demo.py \
  --config /tmp/linnerud-legacy-seeds.yaml \
  --output-root data/outputs/ndis_covariance_correction/bell_py311/legacy_seeds/verified_final_after/gpr
```

That temporary config copied `configs/gpr_demo/linnerud.yaml` and changed only `seeds` to `[42, 43, 44, 45, 46]` (plus its output root). It therefore preserves the published-run protocol without silently changing preprocessing, metrics, or model parameters.

Execution environment: Purdue Bell host `bell-a153.rcac.purdue.edu`, CPU-only AMD EPYC 7662 node; Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0, and pytest 9.1.1. No GPU was used.

## Reproducibility gaps and protocol resolution

1. The checked-in GPR config uses seeds `2734335219, 2905568749, 2703185305, 3085492084, 1659915345`, but the raw run behind the published GPR summary uses seeds `42, 43, 44, 45, 46`. The corrected current-config result is absolute error `2.3114286897 +/- 1.1523317203`; the table above uses a separate corrected rerun with legacy seeds `42-46` so it isolates the change to the published row.
2. There is no NDIS-summary generator analogous to the RP report builder. The two normal summary CSVs were regenerated from successful private JSONL records using mean, population standard deviation, minimum, maximum, and count. `gpr_summary.csv` follows the checked-in current config; the legacy-seed table comparison is retained separately in the before/after CSV and this report.
3. `requirements.txt` is unpinned and historical JSONL records do not store package or CPU versions. A controlled pre-fix rerun and the corrected rerun were therefore both performed on the same documented Bell Python 3.11 environment.
4. The submitted PDF writes the combined `delta_LC` bound in covariance-first order, while the existing repository function retains the different ordering already used by the current workflow. The work order specifically scoped the code change to `delta_bar_cov`, so `delta_bar_LC` was not changed; the reported numbers are the requested rerun of the existing generic-wrapper workflow with the hard-covariance correction.
