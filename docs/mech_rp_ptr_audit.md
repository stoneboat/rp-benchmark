# Mech_RP_PTR Audit

Date: 2026-04-05

## Implementation Location

`Mech_RP_PTR` is implemented as `MechRPPTR` in:

- `src/rpbench/mechanisms/rp_ndis.py`

Key locations:

- class definition: `src/rpbench/mechanisms/rp_ndis.py:373`
- calibration logic: `src/rpbench/mechanisms/rp_ndis.py:419`
- release logic: `src/rpbench/mechanisms/rp_ndis.py:480`

Related wiring:

- mechanism registry: `src/rpbench/runners/run_demo.py:25`
- demo config: `configs/demo/autompg_ptr.yaml`
- release-bundle tests: `tests/test_release_bundle.py:124`
- end-to-end smoke test: `tests/test_smoke.py:87`

## Audit Scope

This audit is a static code audit of the `Mech_RP_PTR` implementation and its
integration points in the current repo. It focuses on:

- mechanism wiring
- calibration and privacy-budget splitting
- release construction
- diagnostics and tests

It does not independently re-derive the paper proof or empirically validate the
privacy guarantee.

## Summary

`Mech_RP_PTR` is implemented as a wrapper around the RP mechanism described in
the NDIS paper's PTR construction. The code follows the intended high-level
structure:

1. split the total privacy budget into RP release, eigenvalue test, and PTR
   failure budgets
2. calibrate the Gaussian eigenvalue test to obtain `epsilon_T`
3. spend the remaining privacy budget `epsilon_R` on the RP release
4. privately lower-bound `lambda_min(D^T D)`
5. reduce the ridge level adaptively when the lower bound is positive
6. generate an RP release using the reduced ridge

From a code-structure perspective, the implementation is coherent and matches
the surrounding mechanism framework well.

## Findings

No clear correctness bug was identified in the current `Mech_RP_PTR`
implementation from static inspection alone.

## Detailed Notes

### 1. Mechanism registration and invocation

`Mech_RP_PTR` is exposed through the main benchmark runner via
`MECHANISM_REGISTRY["Mech_RP_PTR"]` in `src/rpbench/runners/run_demo.py:29`.
That means any demo config can invoke it through the standard `run_demo.py`
path without special-case logic.

The demo config `configs/demo/autompg_ptr.yaml` sets:

- dataset: `autompg`
- mechanisms: `Mech_RP`, `Mech_RP_PTR`
- a delta split with `delta_r + delta_t + delta_ptr = delta`

This is consistent with the implementation contract.

### 2. Calibration logic

In `calibrate()`:

- input validation is explicit for `r`, `epsilon`, `delta`, `l`, `tau`
- the three delta pieces must be positive
- the delta split must match the outer `delta` within tolerance `1e-10`

Then the implementation computes:

- `epsilon_T` from the Gaussian eigenvalue test via `_find_epsilon_T(...)`
- `epsilon_R = max(epsilon - epsilon_T, 0.0)`
- `p_star` for the RP release using `compute_leverage_upper_bound(...)`
- the baseline RP ridge `lambda_rp = l^2 / p_star`
- `alpha = tau * Phi^{-1}(1 - delta_ptr)`

This decomposition is internally consistent with the intended PTR flow.

### 3. Release logic

In `release()`:

- the raw minimum eigenvalue of `D^T D` is computed
- Gaussian noise `eta ~ N(0, tau^2)` is added
- the private lower bound is `lambda_lb = max(lambda_min_raw + eta - alpha, 0)`
- the PTR ridge is `lambda_ptr = max(l^2 / p_star - lambda_lb, 0)`

Then the mechanism uses the same RP sketch formula as `Mech_RP`, but replaces
the baseline ridge with `lambda_ptr`.

This reuse is clean and makes the PTR mechanism easy to compare against the
baseline RP mechanism.

### 4. Diagnostics

The release returns a reasonably complete diagnostics block, including:

- `epsilon_T`, `epsilon_R`
- `delta_R`, `delta_T`, `delta_ptr`
- `tau`, `p_star`
- `lambda_min_raw`, `eta`, `lambda_lb`
- `lambda_ptr`, `lambda_rp`
- `prop5_condition_met`

This is useful for debugging and for interpreting whether PTR actually reduced
the ridge on a given run.

### 5. Test coverage

The implementation has both unit-style and integration-style coverage:

- `tests/test_release_bundle.py:124`
  checks release-bundle structure, output shapes, and required diagnostics
- `tests/test_release_bundle.py:159`
  checks deterministic behavior under the same seed
- `tests/test_smoke.py:87`
  runs an end-to-end benchmark path with `Mech_RP_PTR`

That is adequate smoke coverage for framework integration.

## Residual Risks

These are not confirmed bugs, but they are the main audit caveats:

- The audit did not numerically verify the privacy accounting against the paper.
- The code uses a strict delta-sum tolerance; this is reasonable, but YAML
  float formatting could make user configs fragile if they are not chosen
  carefully.
- There is no dedicated regression test for invalid delta splits or extreme
  `tau` values.
- The release step computes `eigvalsh(D^T D)` directly each run; this is fine
  for small demos, but may become expensive on larger datasets.

## Recommendation

Keep the current implementation. It is wired correctly, documented in the
config, and has basic release/integration tests.

If you want stronger assurance, the next useful additions would be:

1. a negative test that rejects mismatched `delta_r + delta_t + delta_ptr`
2. a calibration test that checks PTR diagnostics against simple invariants
3. a benchmark note showing when `lambda_ptr < lambda_rp` actually occurs in
   practice
