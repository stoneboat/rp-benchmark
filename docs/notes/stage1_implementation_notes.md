# Stage-1 Implementation Notes

This document logs design choices and simplifications made in the
stage-1 implementation of WF-001.

## Mech_RP

- **Ground truth**: NDIS paper (arxiv 2309.01243), Figure 1.
- **Sketch dimension**: `r` is provided explicitly by benchmark config so it can
  be matched exactly against baselines.
- **Code provenance**: Core privacy-curve functions (`_compute_gamma_delta`,
  `compute_IS`) and the calibration binary search (`compute_leverage_upper_bound`)
  are copied directly from the old NDIS repo (`NDIS/src/analysis/
  RP_privacy_analysis_advanced.py` and `NDIS/src/RP_mechanisms/optim_RP_DP.py`).
  They are **not** imported as a submodule — the code is inlined in
  `src/rpbench/mechanisms/rp_ndis.py`.
- **Augmented-data convention**: For the OLS benchmark, `D = [X | y]`
  so the mechanism's sketch covers both the feature Gram and the
  feature-label cross-term. This follows the old repo's
  `LS_fromoptim_RP_mech` pattern.
- **Gram estimator**: We use Theorem 4's unbiased formula
  `Sigma_hat = (1/r) M_tilde M_tilde^T - lambda I` rather than solving
  OLS directly on the sketch. Both approaches are mathematically
  equivalent for a single sketch; the Gram route is chosen because the
  release metric requires `xtx_hat` explicitly.
- **Single-trial per expanded seed**: The old repo's `OptimalRP_mech`
  generates `num_samples` sketches in one call. For the benchmark, each
  `(mechanism, epsilon, expanded_seed)` trial produces exactly one sketch
  realization. Expanded seeds are generated deterministically from one
  batch root seed.

## Blocki12_JL

- **Ground truth**: Blocki et al. 2012 (arxiv 1204.2136), Algorithm 3.
- **Sketch dimension**: `r` is provided explicitly by benchmark config so both
  mechanisms run at the same projection dimension.
- **Shared input matrix**: The benchmark passes the same augmented training
  matrix `D = [X | y]` directly to both mechanisms; `Blocki12_JL` does not apply
  an extra internal centering step.
- **Augmented data**: Same convention as Mech_RP — apply to `[X | y]`.

## Preprocessing

- Row clipping at 95th percentile is a pragmatic choice; a stricter
  public bound can be substituted later.
- Label standardization is applied for numerical stability.

## Output schema

- JSONL records with fields: run_id, dataset, mechanism, task,
  epsilon, delta, seed, release_metrics, downstream_metrics,
  runtime, diagnostics, provenance.
