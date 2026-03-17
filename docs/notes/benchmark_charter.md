# Benchmark Charter — WF-001 Phase-1 RP Benchmark

**Status:** frozen for stage 1

## Adjacency

Add/remove row (unbounded DP).

## Privacy notion

(epsilon, delta)-DP.

## Epsilon grid (demo)

{0.5, 1.0, 2.0, 4.0}

## Delta rule

delta = 1 / n_train^2

## Mechanisms

| ID | Role | Stage-1 status |
|---|---|---|
| Mech_RP | Primary project mechanism | mandatory |
| Blocki12_JL | Historical RP/JL baseline | mandatory |

## Dataset

AutoMPG (UCI, data_id=196) — single dataset for stage 1.

## Release metric

Relative Frobenius error of the released Gram estimate:
`||X^T X - xtx_hat||_F / ||X^T X||_F`

## Downstream task

OLSFromRelease — OLS fitted from released X^T X and X^T y.

## Downstream metric

Test MSE.

## Preprocessing

Shared across all mechanisms. See `preprocessing_contract.md`.

## Seeds

5 seeds (0-4) for the demo; 20+ for final figures.

## Split

80/20 train/test with fixed seed 42.

## Report

One epsilon-vs-test-MSE figure, one CSV summary, one markdown summary.
