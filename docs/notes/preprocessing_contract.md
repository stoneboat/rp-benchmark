# Preprocessing Contract — WF-001 Stage 1

This document defines the shared preprocessing recipe applied to every
dataset before it is passed to any mechanism. Both mechanisms see the
same preprocessed data.

## Missing values

Drop any row with a missing value in features or target.

## Feature / target split

- Features: all continuous numeric columns defined by the dataset adapter.
- Target: the regression label (e.g., `mpg` for AutoMPG).

## Train / test split

- 80% train, 20% test.
- Fixed seed = 42, shuffled.
- Split happens before any scaling or clipping.

## Feature scaling

- Fit `sklearn.preprocessing.StandardScaler` on training features only.
- Transform both train and test features.

## Feature clipping

- After scaling, clip each feature row to a public L2 bound `C_X`.
- Clipping is done by rescaling, not dropping:

      x_i <- x_i * min(1, C_X / ||x_i||_2)

- The same public `C_X` is applied to both train and test features.

## Label scaling

- Standardize labels: subtract training mean, divide by training std.
- Apply the same transformation to test labels.

## Label clipping

- After standardization, clip labels to a public bound `C_Y`.
- Clip training and test labels to [-C_Y, C_Y].

## Row-norm bound for mechanisms

After preprocessing, the public row-norm bound `l` is set from the public
feature/label clipping bounds:

    l = sqrt(C_X^2 + C_Y^2)

This is the bound used in MRP calibration
(Figure 1, Step 2 of the NDIS paper).

## Non-private reference

The non-private OLS baseline is computed on the same preprocessed
training data. Its test MSE is reported alongside the private results.
