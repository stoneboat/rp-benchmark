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

## Label scaling

- Standardize labels: subtract training mean, divide by training std.
- Apply the same transformation to test labels.

## Row clipping (features)

- Compute L2 norms of training feature rows.
- Set C_X = 95th percentile of those norms.
- Remove any training row with ||x_i||_2 > C_X.
- Remove any test row with ||x_i||_2 > C_X.

## Label clipping

- Set C_Y = 95th percentile of |y_train|.
- Clip training and test labels to [-C_Y, C_Y].

## Row-norm bound for mechanisms

After preprocessing, the public row-norm bound `l` is computed as:

    l = max_i ||[x_i, y_i]||_2

over the training set. This is the bound used in MRP calibration
(Figure 1, Step 2 of the NDIS paper).

## Non-private reference

The non-private OLS baseline is computed on the same preprocessed
training data. Its test MSE is reported alongside the private results.
