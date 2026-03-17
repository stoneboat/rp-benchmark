"""Blocki12_JL — Johnson-Lindenstrauss transform baseline.

Ground truth: Blocki, Blum, Datta, Sheffet (2012), Algorithm 3
(arxiv 1204.2136).  Also referenced as [15] in the NDIS paper.

Applied to augmented data [X | y] so that both X^T X and X^T y
can be recovered from a single covariance release.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from rpbench.mechanisms.base import Mechanism, ReleaseBundle


class Blocki12JL(Mechanism):
    """JL-transform DP covariance estimation (Blocki et al. 2012, Alg. 3)."""

    name = "Blocki12_JL"

    def __init__(self, r: int):
        self.r = r
        self._calibrated = False
        self._cal: dict[str, Any] = {}

    def calibrate(self, privacy_spec, public_meta: dict) -> None:
        epsilon = privacy_spec.epsilon
        delta = privacy_spec.delta
        r = self.r
        if r <= 0:
            raise ValueError("Blocki12_JL calibration requires r > 0")

        w = 16.0 * math.sqrt(r * math.log(2.0 / delta)) * math.log(16.0 * r / delta) / epsilon

        self._cal = {
            "epsilon": epsilon,
            "delta": delta,
            "r": r,
            "w": w,
            "d_aug": public_meta["d_aug"],
        }
        self._calibrated = True

    def release(self, train_data: np.ndarray, seed: int) -> ReleaseBundle:
        assert self._calibrated, "Must call calibrate() first"
        t_start = time.time()

        rng = np.random.RandomState(seed)
        A = train_data.copy()  # (n, d_aug)
        n, d_aug = A.shape
        r = self._cal["r"]
        w = self._cal["w"]

        # Step 1: SVD on the input matrix as provided by the benchmark.
        U, S, Vt = np.linalg.svd(A, full_matrices=False)  # economy SVD

        # Step 2: spectral regularization — replace singular values
        S_reg = np.sqrt(S ** 2 + w ** 2)
        A_reg = U * S_reg[np.newaxis, :] @ Vt  # (n, d_aug)

        # Step 3: random projection
        M = rng.standard_normal((r, n))  # (r, n)
        MA = M @ A_reg                   # (r, d_aug)

        # Step 4: covariance estimate
        C_tilde = (MA.T @ MA) / r        # (d_aug, d_aug)

        d = d_aug - 1
        xtx_hat = C_tilde[:d, :d] - w ** 2 * np.eye(d)
        xty_hat = C_tilde[:d, d]

        runtime = time.time() - t_start

        return ReleaseBundle(
            mechanism_name=self.name,
            release_kind="gram_blocks",
            xtx_hat=xtx_hat,
            xty_hat=xty_hat,
            sketch_matrix=MA,
            calibration=dict(self._cal),
            diagnostics={"w": w, "r": r},
            runtime_sec=runtime,
        )

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._cal)
