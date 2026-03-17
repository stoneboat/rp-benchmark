"""Deterministic seed expansion."""

from __future__ import annotations

import secrets

import numpy as np


def expand_seed(base_seed: int, trial_index: int) -> int:
    """Produce a deterministic seed from a base seed and trial index."""
    ss = np.random.SeedSequence([int(base_seed), int(trial_index)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def choose_base_seed(mode: str, base_seed: int | None = None) -> int:
    """Resolve the root seed for a batch of trials."""
    if mode == "fixed":
        if base_seed is None:
            raise ValueError("fixed seed mode requires base_seed")
        return int(base_seed)
    if mode == "random":
        return secrets.randbits(32)
    raise ValueError(f"Unknown seed mode: {mode}")


def generate_trial_seeds(base_seed: int, count: int) -> list[int]:
    """Generate a deterministic batch of per-trial seeds from one root seed."""
    if count <= 0:
        raise ValueError("seed count must be positive")
    return [expand_seed(base_seed, idx) for idx in range(count)]
