"""Deterministic seed expansion."""

from __future__ import annotations

import hashlib


def expand_seed(base_seed: int, trial_index: int) -> int:
    """Produce a deterministic seed from a base seed and trial index."""
    h = hashlib.sha256(f"{base_seed}_{trial_index}".encode()).hexdigest()
    return int(h[:8], 16)
