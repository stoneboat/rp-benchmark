"""Configuration dataclasses and YAML loader for rpbench."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rpbench.utils.seeding import choose_base_seed, generate_trial_seeds


@dataclass
class PrivacySpec:
    epsilon: float
    delta: float
    adjacency: str = "add_remove"


@dataclass
class SplitSpec:
    train_fraction: float = 0.8
    seed: int = 42


@dataclass
class PreprocessSpec:
    scale_x: bool = True
    clip_x: bool = True
    clip_y: bool = True
    clip_x_bound: float = 3.0
    clip_y_bound: float = 3.0
    missing_policy: str = "drop"


@dataclass
class SeedBatchSpec:
    mode: str = "fixed"
    base_seed: int | None = 0
    count: int = 5


@dataclass
class DemoConfig:
    dataset: str
    mechanisms: list[str]
    task: str
    epsilon_grid: list[float]
    delta_rule: str  # e.g. "1/n^2"
    seed_batch: SeedBatchSpec
    split: SplitSpec
    preprocess: PreprocessSpec
    mech_params: dict[str, dict[str, Any]]
    output_root: str = "data/outputs/runs"
    dataset_params: dict[str, Any] = field(default_factory=dict)
    _resolved_base_seed: int | None = field(default=None, init=False, repr=False)
    _resolved_trial_seeds: list[int] | None = field(default=None, init=False, repr=False)

    def compute_delta(self, n_train: int) -> float:
        if self.delta_rule == "1/n^2":
            return 1.0 / (n_train ** 2)
        return float(self.delta_rule)

    def resolve_trial_seeds(self) -> tuple[int, list[int]]:
        if self._resolved_base_seed is None or self._resolved_trial_seeds is None:
            self._resolved_base_seed = choose_base_seed(
                self.seed_batch.mode, self.seed_batch.base_seed
            )
            self._resolved_trial_seeds = generate_trial_seeds(
                self._resolved_base_seed, self.seed_batch.count
            )
        return self._resolved_base_seed, list(self._resolved_trial_seeds)


def load_config(path: str | Path) -> DemoConfig:
    """Load a DemoConfig from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    split = SplitSpec(**raw.get("split", {}))
    preprocess = PreprocessSpec(**raw.get("preprocess", {}))
    seed_batch_raw = raw.get("seed_batch")
    if seed_batch_raw is None:
        legacy_seeds = raw.get("seeds")
        if legacy_seeds is None:
            raise ValueError("Config requires either seed_batch or legacy seeds")
        if len(legacy_seeds) == 0:
            raise ValueError("Legacy seeds list must be non-empty")
        seed_batch = SeedBatchSpec(
            mode="fixed",
            base_seed=int(legacy_seeds[0]),
            count=len(legacy_seeds),
        )
    else:
        seed_batch = SeedBatchSpec(**seed_batch_raw)

    return DemoConfig(
        dataset=raw["dataset"],
        mechanisms=raw["mechanisms"],
        task=raw["task"],
        epsilon_grid=raw["epsilon_grid"],
        delta_rule=raw["delta_rule"],
        seed_batch=seed_batch,
        split=split,
        preprocess=preprocess,
        mech_params=raw.get("mech_params", {}),
        output_root=raw.get("output_root", "data/outputs/runs"),
        dataset_params=raw.get("dataset_params") or {},
    )
