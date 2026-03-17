"""Configuration dataclasses and YAML loader for rpbench."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
    clip_x_norm: float | None = None  # auto-determined if None
    clip_y_abs: float | None = None   # auto-determined if None
    missing_policy: str = "drop"


@dataclass
class MechParams:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DemoConfig:
    dataset: str
    mechanisms: list[str]
    task: str
    epsilon_grid: list[float]
    delta_rule: str  # e.g. "1/n^2"
    seeds: list[int]
    split: SplitSpec
    preprocess: PreprocessSpec
    mech_params: dict[str, dict[str, Any]]
    output_root: str = "data/outputs/runs"

    def compute_delta(self, n_train: int) -> float:
        if self.delta_rule == "1/n^2":
            return 1.0 / (n_train ** 2)
        return float(self.delta_rule)


def load_config(path: str | Path) -> DemoConfig:
    """Load a DemoConfig from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    split = SplitSpec(**raw.get("split", {}))
    preprocess = PreprocessSpec(**raw.get("preprocess", {}))

    return DemoConfig(
        dataset=raw["dataset"],
        mechanisms=raw["mechanisms"],
        task=raw["task"],
        epsilon_grid=raw["epsilon_grid"],
        delta_rule=raw["delta_rule"],
        seeds=raw["seeds"],
        split=split,
        preprocess=preprocess,
        mech_params=raw.get("mech_params", {}),
        output_root=raw.get("output_root", "data/outputs/runs"),
    )
