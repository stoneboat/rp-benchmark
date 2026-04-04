"""Core demo runner: loops over mechanisms x epsilons x generated trial seeds."""

from __future__ import annotations

import inspect
import time
from typing import Any

import numpy as np

from rpbench.config import DemoConfig, PrivacySpec
from rpbench.datasets.autompg import AutoMPGAdapter
from rpbench.datasets.bike_sharing import BikeSharingAdapter

from rpbench.datasets.synthetic_redundant_regression import SyntheticRedundantRegressionAdapter
from rpbench.datasets.base import DatasetBundle
from rpbench.mechanisms.base import Mechanism
from rpbench.mechanisms.rp_ndis import MechRP, MechRPPois
from rpbench.mechanisms.baselines.blocki12_jl import Blocki12JL
from rpbench.metrics.release import relative_frobenius_xtx
from rpbench.tasks.ols_from_release import OLSFromRelease
from rpbench.utils.io import generate_run_id, save_jsonl
from rpbench.utils.linear_algebra import augmented_data, gram_matrix


MECHANISM_REGISTRY: dict[str, type] = {
    "Mech_RP": MechRP,
    "Mech_RP_Pois": MechRPPois,
    "Blocki12_JL": Blocki12JL,
}

DATASET_REGISTRY: dict[str, type] = {
    "autompg": AutoMPGAdapter,
    "bike_sharing": BikeSharingAdapter,
    "synthetic_redundant_regression": SyntheticRedundantRegressionAdapter,
}


def _adapter_init_kwargs(adapter_cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """Pass only kwargs accepted by the adapter ``__init__`` (ignores unknown keys)."""
    if not params:
        return {}
    sig = inspect.signature(adapter_cls.__init__)
    names = {p for p in sig.parameters if p != "self"}
    return {k: v for k, v in params.items() if k in names}


def _build_mechanism(name: str, params: dict[str, Any]) -> Mechanism:
    cls = MECHANISM_REGISTRY[name]
    return cls(**params)


def run_demo(cfg: DemoConfig) -> list[dict[str, Any]]:
    """Execute the full demo benchmark and return row-level records."""

    # Load dataset
    adapter_cls = DATASET_REGISTRY[cfg.dataset]
    adapter = adapter_cls(**_adapter_init_kwargs(adapter_cls, cfg.dataset_params))
    bundle = adapter.load(cfg.split, cfg.preprocess)
    pub_meta = adapter.public_meta(bundle)
    n_train = pub_meta["n"]
    delta = cfg.compute_delta(n_train)
    base_seed, trial_seeds = cfg.resolve_trial_seeds()

    # Precompute true Gram for release metric
    xtx_true = gram_matrix(bundle.X_train)

    # Augmented training data
    A_train = augmented_data(bundle.X_train, bundle.y_train)

    # Task
    task = OLSFromRelease()

    # Non-private baseline
    np_beta = np.linalg.lstsq(bundle.X_train, bundle.y_train, rcond=None)[0]
    np_pred = bundle.X_test @ np_beta
    np_mse = float(np.mean((bundle.y_test - np_pred) ** 2))

    records: list[dict[str, Any]] = []

    # Add non-private baseline record
    records.append({
        "run_id": generate_run_id(),
        "dataset": cfg.dataset,
        "mechanism": "NonPrivate",
        "task": cfg.task,
        "epsilon": None,
        "delta": None,
        "seed": None,
        "release_metrics": {"rel_fro_xtx": 0.0},
        "downstream_metrics": {"test_mse": np_mse},
        "runtime": {"runtime_sec": 0.0},
        "diagnostics": {},
        "provenance": {"n_train": n_train, "n_test": pub_meta.get("n_test", len(bundle.y_test)), "d": pub_meta["d"]},
    })

    total = len(cfg.mechanisms) * len(cfg.epsilon_grid) * len(trial_seeds)
    done = 0

    for mech_name in cfg.mechanisms:
        params = cfg.mech_params.get(mech_name, {})
        for epsilon in cfg.epsilon_grid:
            for seed_index, seed in enumerate(trial_seeds):
                done += 1
                print(
                    f"  [{done}/{total}] {mech_name} eps={epsilon} "
                    f"trial={seed_index} seed={seed}"
                )

                mech = _build_mechanism(mech_name, params)
                ps = PrivacySpec(epsilon=epsilon, delta=delta)
                mech.calibrate(ps, pub_meta)

                rb = mech.release(A_train, seed)

                # Release metric
                rel_fro = relative_frobenius_xtx(xtx_true, rb.xtx_hat)

                # Downstream
                beta_hat = task.fit_from_release(rb, pub_meta)
                ds_metrics = task.evaluate(beta_hat, bundle.X_test, bundle.y_test)

                records.append({
                    "run_id": generate_run_id(),
                    "dataset": cfg.dataset,
                    "mechanism": mech_name,
                    "task": cfg.task,
                    "epsilon": epsilon,
                    "delta": delta,
                    "seed": seed,
                    "seed_index": seed_index,
                    "release_metrics": {"rel_fro_xtx": rel_fro},
                    "downstream_metrics": ds_metrics,
                    "runtime": {"runtime_sec": rb.runtime_sec},
                    "diagnostics": rb.diagnostics,
                    "provenance": {
                        "n_train": n_train,
                        "d": pub_meta["d"],
                        "l": pub_meta["l"],
                        "seed_batch": {
                            "mode": cfg.seed_batch.mode,
                            "base_seed": base_seed,
                            "count": len(trial_seeds),
                        },
                        "calibration": rb.calibration,
                    },
                })

    return records
