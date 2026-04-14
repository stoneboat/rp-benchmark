#!/usr/bin/env python
"""Standalone BLR + NDIS-Calibrated Gaussian Mechanism demo.

Pipeline (Figure 5 of the NDIS paper):
  1. Load Breast Cancer dataset (BreastCancerAdapter).
  2. Fit BLR surrogate (BLRGaussianOutput) -> (theta_hat, Sigma).
  3. Calibrate NDISGaussianWrapper -> tau* via binary search.
  4. Release theta_tilde ~ N(theta_hat, Sigma + tau* * I_d).
  5. Evaluate accuracy and log-loss on held-out test set.
  6. Save per-run records to JSONL.

Usage
-----
  python scripts/run_blr_ndis_demo.py --config configs/blr_demo/breast_cancer.yaml
  python scripts/run_blr_ndis_demo.py --config configs/blr_demo/breast_cancer.yaml \
      --output-root data/outputs/blr_ndis_runs

This script intentionally avoids rpbench.runners, rpbench.mechanisms, and
rpbench.tasks — it is a self-contained demo for the BLR/NDIS stack.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable when run from the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from rpbench.config import SplitSpec, PreprocessSpec
from rpbench.datasets.breast_cancer import BreastCancerAdapter

from ndis_gaussian import (
    BLRGaussianOutput,
    NDISGaussianWrapper,
)


# ---------------------------------------------------------------------------
# Config loading (lightweight; not using rpbench.config.DemoConfig which is
# designed for the RP-mechanism pipeline).
# ---------------------------------------------------------------------------

def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_delta(delta_rule: str, n_train: int) -> float:
    """Convert delta_rule string to a float."""
    if delta_rule == "1/n^2":
        return 1.0 / (n_train ** 2)
    return float(delta_rule)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate(theta: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Compute accuracy and logistic log-loss on the test set.

    Parameters
    ----------
    theta : np.ndarray, shape (d,)
        Logistic regression parameter vector.
    X_test : np.ndarray, shape (n_test, d)
    y_test : np.ndarray, shape (n_test,), values in {-1, +1}

    Returns
    -------
    dict with keys: accuracy, log_loss, n_test
    """
    margins = y_test * (X_test @ theta)          # shape (n_test,)
    # Log-loss: mean(log(1 + exp(-margin)))
    log_loss = float(np.mean(np.logaddexp(0.0, -margins)))
    # Accuracy: fraction where sign(theta^T x) == y, i.e. margin > 0
    accuracy = float(np.mean(margins > 0.0))
    return {
        "accuracy": accuracy,
        "log_loss": log_loss,
        "n_test": int(len(y_test)),
    }


# ---------------------------------------------------------------------------
# Main per-run function
# ---------------------------------------------------------------------------

def _run_single(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    public_meta: dict[str, Any],
    epsilon: float,
    delta: float,
    lambda_reg: float,
    seed: int,
) -> dict[str, Any]:
    """Fit BLR, calibrate NDIS wrapper, release, evaluate."""
    t0 = time.perf_counter()

    # Step 1: Fit BLR (Definition 7)
    blr = BLRGaussianOutput(lambda_reg=lambda_reg)
    gaussian_output = blr.fit(X_train, y_train)

    # Step 2: Calibrate NDIS wrapper (Figure 5, Step 1)
    wrapper = NDISGaussianWrapper()
    wrapper.calibrate(blr, epsilon=epsilon, delta=delta, public_meta=public_meta)
    tau_star = wrapper.tau_star
    sigma_std = wrapper.sigma_std

    # Step 3: Release theta_tilde (Figure 5, Step 2)
    wrapped = wrapper.release(gaussian_output, seed=seed)
    theta_tilde = wrapped.theta_tilde

    t1 = time.perf_counter()

    # Step 4: Evaluate both non-private (theta_hat) and private (theta_tilde)
    metrics_hat = _evaluate(gaussian_output.mu, X_test, y_test)
    metrics_tilde = _evaluate(theta_tilde, X_test, y_test)

    sens = wrapped.sensitivity

    record: dict[str, Any] = {
        "mechanism": "BLR_NDIS",
        "dataset": "breast_cancer",
        "epsilon": epsilon,
        "delta": delta,
        "lambda_reg": lambda_reg,
        "seed": seed,
        "privacy": {
            "tau_star": tau_star,
            "sigma_std": sigma_std,
            "epsilon": epsilon,
            "delta": delta,
        },
        "sensitivity": {
            "Delta": sens.Delta,
            "rho_inf": sens.rho_inf,
            "nu": sens.nu,
            "tau_at_calibration": sens.tau,
        },
        "public_meta": {k: v for k, v in public_meta.items()},
        "metrics_nonprivate": metrics_hat,
        "metrics_private": metrics_tilde,
        "wall_time_s": t1 - t0,
    }
    return record


def _run_nonprivate(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambda_reg: float,
    public_meta: dict[str, Any],
) -> dict[str, Any]:
    """Fit BLR without privacy noise (baseline reference)."""
    t0 = time.perf_counter()
    blr = BLRGaussianOutput(lambda_reg=lambda_reg)
    gaussian_output = blr.fit(X_train, y_train)
    t1 = time.perf_counter()

    metrics = _evaluate(gaussian_output.mu, X_test, y_test)
    return {
        "mechanism": "NonPrivate",
        "dataset": "breast_cancer",
        "epsilon": None,
        "delta": None,
        "lambda_reg": lambda_reg,
        "seed": None,
        "public_meta": {k: v for k, v in public_meta.items()},
        "metrics_nonprivate": metrics,
        "metrics_private": metrics,  # same — no noise added
        "wall_time_s": t1 - t0,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BLR + NDIS-Calibrated Gaussian Mechanism demo")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/blr_demo/breast_cancer.yaml)",
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="Override output_root from config",
    )
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config)
    output_root = args.output_root or cfg.get("output_root", "data/outputs/blr_ndis_runs")

    epsilon_grid: list[float] = [float(e) for e in cfg["epsilon_grid"]]
    delta_rule: str = str(cfg["delta_rule"])
    lambda_reg: float = float(cfg["lambda_reg"])
    seeds: list[int] = [int(s) for s in cfg["seeds"]]

    split_raw = cfg.get("split", {})
    split_spec = SplitSpec(
        train_fraction=float(split_raw.get("train_fraction", 0.8)),
        seed=int(split_raw.get("seed", 42)),
    )

    pre_raw = cfg.get("preprocess", {})
    preprocess_spec = PreprocessSpec(
        scale_x=bool(pre_raw.get("scale_x", True)),
        clip_x=bool(pre_raw.get("clip_x", True)),
        clip_x_bound=float(pre_raw.get("clip_x_bound", 3.0)),
        clip_y=False,   # ignored for classification
        clip_y_bound=3.0,
        missing_policy=str(pre_raw.get("missing_policy", "drop")),
    )

    # Load dataset once (no randomness from multiple seeds here — the split
    # seed is fixed in split_spec)
    print(f"Loading dataset: {cfg.get('dataset', 'breast_cancer')} ...")
    adapter = BreastCancerAdapter()
    bundle = adapter.load(split_spec, preprocess_spec)
    public_meta = adapter.public_meta(bundle)

    n_train = bundle.meta["n_train"]
    delta = _parse_delta(delta_rule, n_train)

    print(f"  n_train={n_train}, n_test={bundle.meta['n_test']}, d={bundle.meta['d']}")
    print(f"  C_X={bundle.meta['C_X']:.4f}, lambda_reg={lambda_reg}")
    print(f"  epsilon_grid={epsilon_grid}, delta={delta:.2e}")
    print(f"  seeds={seeds}\n")

    records: list[dict[str, Any]] = []

    # Non-private baseline
    rec_np = _run_nonprivate(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        lambda_reg=lambda_reg,
        public_meta=public_meta,
    )
    records.append(rec_np)
    print(
        f"[NonPrivate] accuracy={rec_np['metrics_private']['accuracy']:.4f}, "
        f"log_loss={rec_np['metrics_private']['log_loss']:.4f}"
    )

    # Private runs
    for epsilon in epsilon_grid:
        for seed in seeds:
            try:
                rec = _run_single(
                    X_train=bundle.X_train,
                    y_train=bundle.y_train,
                    X_test=bundle.X_test,
                    y_test=bundle.y_test,
                    public_meta=public_meta,
                    epsilon=epsilon,
                    delta=delta,
                    lambda_reg=lambda_reg,
                    seed=seed,
                )
                records.append(rec)
                print(
                    f"[eps={epsilon:.1f}, seed={seed}] "
                    f"tau*={rec['privacy']['tau_star']:.4e}, "
                    f"sigma={rec['privacy']['sigma_std']:.4e}, "
                    f"acc={rec['metrics_private']['accuracy']:.4f}, "
                    f"loss={rec['metrics_private']['log_loss']:.4f}"
                )
            except Exception as exc:
                print(f"[eps={epsilon:.1f}, seed={seed}] ERROR: {exc}")
                records.append({
                    "mechanism": "BLR_NDIS",
                    "dataset": "breast_cancer",
                    "epsilon": epsilon,
                    "delta": delta,
                    "lambda_reg": lambda_reg,
                    "seed": seed,
                    "error": str(exc),
                })

    # Save JSONL
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"blr_ndis_results_{ts}.jsonl"

    def _default_serializer(obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_default_serializer) + "\n")

    print(f"\nSaved {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
