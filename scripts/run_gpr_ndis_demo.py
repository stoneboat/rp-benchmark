#!/usr/bin/env python
"""Standalone GPR + NDIS-Calibrated Gaussian Mechanism demo.

Implements a single-query Gaussian Process Regression (Definition 8)
wrapped with the NDIS-Calibrated Gaussian Mechanism (Figure 5).

Pipeline:
  1. Load the Linnerud scalar-target regression demo dataset.
  2. Choose a fixed public query point x_★ from the test split.
  3. Fit GPRGaussianOutput (Definition 8) → scalar (μ_★, Σ_★).
  4. Calibrate NDISGaussianWrapper → τ* (binary search, Proposition 8).
  5. Release θ_tilde ~ N(μ_★, Σ_★ + τ*) for multiple seeds.
  6. Compare non-private prediction, private release, and true label y_★.
  7. Save per-run records to JSONL.

This script:
  - Does NOT use rpbench.runners, rpbench.mechanisms, or rpbench.tasks.
  - Does NOT modify the existing run_demo.py pipeline.
  - GPR output is SCALAR (m=1): single prediction at a public query point.

Usage
-----
  python scripts/run_gpr_ndis_demo.py --config configs/gpr_demo/linnerud.yaml
  python scripts/run_gpr_ndis_demo.py --config configs/gpr_demo/linnerud.yaml \\
      --output-root data/outputs/gpr_ndis_runs

Privacy-utility note
--------------------
Under add/remove DP, the updated GPR sensitivity is
Δ(τ) = l·B·√n/(σ_n·√τ). This still diverges at τ=0, but it is much tighter than
the earlier n-scaling bound. The Linnerud demo keeps a scalar target, clips it to
the configured public B bound, and uses a single fixed public query point.
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
from rpbench.datasets.linnerud import LinnerudAdapter

from ndis_gaussian import (
    GPRGaussianOutput,
    NDISGaussianWrapper,
    RBFKernel,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_delta(delta_rule: str, n_train: int) -> float:
    if delta_rule == "1/n^2":
        return 1.0 / (n_train ** 2)
    return float(delta_rule)


def _load_dataset_from_config(cfg: dict[str, Any], split_spec: SplitSpec):
    """Load a supported standalone GPR demo dataset."""
    dataset_name = str(cfg.get("dataset", "linnerud"))
    dataset_params = cfg.get("dataset_params") or {}

    if dataset_name == "linnerud":
        target_index = int(dataset_params.get("target_index", cfg.get("target_index", 0)))
        adapter = LinnerudAdapter(target_index=target_index)
        pre_raw = cfg.get("preprocess", {})
        preprocess_spec = PreprocessSpec(
            scale_x=bool(pre_raw.get("scale_x", True)),
            clip_x=bool(pre_raw.get("clip_x", True)),
            clip_y=bool(pre_raw.get("clip_y", True)),
            clip_x_bound=float(pre_raw.get("clip_x_bound", 3.0)),
            clip_y_bound=float(pre_raw.get("clip_y_bound", 1.0)),
            missing_policy=str(pre_raw.get("missing_policy", "drop")),
        )
        bundle = adapter.load(split_spec, preprocess_spec)
    else:
        raise ValueError(
            f"Unsupported GPR demo dataset {dataset_name!r}. "
            "Supported dataset: 'linnerud'."
        )

    return adapter, bundle


# ---------------------------------------------------------------------------
# Main run functions
# ---------------------------------------------------------------------------

def _run_private_release(
    *,
    gaussian_output: Any,
    wrapper: NDISGaussianWrapper,
    dataset_name: str,
    y_star: float,
    x_star_index: int,
    epsilon: float,
    delta: float,
    sigma_n2: float,
    seed: int,
    calibration_wall_time_s: float,
) -> dict[str, Any]:
    """Sample one private release and evaluate vs. true label."""
    t0 = time.perf_counter()
    release = wrapper.release(gaussian_output, seed=seed)
    t1 = time.perf_counter()

    theta_tilde = float(release.theta_tilde[0])  # scalar release
    mu_star = float(gaussian_output.mu[0])
    Sigma_star = float(gaussian_output.Sigma[0, 0])
    sens = release.sensitivity

    return {
        "mechanism": "GPR_NDIS",
        "dataset": dataset_name,
        "x_star_index": x_star_index,
        "epsilon": epsilon,
        "delta": delta,
        "sigma_n2": sigma_n2,
        "seed": seed,
        "privacy": {
            "tau_star": wrapper.tau_star,
            "sigma_std": wrapper.sigma_std,
        },
        "sensitivity": {
            "Delta": sens.Delta,
            "rho_inf": sens.rho_inf,
            "nu": sens.nu,
            "tau_at_calibration": sens.tau,
        },
        "nonprivate": {
            "mu_star": mu_star,
            "Sigma_star": Sigma_star,
        },
        "private": {
            "theta_tilde": theta_tilde,
        },
        "true_label": y_star,
        "error_nonprivate": abs(mu_star - y_star),
        "error_private": abs(theta_tilde - y_star),
        "wall_time_s": t1 - t0,
        "calibration_wall_time_s": calibration_wall_time_s,
    }


def _run_nonprivate(
    *,
    gaussian_output: Any,
    dataset_name: str,
    y_star: float,
    x_star_index: int,
    sigma_n2: float,
    fit_wall_time_s: float,
) -> dict[str, Any]:
    """Non-private GPR prediction (baseline reference)."""
    mu_star = float(gaussian_output.mu[0])
    Sigma_star = float(gaussian_output.Sigma[0, 0])
    return {
        "mechanism": "NonPrivate",
        "dataset": dataset_name,
        "x_star_index": x_star_index,
        "epsilon": None,
        "delta": None,
        "sigma_n2": sigma_n2,
        "seed": None,
        "nonprivate": {
            "mu_star": mu_star,
            "Sigma_star": Sigma_star,
        },
        "private": None,
        "true_label": y_star,
        "error_nonprivate": abs(mu_star - y_star),
        "wall_time_s": fit_wall_time_s,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPR + NDIS-Calibrated Gaussian Mechanism demo (single-query)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (e.g. configs/gpr_demo/linnerud.yaml)",
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="Override output_root from config",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    output_root = args.output_root or cfg.get("output_root", "data/outputs/gpr_ndis_runs")

    epsilon = float(cfg["epsilon"])
    delta_rule = str(cfg["delta_rule"])
    sigma_n2 = float(cfg["sigma_n2"])
    lengthscale = float(cfg.get("lengthscale", 1.0))
    l_bound = float(cfg.get("l_bound", 1.0))
    B_bound = float(cfg.get("B_bound", 1.0))
    x_star_index = int(cfg.get("x_star_index", 0))
    seeds = [int(s) for s in cfg["seeds"]]

    split_raw = cfg.get("split", {})
    split_spec = SplitSpec(
        train_fraction=float(split_raw.get("train_fraction", 0.8)),
        seed=int(split_raw.get("seed", 42)),
    )

    print(f"Loading dataset: {cfg.get('dataset', 'linnerud')} ...")
    adapter, bundle = _load_dataset_from_config(cfg, split_spec)
    dataset_name = str(bundle.meta.get("dataset", cfg.get("dataset", "linnerud")))

    n_train = bundle.meta["n_train"]
    n_test = bundle.meta["n_test"]
    d_feat = bundle.meta["d"]
    C_X = bundle.meta["C_X"]
    C_Y = bundle.meta["C_Y"]
    delta = _parse_delta(delta_rule, n_train)
    if B_bound + 1e-12 < C_Y:
        raise ValueError(
            f"B_bound={B_bound} is smaller than dataset target bound C_Y={C_Y}. "
            "Increase B_bound or adjust preprocessing."
        )
    if l_bound + 1e-12 < 1.0:
        raise ValueError(
            f"l_bound={l_bound} is invalid for the unit-amplitude RBF kernel; "
            "use l_bound >= 1.0 because k(x,x)=1."
        )

    print(f"  n_train={n_train}, n_test={n_test}, d_feat={d_feat}")
    print(f"  C_X={C_X:.4f} (max row norm), C_Y={C_Y:.4f} (max |y_train|)")
    if "target_name" in bundle.meta:
        print(
            f"  target={bundle.meta['target_name']} "
            f"(target_index={bundle.meta.get('target_index')})"
        )
    print(f"  B_bound={B_bound}, l_bound={l_bound} (RBF: k(x,x)=1)")
    print(f"  epsilon={epsilon}, delta={delta:.2e}")
    print(f"  sigma_n2={sigma_n2}, lengthscale={lengthscale}, seeds={seeds}\n")

    # Public query point from test split (must be chosen before seeing test labels)
    if x_star_index >= n_test:
        raise ValueError(
            f"x_star_index={x_star_index} out of range for n_test={n_test}"
        )
    x_star = bundle.X_test[x_star_index]
    y_star = float(bundle.y_test[x_star_index])

    print(f"Public query point: x_star = X_test[{x_star_index}], true y = {y_star:.4f}\n")

    # Build GPR Gaussian-output algorithm
    kernel = RBFKernel(lengthscale=lengthscale)
    gpr = GPRGaussianOutput(
        x_star=x_star,
        kernel=kernel,
        sigma_n2=sigma_n2,
        B_bound=B_bound,
        l_bound=l_bound,
    )

    # Public meta for sensitivity computation
    # NOTE: "d" = 1 here is the OUTPUT dimension (scalar GPR), NOT the feature dim.
    public_meta: dict[str, Any] = {
        "n": n_train,
        "d": 1,           # scalar output dimension (m=1 per Def. 8)
        "B": B_bound,
        "l": l_bound,
        "sigma_n2": sigma_n2,
    }

    # Fit GPR (Definition 8): compute μ_★, Σ_★ on training data
    print("Fitting GPR ...")
    t_fit = time.perf_counter()
    gaussian_output = gpr.fit(bundle.X_train, bundle.y_train)
    fit_wall_time_s = time.perf_counter() - t_fit

    mu_star = float(gaussian_output.mu[0])
    Sigma_star = float(gaussian_output.Sigma[0, 0])
    print(f"  Non-private: μ_★ = {mu_star:.4f}, Σ_★ = {Sigma_star:.6f}")
    print(f"  True label:  y_★ = {y_star:.4f}")
    print(f"  |μ_★ - y_★| = {abs(mu_star - y_star):.4f}")
    print(f"  Fit time: {fit_wall_time_s:.3f}s\n")

    # Non-private record
    records: list[dict[str, Any]] = []
    rec_np = _run_nonprivate(
        gaussian_output=gaussian_output,
        dataset_name=dataset_name,
        y_star=y_star,
        x_star_index=x_star_index,
        sigma_n2=sigma_n2,
        fit_wall_time_s=fit_wall_time_s,
    )
    records.append(rec_np)

    # Calibrate NDIS wrapper (Figure 5, Step 1)
    print("Calibrating NDIS wrapper (finding tau*) ...")
    print("  [This usually takes a few seconds for the standalone GPR demo configs]")
    try:
        t_cal = time.perf_counter()
        wrapper = NDISGaussianWrapper()
        wrapper.calibrate(gpr, epsilon=epsilon, delta=delta, public_meta=public_meta)
        calibration_wall_time_s = time.perf_counter() - t_cal

        tau_star = wrapper.tau_star
        sigma_std = wrapper.sigma_std
        sens_at_tau = gpr.sensitivity(tau_star, public_meta)

        print(f"  tau* = {tau_star:.4e}  (covariance inflation)")
        print(f"  sigma_std = sqrt(tau*) = {sigma_std:.4e}  (noise std dev)")
        print(f"  Delta(tau*) = {sens_at_tau.Delta:.6f}")
        print(f"  rho_inf(tau*) = {sens_at_tau.rho_inf:.6e}")
        print(f"  Cal time: {calibration_wall_time_s:.3f}s\n")

        # Release samples (Figure 5, Step 2)
        print(f"Releasing {len(seeds)} private samples ...")
        for seed in seeds:
            rec = _run_private_release(
                gaussian_output=gaussian_output,
                wrapper=wrapper,
                dataset_name=dataset_name,
                y_star=y_star,
                x_star_index=x_star_index,
                epsilon=epsilon,
                delta=delta,
                sigma_n2=sigma_n2,
                seed=seed,
                calibration_wall_time_s=calibration_wall_time_s,
            )
            records.append(rec)
            print(
                f"  [seed={seed}] θ_tilde = {rec['private']['theta_tilde']:.4f}, "
                f"|θ_tilde - y_★| = {rec['error_private']:.4f}"
            )

    except Exception as exc:
        print(f"  CALIBRATION ERROR: {exc}")
        for seed in seeds:
            records.append({
                "mechanism": "GPR_NDIS",
                "dataset": dataset_name,
                "epsilon": epsilon,
                "delta": delta,
                "sigma_n2": sigma_n2,
                "seed": seed,
                "error": str(exc),
            })

    # Save JSONL
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"gpr_ndis_results_{ts}.jsonl"

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
