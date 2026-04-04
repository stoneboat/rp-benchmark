#!/usr/bin/env python
"""Search for synthetic redundant-regression regimes where Mech_RP_Pois beats Mech_RP."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from rpbench.config import DemoConfig, PreprocessSpec, SeedBatchSpec, SplitSpec
from rpbench.datasets.synthetic_redundant_regression import SyntheticRedundantRegressionAdapter
from rpbench.runners.run_demo import run_demo
from rpbench.utils.io import short_timestamp


DEFAULT_DATASET_BASE: dict[str, Any] = {
    "feature_dim": 12,
    "n_prototypes": 60,
    "copies_per_prototype": 100,
    "cluster_noise": 0.025,
    "label_noise": 0.05,
    "generation_seed": 12345,
    "prototype_eigenvalues": [1.60, 1.30, 1.05, 0.86, 0.70, 0.57, 0.46, 0.37, 0.30, 0.24, 0.19, 0.15],
    "beta_mode": "mid_weak",
    "beta_strength": 1.5,
    "beta_midpoint": 0.72,
    "beta_bandwidth": 0.16,
    "poisson_diag_qs": [0.2, 0.3, 0.5, 0.8, 0.9],
    "poisson_diag_trials": 20,
    "poisson_diag_seed": 4242,
    "blocki12_jl_diag_rs": [20, 30],
    "blocki12_jl_diag_trials": 15,
    "blocki12_jl_diag_seed": 4243,
}

DEFAULT_PREPROCESS = PreprocessSpec(
    scale_x=True,
    clip_x=True,
    clip_x_bound=3.0,
    clip_y=True,
    clip_y_bound=1.0,
    missing_policy="drop",
)

DEFAULT_SPLIT = SplitSpec(train_fraction=0.8, seed=42)

CASE_SPECS: list[dict[str, Any]] = [
    {
        "name": "positive_q03_r20",
        "dataset_params": {},
        "q": 0.3,
        "r": 20,
    },
    {
        "name": "positive_q05_r20",
        "dataset_params": {},
        "q": 0.5,
        "r": 20,
    },
    {
        "name": "positive_q08_r20",
        "dataset_params": {},
        "q": 0.8,
        "r": 20,
    },
    {
        "name": "weak_beta_q03_r24",
        "dataset_params": {
            "feature_dim": 10,
            "n_prototypes": 75,
            "copies_per_prototype": 120,
            "cluster_noise": 0.03,
            "label_noise": 0.05,
            "generation_seed": 12346,
            "prototype_eigenvalues": [1.45, 1.18, 0.96, 0.79, 0.65, 0.53, 0.43, 0.34, 0.27, 0.21],
            "beta_mode": "weak",
            "beta_strength": 1.35,
            "beta_midpoint": 0.7,
            "beta_bandwidth": 0.18,
        },
        "q": 0.3,
        "r": 24,
    },
    {
        "name": "neutral_random_q05_r30",
        "dataset_params": {
            "feature_dim": 10,
            "n_prototypes": 80,
            "copies_per_prototype": 60,
            "cluster_noise": 0.05,
            "label_noise": 0.05,
            "generation_seed": 22345,
            "prototype_eigenvalues": [1.40, 1.18, 0.98, 0.82, 0.68, 0.56, 0.46, 0.37, 0.30, 0.24],
            "beta_mode": "random",
        },
        "q": 0.5,
        "r": 30,
    },
    {
        "name": "negative_low_redundancy_q02_r50",
        "dataset_params": {
            "feature_dim": 12,
            "n_prototypes": 50,
            "copies_per_prototype": 12,
            "cluster_noise": 0.12,
            "label_noise": 0.04,
            "generation_seed": 32345,
            "prototype_eigenvalues": [1.60, 1.35, 1.10, 0.92, 0.78, 0.65, 0.54, 0.45, 0.38, 0.31, 0.25, 0.20],
            "beta_mode": "random",
            "poisson_diag_qs": [0.2, 0.5, 0.8],
        },
        "q": 0.2,
        "r": 50,
    },
]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    out.update(override)
    return out


def _build_config(case: dict[str, Any], seed_count: int, epsilon_grid: list[float]) -> DemoConfig:
    dataset_params = _deep_merge(DEFAULT_DATASET_BASE, case["dataset_params"])
    r = int(case["r"])
    q = float(case["q"])
    if q not in dataset_params["poisson_diag_qs"]:
        dataset_params["poisson_diag_qs"] = sorted({*dataset_params["poisson_diag_qs"], q})

    return DemoConfig(
        dataset="synthetic_redundant_regression",
        mechanisms=["Mech_RP", "Mech_RP_Pois"],
        task="OLSFromRelease",
        epsilon_grid=epsilon_grid,
        delta_rule="1e-6",
        seed_batch=SeedBatchSpec(mode="fixed", base_seed=0, count=seed_count),
        split=DEFAULT_SPLIT,
        preprocess=DEFAULT_PREPROCESS,
        mech_params={
            "Mech_RP": {"r": r},
            "Mech_RP_Pois": {"r": r, "q": q},
        },
        dataset_params=dataset_params,
        output_root="data/outputs/search_runs",
    )


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _paired_seed_win_fraction(records: list[dict[str, Any]], epsilon: float) -> float:
    rp = {
        rec["seed"]: rec["downstream_metrics"]["test_mse"]
        for rec in records
        if rec["mechanism"] == "Mech_RP" and rec["epsilon"] == epsilon
    }
    pois = {
        rec["seed"]: rec["downstream_metrics"]["test_mse"]
        for rec in records
        if rec["mechanism"] == "Mech_RP_Pois" and rec["epsilon"] == epsilon
    }
    shared = sorted(set(rp) & set(pois))
    if not shared:
        return float("nan")
    return float(np.mean([pois[s] < rp[s] for s in shared]))


def _summarize_case(
    case: dict[str, Any],
    cfg: DemoConfig,
    records: list[dict[str, Any]],
    dataset_meta: dict[str, Any],
    min_relative_gain: float,
    max_relative_opnorm_p90: float,
    max_whitened_opnorm_p90: float,
    min_expected_retained: float,
) -> dict[str, Any]:
    q_key = str(case["q"])
    poisson_diag = dataset_meta["poisson_subsampling_diagnostics"][q_key]
    retention_diag = dataset_meta["prototype_retention_diagnostics"]["expected_retained_copies_per_prototype"][q_key]
    epsilon_rows: list[dict[str, Any]] = []
    positive_epsilons: list[float] = []

    for epsilon in cfg.epsilon_grid:
        rp_vals = [
            rec["downstream_metrics"]["test_mse"]
            for rec in records
            if rec["mechanism"] == "Mech_RP" and rec["epsilon"] == epsilon
        ]
        pois_vals = [
            rec["downstream_metrics"]["test_mse"]
            for rec in records
            if rec["mechanism"] == "Mech_RP_Pois" and rec["epsilon"] == epsilon
        ]
        rp_mean = _mean(rp_vals)
        pois_mean = _mean(pois_vals)
        absolute_gain = rp_mean - pois_mean
        relative_gain = absolute_gain / max(rp_mean, 1e-12)
        win_fraction = _paired_seed_win_fraction(records, epsilon)
        if relative_gain >= min_relative_gain and win_fraction >= 0.5:
            positive_epsilons.append(float(epsilon))
        epsilon_rows.append(
            {
                "epsilon": float(epsilon),
                "rp_mean_test_mse": rp_mean,
                "rp_pois_mean_test_mse": pois_mean,
                "absolute_gain": absolute_gain,
                "relative_gain": relative_gain,
                "paired_seed_win_fraction": win_fraction,
            }
        )

    geometry_supported = (
        poisson_diag["relative_opnorm_p90"] <= max_relative_opnorm_p90
        and poisson_diag["whitened_relative_opnorm_p90"] <= max_whitened_opnorm_p90
        and retention_diag >= min_expected_retained
    )
    score = 0.0
    if epsilon_rows:
        score += max(row["relative_gain"] for row in epsilon_rows)
    if geometry_supported:
        score += 0.25

    return {
        "name": case["name"],
        "q": float(case["q"]),
        "r": int(case["r"]),
        "dataset_params": cfg.dataset_params,
        "geometry_supported": geometry_supported,
        "positive_regime": geometry_supported and bool(positive_epsilons),
        "positive_epsilons": positive_epsilons,
        "score": float(score),
        "geometry": {
            "xtx_condition_number": dataset_meta["geometry_diagnostics"]["xtx_condition_number"],
            "xtx_eig_min": dataset_meta["geometry_diagnostics"]["xtx_eig_min"],
            "xtx_eig_max": dataset_meta["geometry_diagnostics"]["xtx_eig_max"],
            "max_leverage": dataset_meta["geometry_diagnostics"]["max_leverage"],
            "q_times_copies_per_prototype": dataset_meta["geometry_diagnostics"]["q_times_copies_per_prototype"][q_key],
            "q_times_train_copies_per_prototype_mean": dataset_meta["geometry_diagnostics"]["q_times_train_copies_per_prototype_mean"][q_key],
            "poisson_relative_opnorm_p90": poisson_diag["relative_opnorm_p90"],
            "poisson_whitened_relative_opnorm_p90": poisson_diag["whitened_relative_opnorm_p90"],
            "expected_retained_copies_per_prototype": retention_diag,
            "prototype_retention_per_trial": poisson_diag.get("prototype_retention_per_trial", {}),
            "beta_mode": dataset_meta["synthetic_params"]["beta_mode"],
            "beta_weak_half_mass": dataset_meta["synthetic_params"]["beta_weak_half_mass"],
            "beta_mid_to_weak_mass": dataset_meta["synthetic_params"]["beta_mid_to_weak_mass"],
        },
        "epsilon_summary": epsilon_rows,
    }


def _write_summary(results: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Synthetic RP vs RP_Pois Search",
        "",
        "Cases are ranked by downstream gain plus a small bonus for geometry support.",
        "",
    ]
    for rank, result in enumerate(results, start=1):
        lines.extend(
            [
                f"## {rank}. {result['name']}",
                "",
                f"- positive_regime: {result['positive_regime']}",
                f"- positive_epsilons: {result['positive_epsilons']}",
                f"- q={result['q']}, r={result['r']}",
                f"- beta_mode={result['geometry']['beta_mode']}, beta_weak_half_mass={result['geometry']['beta_weak_half_mass']:.3f}",
                f"- rel_opnorm_p90={result['geometry']['poisson_relative_opnorm_p90']:.4f}, "
                f"whitened_p90={result['geometry']['poisson_whitened_relative_opnorm_p90']:.4f}",
                f"- expected_retained_copies={result['geometry']['expected_retained_copies_per_prototype']:.2f}",
                "",
            ]
        )
        for row in result["epsilon_summary"]:
            lines.append(
                f"  eps={row['epsilon']:.2f}: RP={row['rp_mean_test_mse']:.6f}, "
                f"RP_Pois={row['rp_pois_mean_test_mse']:.6f}, "
                f"rel_gain={row['relative_gain']:.3%}, "
                f"paired_win_frac={row['paired_seed_win_fraction']:.2f}"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search synthetic settings where Mech_RP_Pois beats Mech_RP")
    parser.add_argument("--seed-count", type=int, default=5, help="Trial seeds per case")
    parser.add_argument(
        "--epsilon-grid",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0],
        help="Epsilon values to evaluate",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/outputs/searches",
        help="Directory for search summaries",
    )
    parser.add_argument(
        "--min-relative-gain",
        type=float,
        default=0.02,
        help="Minimum relative mean-MSE gain to count as a positive epsilon",
    )
    parser.add_argument(
        "--max-relative-opnorm-p90",
        type=float,
        default=0.25,
        help="Maximum acceptable p90 relative Gram error for geometry support",
    )
    parser.add_argument(
        "--max-whitened-opnorm-p90",
        type=float,
        default=0.45,
        help="Maximum acceptable p90 whitened Gram error for geometry support",
    )
    parser.add_argument(
        "--min-expected-retained",
        type=float,
        default=8.0,
        help="Minimum expected retained copies per prototype for geometry support",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    timestamp = short_timestamp()
    results: list[dict[str, Any]] = []

    for case in CASE_SPECS:
        cfg = _build_config(case, seed_count=args.seed_count, epsilon_grid=list(args.epsilon_grid))
        adapter = SyntheticRedundantRegressionAdapter(**cfg.dataset_params)
        bundle = adapter.load(cfg.split, cfg.preprocess)
        print(f"Running case {case['name']} with q={case['q']} r={case['r']}")
        records = run_demo(cfg)
        results.append(
            _summarize_case(
                case=case,
                cfg=cfg,
                records=records,
                dataset_meta=bundle.meta,
                min_relative_gain=args.min_relative_gain,
                max_relative_opnorm_p90=args.max_relative_opnorm_p90,
                max_whitened_opnorm_p90=args.max_whitened_opnorm_p90,
                min_expected_retained=args.min_expected_retained,
            )
        )

    results.sort(key=lambda row: row["score"], reverse=True)

    json_path = output_root / f"synthetic_rp_pois_search_{timestamp}.json"
    latest_json_path = output_root / "synthetic_rp_pois_search_latest.json"
    summary_path = output_root / f"synthetic_rp_pois_search_{timestamp}.md"
    latest_summary_path = output_root / "synthetic_rp_pois_search_latest.md"

    output_root.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2))
    latest_json_path.write_text(json.dumps(results, indent=2))
    _write_summary(results, summary_path)
    _write_summary(results, latest_summary_path)

    best = results[0] if results else None
    if best is not None:
        print(
            f"Best case: {best['name']} | positive_regime={best['positive_regime']} | "
            f"positive_epsilons={best['positive_epsilons']}"
        )
    print(f"Wrote search summary to {summary_path}")


if __name__ == "__main__":
    main()
