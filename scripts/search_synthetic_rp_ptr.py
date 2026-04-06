#!/usr/bin/env python
"""Automatic search for realistic synthetic regimes favorable to Mech_RP_PTR."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import yaml

from rpbench.config import DemoConfig, PreprocessSpec, PrivacySpec, SeedBatchSpec, SplitSpec
from rpbench.datasets.synthetic_redundant_regression import SyntheticRedundantRegressionAdapter
from rpbench.mechanisms.rp_ndis import MechRP, MechRPPTR
from rpbench.runners.run_demo import run_demo_from_bundle
from rpbench.utils.io import short_timestamp
from rpbench.utils.linear_algebra import augmented_data


DEFAULT_EPSILON_GRIDS: list[list[float]] = [
    [1.0, 2.0, 4.0, 8.0],
    [2.0, 4.0, 8.0, 16.0],
]

DEFAULT_SPLIT = SplitSpec(train_fraction=0.8, seed=42)
DEFAULT_DELTA_RULE = "1e-5"
DEFAULT_SEED_BATCH = SeedBatchSpec(mode="fixed", base_seed=0, count=5)
DEFAULT_DELTA_SPLITS = [
    (0.5, 0.3, 0.2),
    (0.3, 0.5, 0.2),
]
DEFAULT_DATASET_SEED = 314159

FEATURE_DIMS = [4, 5, 6, 8]
N_PROTOTYPES = [60, 80, 120]
COPIES_PER_PROTOTYPE = [30, 60, 100, 150]
SIGNAL_PROFILES = [
    {"cluster_noise": 0.04, "label_noise": 0.08, "beta_strength": 0.8, "beta_midpoint": 0.58, "beta_bandwidth": 0.24},
    {"cluster_noise": 0.07, "label_noise": 0.12, "beta_strength": 1.1, "beta_midpoint": 0.66, "beta_bandwidth": 0.20},
    {"cluster_noise": 0.10, "label_noise": 0.16, "beta_strength": 1.4, "beta_midpoint": 0.74, "beta_bandwidth": 0.16},
]
CLIP_CHOICES = [
    {"clip_x_bound": 2.5, "clip_y_bound": 1.25},
    {"clip_x_bound": 3.0, "clip_y_bound": 1.50},
    {"clip_x_bound": 3.0, "clip_y_bound": 1.75},
    {"clip_x_bound": 3.5, "clip_y_bound": 1.75},
]
PTR_R_CHOICES = [12, 16, 20]
PTR_TAU_CHOICES = [8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0]


def _spectrum_family(feature_dim: int, family: str) -> list[float]:
    idx = np.arange(feature_dim, dtype=float)
    if family == "flatish":
        vals = 1.25 * np.power(0.90, idx)
    elif family == "balanced":
        vals = 1.40 * np.power(0.84, idx)
    elif family == "tail":
        vals = 1.55 * np.power(0.78, idx)
    else:
        raise ValueError(f"unknown spectrum family: {family}")
    return [round(float(v), 4) for v in vals]


def _dataset_param_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    generation_seed = DEFAULT_DATASET_SEED
    for d in FEATURE_DIMS:
        for n_prototypes in N_PROTOTYPES:
            for copies in COPIES_PER_PROTOTYPE:
                for family in ("flatish", "balanced", "tail"):
                    for signal in SIGNAL_PROFILES:
                        params = {
                            "feature_dim": d,
                            "n_prototypes": n_prototypes,
                            "copies_per_prototype": copies,
                            "cluster_noise": signal["cluster_noise"],
                            "label_noise": signal["label_noise"],
                            "generation_seed": generation_seed,
                            "prototype_eigenvalues": _spectrum_family(d, family),
                            "beta_mode": "mid_weak",
                            "beta_strength": signal["beta_strength"],
                            "beta_midpoint": signal["beta_midpoint"],
                            "beta_bandwidth": signal["beta_bandwidth"],
                            "poisson_diag_qs": [0.2, 0.5, 0.8],
                            "poisson_diag_trials": 2,
                            "poisson_diag_seed": 4242,
                            "blocki12_jl_diag_rs": [12, 20],
                            "blocki12_jl_diag_trials": 2,
                            "blocki12_jl_diag_seed": 4243,
                        }
                        candidates.append(params)
                        generation_seed += 1
    return candidates


def _preprocess_spec(choice: dict[str, float]) -> PreprocessSpec:
    return PreprocessSpec(
        scale_x=True,
        clip_x=True,
        clip_y=True,
        clip_x_bound=float(choice["clip_x_bound"]),
        clip_y_bound=float(choice["clip_y_bound"]),
        missing_policy="drop",
    )


def _dataset_case_name(params: dict[str, Any], preprocess: PreprocessSpec) -> str:
    return (
        f"d{params['feature_dim']}_p{params['n_prototypes']}_c{params['copies_per_prototype']}"
        f"_cn{int(round(100 * params['cluster_noise']))}"
        f"_ln{int(round(100 * params['label_noise']))}"
        f"_cx{str(preprocess.clip_x_bound).replace('.', 'p')}"
        f"_cy{str(preprocess.clip_y_bound).replace('.', 'p')}"
    )


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-12:
        return default
    return float(num / den)


def _bounded_penalty(value: float, ideal_max: float, hard_max: float) -> float:
    if value <= ideal_max:
        return 0.0
    if value >= hard_max:
        return 1.0
    return float((value - ideal_max) / (hard_max - ideal_max))


def _interpretability_bonus(params: dict[str, Any]) -> float:
    bonus = 0.0
    if params["feature_dim"] in {5, 6}:
        bonus += 0.08
    elif params["feature_dim"] == 4:
        bonus += 0.04
    if params["copies_per_prototype"] in {60, 100}:
        bonus += 0.08
    elif params["copies_per_prototype"] == 30:
        bonus += 0.03
    if params["n_prototypes"] in {80, 120}:
        bonus += 0.06
    if 0.05 <= params["cluster_noise"] <= 0.10:
        bonus += 0.04
    if 0.08 <= params["label_noise"] <= 0.16:
        bonus += 0.04
    return bonus


def _build_cfg(
    dataset_params: dict[str, Any],
    preprocess: PreprocessSpec,
    epsilon_grid: list[float],
    r: int,
    tau: float,
    delta_total: float,
    delta_split: tuple[float, float, float],
    output_root: str,
    seed_count: int,
) -> DemoConfig:
    delta_r = delta_split[0] * delta_total
    delta_t = delta_split[1] * delta_total
    delta_ptr = delta_total - delta_r - delta_t
    return DemoConfig(
        dataset="synthetic_redundant_regression",
        mechanisms=["Mech_RP", "Mech_RP_PTR"],
        task="OLSFromRelease",
        epsilon_grid=list(epsilon_grid),
        delta_rule=DEFAULT_DELTA_RULE,
        seed_batch=SeedBatchSpec(mode="fixed", base_seed=DEFAULT_SEED_BATCH.base_seed, count=seed_count),
        split=DEFAULT_SPLIT,
        preprocess=deepcopy(preprocess),
        mech_params={
            "Mech_RP": {"r": int(r)},
            "Mech_RP_PTR": {
                "r": int(r),
                "tau": float(tau),
                "delta_r": float(delta_r),
                "delta_t": float(delta_t),
                "delta_ptr": float(delta_ptr),
            },
        },
        output_root=output_root,
        dataset_params=deepcopy(dataset_params),
    )


def _load_bundle(dataset_params: dict[str, Any], preprocess: PreprocessSpec) -> tuple[Any, Any, dict[str, Any], np.ndarray, float]:
    adapter = SyntheticRedundantRegressionAdapter(**dataset_params)
    bundle = adapter.load(DEFAULT_SPLIT, preprocess)
    pub_meta = adapter.public_meta(bundle)
    A_train = augmented_data(bundle.X_train, bundle.y_train)
    lambda_min_raw = float(np.linalg.eigvalsh(A_train.T @ A_train).min())
    return adapter, bundle, pub_meta, A_train, lambda_min_raw


def _score_dataset_screen(bundle: Any, params: dict[str, Any], pub_meta: dict[str, Any], lambda_min_raw: float) -> dict[str, Any]:
    prep = bundle.meta["preprocess_diagnostics"]
    geom = bundle.meta["geometry_diagnostics"]
    x_clip = float(prep["x_row_clip_fraction_train"])
    y_clip = float(prep["y_clip_fraction_train"])
    cond = float(geom["xtx_condition_number"])
    cond_penalty = 1.0 if not math.isfinite(cond) else _bounded_penalty(cond, 40.0, 200.0)
    lambda_ratio = _safe_ratio(lambda_min_raw, pub_meta["l"] ** 2, default=0.0)
    score = (
        1.5
        - 1.2 * _bounded_penalty(x_clip, 0.05, 0.20)
        - 1.2 * _bounded_penalty(y_clip, 0.10, 0.30)
        - 0.6 * cond_penalty
        + 0.45 * min(lambda_ratio / 25.0, 1.0)
        + _interpretability_bonus(params)
    )
    return {
        "score": float(score),
        "x_clip_fraction_train": x_clip,
        "y_clip_fraction_train": y_clip,
        "xtx_condition_number": cond,
        "xtx_eig_min": float(geom["xtx_eig_min"]),
        "xtx_eig_max": float(geom["xtx_eig_max"]),
        "lambda_min_raw": lambda_min_raw,
        "lambda_min_to_l2_sq_ratio": lambda_ratio,
    }


def _ptr_proxy_summary(
    pub_meta: dict[str, Any],
    lambda_min_raw: float,
    epsilon_grid: list[float],
    r: int,
    tau: float,
    delta_total: float,
    delta_split: tuple[float, float, float],
) -> dict[str, Any] | None:
    delta_r = delta_split[0] * delta_total
    delta_t = delta_split[1] * delta_total
    delta_ptr = delta_total - delta_r - delta_t
    rp = MechRP(r=r)
    proxy_rows: list[dict[str, Any]] = []
    try:
        for epsilon in epsilon_grid:
            ps = PrivacySpec(epsilon=float(epsilon), delta=float(delta_total))
            rp.calibrate(ps, pub_meta)
            ptr = MechRPPTR(r=r, tau=tau, delta_r=delta_r, delta_t=delta_t, delta_ptr=delta_ptr)
            ptr.calibrate(ps, pub_meta)
            alpha = float(ptr.diagnostics()["alpha"])
            epsilon_t = float(ptr.diagnostics()["epsilon_T"])
            epsilon_r = float(ptr.diagnostics()["epsilon_R"])
            lambda_rp = float(ptr.diagnostics()["lambda_rp"])
            lambda_lb_proxy = max(lambda_min_raw - alpha, 0.0)
            lambda_ptr_proxy = max(pub_meta["l"] ** 2 / ptr.diagnostics()["p_star_ptr"] - lambda_lb_proxy, 0.0)
            proxy_rows.append({
                "epsilon": float(epsilon),
                "epsilon_T": epsilon_t,
                "epsilon_R": epsilon_r,
                "alpha": alpha,
                "lambda_rp": lambda_rp,
                "lambda_lb_proxy": lambda_lb_proxy,
                "lambda_ptr_proxy": float(lambda_ptr_proxy),
                "ptr_lt_rp_proxy": bool(lambda_ptr_proxy < lambda_rp),
            })
    except Exception:
        return None

    positive_epsilon_r = sum(row["epsilon_R"] > 1e-9 for row in proxy_rows)
    positive_lb = sum(row["lambda_lb_proxy"] > 0.0 for row in proxy_rows)
    ptr_better = sum(row["ptr_lt_rp_proxy"] for row in proxy_rows)
    mean_ridge_reduction = _mean(
        [_safe_ratio(row["lambda_rp"] - row["lambda_ptr_proxy"], row["lambda_rp"]) for row in proxy_rows]
    )
    score = (
        1.0 * positive_epsilon_r
        + 1.0 * positive_lb
        + 1.4 * ptr_better
        + 2.0 * mean_ridge_reduction
        - 0.4 * sum(row["epsilon_R"] <= 1e-9 for row in proxy_rows)
    )
    return {
        "proxy_rows": proxy_rows,
        "positive_epsilon_r_count": int(positive_epsilon_r),
        "positive_lambda_lb_count": int(positive_lb),
        "ptr_lt_rp_count": int(ptr_better),
        "mean_ridge_reduction": mean_ridge_reduction,
        "delta_split": list(delta_split),
        "score": float(score),
    }


def _paired_seed_win_fraction(records: list[dict[str, Any]], epsilon: float) -> float:
    rp = {
        rec["seed"]: rec["downstream_metrics"]["test_mse"]
        for rec in records
        if rec["mechanism"] == "Mech_RP" and rec["epsilon"] == epsilon
    }
    ptr = {
        rec["seed"]: rec["downstream_metrics"]["test_mse"]
        for rec in records
        if rec["mechanism"] == "Mech_RP_PTR" and rec["epsilon"] == epsilon
    }
    shared = sorted(set(rp) & set(ptr))
    if not shared:
        return float("nan")
    return float(np.mean([ptr[s] < rp[s] for s in shared]))


def _epsilon_summary(records: list[dict[str, Any]], epsilon: float) -> dict[str, Any]:
    rp_rows = [rec for rec in records if rec["mechanism"] == "Mech_RP" and rec["epsilon"] == epsilon]
    ptr_rows = [rec for rec in records if rec["mechanism"] == "Mech_RP_PTR" and rec["epsilon"] == epsilon]
    rp_mse = [rec["downstream_metrics"]["test_mse"] for rec in rp_rows]
    ptr_mse = [rec["downstream_metrics"]["test_mse"] for rec in ptr_rows]
    ptr_diag = [rec["diagnostics"] for rec in ptr_rows]
    rp_mean = _mean(rp_mse)
    ptr_mean = _mean(ptr_mse)
    abs_gain = rp_mean - ptr_mean
    rel_gain = _safe_ratio(abs_gain, rp_mean)
    lambda_lb_positive_rate = float(np.mean([diag["lambda_lb"] > 0.0 for diag in ptr_diag])) if ptr_diag else float("nan")
    lambda_ptr_lt_rp_rate = float(np.mean([diag["lambda_ptr_lt_lambda_rp"] for diag in ptr_diag])) if ptr_diag else float("nan")
    return {
        "epsilon": float(epsilon),
        "rp_mean_test_mse": rp_mean,
        "ptr_mean_test_mse": ptr_mean,
        "absolute_gain": abs_gain,
        "relative_gain": rel_gain,
        "paired_seed_win_fraction": _paired_seed_win_fraction(records, epsilon),
        "epsilon_T": _mean([diag["epsilon_T"] for diag in ptr_diag]),
        "epsilon_R": _mean([diag["epsilon_R"] for diag in ptr_diag]),
        "lambda_min_raw": _mean([diag["lambda_min_raw"] for diag in ptr_diag]),
        "alpha": _mean([diag["alpha"] for diag in ptr_diag]),
        "lambda_lb_mean": _mean([diag["lambda_lb"] for diag in ptr_diag]),
        "lambda_lb_positive_rate": lambda_lb_positive_rate,
        "lambda_ptr_mean": _mean([diag["lambda_ptr"] for diag in ptr_diag]),
        "lambda_rp_mean": _mean([diag["lambda_rp"] for diag in ptr_diag]),
        "lambda_ptr_lt_rp_rate": lambda_ptr_lt_rp_rate,
    }


def _final_score(
    epsilon_rows: list[dict[str, Any]],
    dataset_screen: dict[str, Any],
    params: dict[str, Any],
) -> float:
    weights = {1.0: 1.4, 2.0: 1.2, 4.0: 1.0, 8.0: 0.8, 16.0: 0.6}
    gain_term = 0.0
    win_term = 0.0
    epsilon_r_penalty = 0.0
    lambda_lb_penalty = 0.0
    lambda_ptr_penalty = 0.0
    for row in epsilon_rows:
        weight = weights.get(row["epsilon"], 1.0)
        gain_term += weight * max(min(row["relative_gain"], 1.0), -2.0)
        win_term += weight * max(row["paired_seed_win_fraction"] - 0.5, 0.0)
        epsilon_r_penalty += weight * float(row["epsilon_R"] <= 1e-9)
        lambda_lb_penalty += weight * max(0.0, 0.5 - row["lambda_lb_positive_rate"])
        lambda_ptr_penalty += weight * max(0.0, 0.5 - row["lambda_ptr_lt_rp_rate"])
    clip_penalty = (
        1.5 * _bounded_penalty(dataset_screen["x_clip_fraction_train"], 0.05, 0.20)
        + 1.2 * _bounded_penalty(dataset_screen["y_clip_fraction_train"], 0.15, 0.35)
    )
    interpretability = _interpretability_bonus(params)
    return float(
        5.0 * gain_term
        + 1.5 * win_term
        + 0.8 * interpretability
        + 0.5 * dataset_screen["score"]
        - 1.3 * clip_penalty
        - 0.8 * epsilon_r_penalty
        - 0.7 * lambda_lb_penalty
        - 0.6 * lambda_ptr_penalty
    )


def _write_yaml_configs(results: list[dict[str, Any]], output_dir: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, result in enumerate(results[:count], start=1):
        cfg = result["config"]
        path = output_dir / f"synthetic_ptr_search_rank{rank:02d}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        paths.append(path)
    return paths


def _config_dict(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "synthetic_redundant_regression",
        "dataset_params": result["dataset_params"],
        "mechanisms": ["Mech_RP", "Mech_RP_PTR"],
        "task": "OLSFromRelease",
        "epsilon_grid": result["epsilon_grid"],
        "delta_rule": DEFAULT_DELTA_RULE,
        "seed_batch": {
            "mode": DEFAULT_SEED_BATCH.mode,
            "base_seed": DEFAULT_SEED_BATCH.base_seed,
            "count": result["seed_count"],
        },
        "split": {
            "train_fraction": DEFAULT_SPLIT.train_fraction,
            "seed": DEFAULT_SPLIT.seed,
        },
        "preprocess": result["preprocess"],
        "mech_params": {
            "Mech_RP": {"r": result["r"]},
            "Mech_RP_PTR": {
                "r": result["r"],
                "tau": result["tau"],
                "delta_r": result["delta_r"],
                "delta_t": result["delta_t"],
                "delta_ptr": result["delta_ptr"],
            },
        },
        "output_root": result["output_root"],
    }


def _write_summary(results: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Synthetic RP vs RP_PTR Search",
        "",
        "Ranked by seeded downstream PTR gain with penalties for heavy clipping and inactive PTR regimes.",
        "",
    ]
    for rank, result in enumerate(results, start=1):
        clip = result["clipping"]
        lines.extend([
            f"## {rank}. {result['name']}",
            "",
            f"- score: {result['score']:.4f}",
            f"- epsilon_grid: {result['epsilon_grid']}",
            f"- r={result['r']}, tau={result['tau']}, delta_split={result['delta_split']}",
            f"- clipping fractions (train): x={clip['x_row_clip_fraction_train']:.3%}, y={clip['y_clip_fraction_train']:.3%}",
            f"- dataset: d={result['dataset_params']['feature_dim']}, "
            f"n_prototypes={result['dataset_params']['n_prototypes']}, "
            f"copies={result['dataset_params']['copies_per_prototype']}, "
            f"cluster_noise={result['dataset_params']['cluster_noise']}, "
            f"label_noise={result['dataset_params']['label_noise']}",
            "",
        ])
        for row in result["epsilon_summary"]:
            lines.append(
                f"  eps={row['epsilon']:.1f}: RP={row['rp_mean_test_mse']:.6f}, "
                f"PTR={row['ptr_mean_test_mse']:.6f}, gain={row['relative_gain']:.2%}, "
                f"win_frac={row['paired_seed_win_fraction']:.2f}, "
                f"eps_R={row['epsilon_R']:.3f}, "
                f"lambda_lb+={row['lambda_lb_positive_rate']:.2f}, "
                f"lambda_ptr<rp={row['lambda_ptr_lt_rp_rate']:.2f}"
            )
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search for realistic synthetic regimes favorable to Mech_RP_PTR")
    parser.add_argument("--seed-count", type=int, default=5, help="Trial seeds per fully evaluated candidate")
    parser.add_argument("--dataset-shortlist", type=int, default=24, help="Dataset/preprocess candidates kept after clipping prescreen")
    parser.add_argument("--full-eval-count", type=int, default=12, help="PTR candidates fully benchmarked after proxy prescreen")
    parser.add_argument("--top-config-count", type=int, default=3, help="Top YAML configs to write")
    parser.add_argument("--output-root", type=str, default="data/outputs/searches", help="Directory for search summaries")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    timestamp = short_timestamp()

    dataset_grid = _dataset_param_grid()
    dataset_screen_rows: list[dict[str, Any]] = []
    stage1_total = len(dataset_grid) * len(CLIP_CHOICES)
    print(f"Stage 1/3: dataset prescreen over {stage1_total} dataset/preprocess candidates", flush=True)
    stage1_done = 0
    for dataset_params in dataset_grid:
        for clip_choice in CLIP_CHOICES:
            preprocess = _preprocess_spec(clip_choice)
            _, bundle, pub_meta, _, lambda_min_raw = _load_bundle(dataset_params, preprocess)
            dataset_screen = _score_dataset_screen(bundle, dataset_params, pub_meta, lambda_min_raw)
            dataset_screen_rows.append({
                "name": _dataset_case_name(dataset_params, preprocess),
                "dataset_params": deepcopy(dataset_params),
                "preprocess_spec": preprocess,
                "dataset_screen": dataset_screen,
            })
            stage1_done += 1
            if stage1_done % 100 == 0 or stage1_done == stage1_total:
                print(f"  screened {stage1_done}/{stage1_total}", flush=True)
    dataset_screen_rows.sort(key=lambda row: row["dataset_screen"]["score"], reverse=True)
    dataset_shortlist = dataset_screen_rows[: args.dataset_shortlist]

    ptr_proxy_rows: list[dict[str, Any]] = []
    stage2_total = len(dataset_shortlist) * len(PTR_R_CHOICES) * len(PTR_TAU_CHOICES) * len(DEFAULT_EPSILON_GRIDS) * len(DEFAULT_DELTA_SPLITS)
    print(f"Stage 2/3: PTR proxy prescreen over {stage2_total} candidates", flush=True)
    stage2_done = 0
    for row in dataset_shortlist:
        adapter, bundle, pub_meta, _, lambda_min_raw = _load_bundle(row["dataset_params"], row["preprocess_spec"])
        delta_total = float(DEFAULT_DELTA_RULE)
        for epsilon_grid in DEFAULT_EPSILON_GRIDS:
            for delta_split in DEFAULT_DELTA_SPLITS:
                for r in PTR_R_CHOICES:
                    for tau in PTR_TAU_CHOICES:
                        proxy = _ptr_proxy_summary(
                            pub_meta=pub_meta,
                            lambda_min_raw=lambda_min_raw,
                            epsilon_grid=epsilon_grid,
                            r=r,
                            tau=tau,
                            delta_total=delta_total,
                            delta_split=delta_split,
                        )
                        stage2_done += 1
                        if proxy is None:
                            if stage2_done % 50 == 0 or stage2_done == stage2_total:
                                print(f"  proxied {stage2_done}/{stage2_total}", flush=True)
                            continue
                        ptr_proxy_rows.append({
                            **row,
                            "bundle": bundle,
                            "pub_meta": pub_meta,
                            "preprocess_meta": bundle.meta["preprocess_diagnostics"],
                            "synthetic_meta": bundle.meta["synthetic_params"],
                            "adapter_name": adapter.name,
                            "lambda_min_raw": lambda_min_raw,
                            "epsilon_grid": list(epsilon_grid),
                            "r": r,
                            "tau": tau,
                            "delta_split": delta_split,
                            "proxy": proxy,
                            "proxy_score": row["dataset_screen"]["score"] + proxy["score"],
                        })
                        if stage2_done % 50 == 0 or stage2_done == stage2_total:
                            print(f"  proxied {stage2_done}/{stage2_total}", flush=True)
    eligible_proxy_rows = [
        row for row in ptr_proxy_rows
        if row["proxy"]["positive_epsilon_r_count"] >= 2
        and row["proxy"]["positive_lambda_lb_count"] >= 1
        and row["proxy"]["ptr_lt_rp_count"] >= 1
        and row["proxy"]["mean_ridge_reduction"] > 0.0
    ]
    ranked_proxy_rows = eligible_proxy_rows if eligible_proxy_rows else ptr_proxy_rows
    ranked_proxy_rows.sort(key=lambda row: row["proxy_score"], reverse=True)
    full_eval_candidates = ranked_proxy_rows[: args.full_eval_count]

    final_results: list[dict[str, Any]] = []
    print(f"Stage 3/3: full seeded evaluation for {len(full_eval_candidates)} candidates", flush=True)
    for rank, row in enumerate(full_eval_candidates, start=1):
        delta_total = float(DEFAULT_DELTA_RULE)
        cfg = _build_cfg(
            dataset_params=row["dataset_params"],
            preprocess=row["preprocess_spec"],
            epsilon_grid=row["epsilon_grid"],
            r=row["r"],
            tau=row["tau"],
            delta_total=delta_total,
            delta_split=row["delta_split"],
            output_root=(
                f"data/outputs/runs/{row['name']}_r{row['r']}_tau{str(row['tau']).replace('.', 'p')}"
                f"_ds{int(10 * row['delta_split'][0])}{int(10 * row['delta_split'][1])}{int(10 * row['delta_split'][2])}"
            ),
            seed_count=args.seed_count,
        )
        print(
            f"  [{rank}/{len(full_eval_candidates)}] {row['name']} "
            f"grid={row['epsilon_grid']} r={row['r']} tau={row['tau']}"
        , flush=True)
        records = run_demo_from_bundle(cfg, row["bundle"], row["pub_meta"])
        epsilon_rows = [_epsilon_summary(records, epsilon) for epsilon in cfg.epsilon_grid]
        result = {
            "name": row["name"],
            "score": _final_score(epsilon_rows, row["dataset_screen"], row["dataset_params"]),
            "dataset_params": row["dataset_params"],
            "preprocess": {
                "scale_x": True,
                "clip_x": True,
                "clip_y": True,
                "clip_x_bound": row["preprocess_spec"].clip_x_bound,
                "clip_y_bound": row["preprocess_spec"].clip_y_bound,
                "missing_policy": row["preprocess_spec"].missing_policy,
            },
            "clipping": deepcopy(row["preprocess_meta"]),
            "synthetic_meta": deepcopy(row["synthetic_meta"]),
            "dataset_screen": deepcopy(row["dataset_screen"]),
            "epsilon_grid": list(cfg.epsilon_grid),
            "epsilon_summary": epsilon_rows,
            "r": row["r"],
            "tau": row["tau"],
            "seed_count": args.seed_count,
            "delta_r": cfg.mech_params["Mech_RP_PTR"]["delta_r"],
            "delta_t": cfg.mech_params["Mech_RP_PTR"]["delta_t"],
            "delta_ptr": cfg.mech_params["Mech_RP_PTR"]["delta_ptr"],
            "delta_split": list(row["delta_split"]),
            "proxy_summary": row["proxy"],
            "output_root": cfg.output_root,
            "config": None,
        }
        result["config"] = _config_dict(result)
        final_results.append(result)

    final_results.sort(key=lambda row: row["score"], reverse=True)

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"synthetic_rp_ptr_search_{timestamp}.json"
    latest_json_path = output_root / "synthetic_rp_ptr_search_latest.json"
    summary_path = output_root / f"synthetic_rp_ptr_search_{timestamp}.md"
    latest_summary_path = output_root / "synthetic_rp_ptr_search_latest.md"

    json_path.write_text(json.dumps(final_results, indent=2))
    latest_json_path.write_text(json.dumps(final_results, indent=2))
    _write_summary(final_results, summary_path)
    _write_summary(final_results, latest_summary_path)
    config_paths = _write_yaml_configs(final_results, Path("configs/demo"), args.top_config_count)

    if final_results:
        best = final_results[0]
        print(
            f"Best candidate: {best['name']} | score={best['score']:.4f} | "
            f"grid={best['epsilon_grid']} | r={best['r']} tau={best['tau']}"
        , flush=True)
    print(f"Wrote JSON results to {json_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    if config_paths:
        print("Wrote YAML configs:", flush=True)
        for path in config_paths:
            print(f"  - {path}", flush=True)


if __name__ == "__main__":
    main()
