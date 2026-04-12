"""Generate benchmark figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _dataset_name(records: list[dict[str, Any]]) -> str:
    for rec in records:
        dataset = rec.get("dataset")
        if dataset:
            return str(dataset)
    return "dataset"


def _aggregate_metric(
    records: list[dict[str, Any]],
    metric_key: str,
    metric_section: str,
) -> tuple[pd.DataFrame, float | None]:
    """Aggregate a scalar metric by mechanism and epsilon."""
    rows = []
    delta_val = None
    for rec in records:
        if rec["mechanism"] == "NonPrivate":
            continue
        if delta_val is None and rec["delta"] is not None:
            delta_val = rec["delta"]
        section = rec.get(metric_section, {})
        if metric_key not in section:
            continue
        rows.append({
            "mechanism": rec["mechanism"],
            "epsilon": rec["epsilon"],
            metric_key: section[metric_key],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, delta_val

    agg = df.groupby(["mechanism", "epsilon"]).agg(
        mean=(metric_key, "mean"),
        std=(metric_key, "std"),
    ).reset_index()
    return agg, delta_val


def ols_plot_eps_vs_mse(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Plot epsilon vs OLS test MSE with one line per mechanism."""

    agg, delta_val = _aggregate_metric(records, "test_mse", "downstream_metrics")
    if agg.empty:
        return
    dataset = _dataset_name(records)

    # Find non-private baseline
    np_mse = None
    for rec in records:
        if rec["mechanism"] == "NonPrivate":
            np_mse = rec["downstream_metrics"]["test_mse"]
            break

    fig, ax = plt.subplots(figsize=(7, 5))

    for mech_name, grp in agg.groupby("mechanism"):
        grp = grp.sort_values("epsilon")
        ax.errorbar(
            grp["epsilon"], grp["mean"], yerr=grp["std"],
            marker="o", capsize=3, label=mech_name,
        )

    if np_mse is not None:
        ax.axhline(np_mse, color="gray", linestyle="--", alpha=0.7, label="NonPrivate")

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("OLS Test MSE")
    delta_str = f"{delta_val:.2e}" if delta_val else "1/n^2"
    ax.set_title(f"OLS Downstream Utility: {dataset} (δ = {delta_str})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_eps_vs_covariance_error(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Plot epsilon vs covariance quality using relative Frobenius error."""

    metric_key = "rel_fro_xtx_normalized"
    agg, delta_val = _aggregate_metric(records, metric_key, "release_metrics")
    normalized = not agg.empty
    if agg.empty:
        metric_key = "rel_fro_xtx"
        agg, delta_val = _aggregate_metric(records, metric_key, "release_metrics")
    if agg.empty:
        return
    dataset = _dataset_name(records)

    fig, ax = plt.subplots(figsize=(7, 5))

    for mech_name, grp in agg.groupby("mechanism"):
        grp = grp.sort_values("epsilon")
        ax.errorbar(
            grp["epsilon"], grp["mean"], yerr=grp["std"],
            marker="o", capsize=3, label=mech_name,
        )

    ax.set_xlabel(r"$\varepsilon$")
    ylabel = r"Relative Frobenius Error of $X^\top X$"
    if normalized:
        ylabel += " (normalized/debiased diagnostic)"
    ax.set_ylabel(ylabel)
    delta_str = f"{delta_val:.2e}" if delta_val else "1/n^2"
    title_prefix = "Covariance Release Quality"
    if normalized:
        title_prefix += " (normalized)"
    ax.set_title(f"{title_prefix}: {dataset} (δ = {delta_str})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
