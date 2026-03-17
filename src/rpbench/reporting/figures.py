"""Generate benchmark figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_eps_vs_mse(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Plot epsilon vs test MSE with one line per mechanism."""

    rows = []
    delta_val = None
    for rec in records:
        if rec["mechanism"] == "NonPrivate":
            continue
        if delta_val is None and rec["delta"] is not None:
            delta_val = rec["delta"]
        rows.append({
            "mechanism": rec["mechanism"],
            "epsilon": rec["epsilon"],
            "test_mse": rec["downstream_metrics"]["test_mse"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return

    agg = df.groupby(["mechanism", "epsilon"]).agg(
        mean=("test_mse", "mean"),
        std=("test_mse", "std"),
    ).reset_index()

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
    ax.set_ylabel("Test MSE")
    delta_str = f"{delta_val:.2e}" if delta_val else "1/n^2"
    ax.set_title(f"Privacy-Utility: AutoMPG OLS (δ = {delta_str})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
