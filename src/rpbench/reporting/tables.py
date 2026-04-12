"""Aggregate run results into summary tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_summary_table(records: list[dict[str, Any]], output_path: str | Path) -> pd.DataFrame:
    """Aggregate JSONL records into a summary CSV grouped by mechanism and epsilon."""

    rows = []
    for rec in records:
        if rec["mechanism"] == "NonPrivate":
            continue
        rows.append({
            "mechanism": rec["mechanism"],
            "epsilon": rec["epsilon"],
            "delta": rec["delta"],
            "seed": rec["seed"],
            "test_mse": rec["downstream_metrics"]["test_mse"],
            "rel_fro_xtx": rec["release_metrics"]["rel_fro_xtx"],
            "rel_fro_xtx_normalized": rec["release_metrics"].get(
                "rel_fro_xtx_normalized",
                rec["release_metrics"]["rel_fro_xtx"],
            ),
            "runtime_sec": rec["runtime"]["runtime_sec"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df.to_csv(output_path, index=False)
        return df

    agg = df.groupby(["mechanism", "epsilon"]).agg(
        test_mse_mean=("test_mse", "mean"),
        test_mse_std=("test_mse", "std"),
        rel_fro_mean=("rel_fro_xtx", "mean"),
        rel_fro_std=("rel_fro_xtx", "std"),
        rel_fro_normalized_mean=("rel_fro_xtx_normalized", "mean"),
        rel_fro_normalized_std=("rel_fro_xtx_normalized", "std"),
        runtime_mean=("runtime_sec", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()

    agg.to_csv(output_path, index=False)
    return agg
