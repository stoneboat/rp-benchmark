#!/usr/bin/env python
"""CLI entry point: build report from saved run outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rpbench.reporting.tables import build_summary_table
from rpbench.reporting.figures import ols_plot_eps_vs_mse, plot_eps_vs_covariance_error
from rpbench.reporting.summary import write_summary
from rpbench.utils.io import load_jsonl, short_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report from demo run outputs")
    parser.add_argument(
        "--input-root", type=str, required=True,
        help="Directory containing demo_results.jsonl",
    )
    parser.add_argument(
        "--output-root", type=str, required=True,
        help="Directory to write report outputs",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    input_path = input_root / "demo_results.jsonl"
    if not input_path.exists():
        candidates = sorted(input_root.glob("demo_results_*.jsonl"))
        if not candidates:
            print(f"Error: no demo result JSONL found under {input_root}")
            sys.exit(1)
        input_path = candidates[-1]
        print(f"Using latest timestamped run file: {input_path}")

    records = load_jsonl(input_path)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    ts = short_timestamp()

    csv_path = out / f"summary_table_{ts}.csv"
    build_summary_table(records, csv_path)
    print(f"Summary table: {csv_path}")

    ols_fig_path = out / f"ols_plot_eps_vs_mse_{ts}.png"
    ols_plot_eps_vs_mse(records, ols_fig_path)
    print(f"OLS figure: {ols_fig_path}")

    covariance_fig_path = out / f"covariance_plot_eps_vs_error_{ts}.png"
    plot_eps_vs_covariance_error(records, covariance_fig_path)
    print(f"Covariance figure: {covariance_fig_path}")

    md_path = out / f"demo_summary_{ts}.md"
    write_summary(records, md_path, ols_fig_path.name, covariance_fig_path.name, csv_path.name)
    print(f"Summary: {md_path}")

    # Also refresh stable "latest" names for convenience.
    build_summary_table(records, out / "summary_table.csv")
    ols_plot_eps_vs_mse(records, out / "ols_plot_eps_vs_mse.png")
    plot_eps_vs_covariance_error(records, out / "covariance_plot_eps_vs_error.png")
    write_summary(
        records,
        out / "demo_summary.md",
        "ols_plot_eps_vs_mse.png",
        "covariance_plot_eps_vs_error.png",
        "summary_table.csv",
    )
    print("Updated latest pointers: summary_table.csv, ols_plot_eps_vs_mse.png, covariance_plot_eps_vs_error.png, demo_summary.md")


if __name__ == "__main__":
    main()
