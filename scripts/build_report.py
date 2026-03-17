#!/usr/bin/env python
"""CLI entry point: build report from saved run outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rpbench.reporting.tables import build_summary_table
from rpbench.reporting.figures import plot_eps_vs_mse
from rpbench.reporting.summary import write_summary
from rpbench.utils.io import load_jsonl


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

    input_path = Path(args.input_root) / "demo_results.jsonl"
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    records = load_jsonl(input_path)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "summary_table.csv"
    build_summary_table(records, csv_path)
    print(f"Summary table: {csv_path}")

    fig_path = out / "eps_vs_test_mse.png"
    plot_eps_vs_mse(records, fig_path)
    print(f"Figure: {fig_path}")

    md_path = out / "demo_summary.md"
    write_summary(records, md_path, fig_path.name, csv_path.name)
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
