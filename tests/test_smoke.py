"""Smoke test: import rpbench and run a minimal config."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def test_import():
    import rpbench
    assert rpbench.__version__


def test_mini_run():
    """Run 1 mechanism, 1 epsilon, 1 seed end-to-end."""
    from rpbench.config import DemoConfig, SplitSpec, PreprocessSpec
    from rpbench.runners.run_demo import run_demo

    cfg = DemoConfig(
        dataset="autompg",
        mechanisms=["Mech_RP"],
        task="OLSFromRelease",
        epsilon_grid=[2.0],
        delta_rule="1/n^2",
        seeds=[0],
        split=SplitSpec(train_fraction=0.8, seed=42),
        preprocess=PreprocessSpec(),
        mech_params={"Mech_RP": {"r_over_d": 4}},
        output_root=tempfile.mkdtemp(),
    )

    records = run_demo(cfg)
    assert len(records) >= 2  # NonPrivate + 1 run
    for rec in records:
        assert "downstream_metrics" in rec
        assert "release_metrics" in rec


def test_report_builder():
    """Test that the report builder can consume saved outputs."""
    import tempfile
    from pathlib import Path

    from rpbench.config import DemoConfig, SplitSpec, PreprocessSpec
    from rpbench.runners.run_demo import run_demo
    from rpbench.utils.io import save_jsonl, load_jsonl
    from rpbench.reporting.tables import build_summary_table
    from rpbench.reporting.figures import plot_eps_vs_mse
    from rpbench.reporting.summary import write_summary

    tmpdir = Path(tempfile.mkdtemp())

    cfg = DemoConfig(
        dataset="autompg",
        mechanisms=["Mech_RP"],
        task="OLSFromRelease",
        epsilon_grid=[1.0, 2.0],
        delta_rule="1/n^2",
        seeds=[0],
        split=SplitSpec(train_fraction=0.8, seed=42),
        preprocess=PreprocessSpec(),
        mech_params={"Mech_RP": {"r_over_d": 4}},
        output_root=str(tmpdir / "runs"),
    )

    records = run_demo(cfg)
    jsonl_path = tmpdir / "runs" / "demo_results.jsonl"
    save_jsonl(records, jsonl_path)

    loaded = load_jsonl(jsonl_path)
    assert len(loaded) == len(records)

    report_dir = tmpdir / "reports"
    report_dir.mkdir()

    csv_path = report_dir / "summary_table.csv"
    build_summary_table(loaded, csv_path)
    assert csv_path.exists()

    fig_path = report_dir / "test_figure.png"
    plot_eps_vs_mse(loaded, fig_path)
    assert fig_path.exists()

    md_path = report_dir / "summary.md"
    write_summary(loaded, md_path, "test_figure.png", "summary_table.csv")
    assert md_path.exists()
