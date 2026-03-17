"""Smoke test: import rpbench and run a minimal config."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def test_import():
    import rpbench
    assert rpbench.__version__


def test_seed_batch_resolution_is_cached():
    from rpbench.config import DemoConfig, SplitSpec, PreprocessSpec, SeedBatchSpec

    cfg = DemoConfig(
        dataset="autompg",
        mechanisms=["Mech_RP"],
        task="OLSFromRelease",
        epsilon_grid=[2.0],
        delta_rule="1/n^2",
        seed_batch=SeedBatchSpec(mode="random", count=3),
        split=SplitSpec(train_fraction=0.8, seed=42),
        preprocess=PreprocessSpec(),
        mech_params={"Mech_RP": {"r": 24}},
        output_root=tempfile.mkdtemp(),
    )

    base_seed_1, seeds_1 = cfg.resolve_trial_seeds()
    base_seed_2, seeds_2 = cfg.resolve_trial_seeds()

    assert base_seed_1 == base_seed_2
    assert seeds_1 == seeds_2
    assert len(seeds_1) == 3


def test_mini_run():
    """Run 1 mechanism, 1 epsilon, 1 seed end-to-end."""
    from rpbench.config import DemoConfig, SplitSpec, PreprocessSpec, SeedBatchSpec
    from rpbench.runners.run_demo import run_demo

    cfg = DemoConfig(
        dataset="autompg",
        mechanisms=["Mech_RP"],
        task="OLSFromRelease",
        epsilon_grid=[2.0],
        delta_rule="1/n^2",
        seed_batch=SeedBatchSpec(mode="fixed", base_seed=0, count=1),
        split=SplitSpec(train_fraction=0.8, seed=42),
        preprocess=PreprocessSpec(),
        mech_params={"Mech_RP": {"r": 24}},
        output_root=tempfile.mkdtemp(),
    )

    records = run_demo(cfg)
    assert len(records) >= 2  # NonPrivate + 1 run
    for rec in records:
        assert "downstream_metrics" in rec
        assert "release_metrics" in rec


def test_mini_run_with_pois_wrapper():
    """Run the subsampled RP wrapper end-to-end."""
    from rpbench.config import DemoConfig, SplitSpec, PreprocessSpec, SeedBatchSpec
    from rpbench.runners.run_demo import run_demo

    cfg = DemoConfig(
        dataset="autompg",
        mechanisms=["Mech_RP_Pois"],
        task="OLSFromRelease",
        epsilon_grid=[2.0],
        delta_rule="1/n^2",
        seed_batch=SeedBatchSpec(mode="fixed", base_seed=0, count=1),
        split=SplitSpec(train_fraction=0.8, seed=42),
        preprocess=PreprocessSpec(),
        mech_params={"Mech_RP_Pois": {"r": 24, "q": 0.5}},
        output_root=tempfile.mkdtemp(),
    )

    records = run_demo(cfg)
    assert len(records) >= 2
    assert any(rec["mechanism"] == "Mech_RP_Pois" for rec in records)


def test_report_builder():
    """Test that the report builder can consume saved outputs."""
    import tempfile
    from pathlib import Path

    from rpbench.config import DemoConfig, SplitSpec, PreprocessSpec, SeedBatchSpec
    from rpbench.runners.run_demo import run_demo
    from rpbench.utils.io import save_jsonl, load_jsonl
    from rpbench.reporting.tables import build_summary_table
    from rpbench.reporting.figures import ols_plot_eps_vs_mse, plot_eps_vs_covariance_error
    from rpbench.reporting.summary import write_summary

    tmpdir = Path(tempfile.mkdtemp())

    cfg = DemoConfig(
        dataset="autompg",
        mechanisms=["Mech_RP"],
        task="OLSFromRelease",
        epsilon_grid=[1.0, 2.0],
        delta_rule="1/n^2",
        seed_batch=SeedBatchSpec(mode="fixed", base_seed=0, count=1),
        split=SplitSpec(train_fraction=0.8, seed=42),
        preprocess=PreprocessSpec(),
        mech_params={"Mech_RP": {"r": 24}},
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

    fig_path = report_dir / "ols_test_figure.png"
    ols_plot_eps_vs_mse(loaded, fig_path)
    assert fig_path.exists()

    cov_fig_path = report_dir / "covariance_test_figure.png"
    plot_eps_vs_covariance_error(loaded, cov_fig_path)
    assert cov_fig_path.exists()

    md_path = report_dir / "summary.md"
    write_summary(
        loaded,
        md_path,
        "ols_test_figure.png",
        "covariance_test_figure.png",
        "summary_table.csv",
    )
    assert md_path.exists()
