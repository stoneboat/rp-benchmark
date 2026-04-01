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


def test_synthetic_redundant_regression_registry():
    from rpbench.runners.run_demo import DATASET_REGISTRY

    assert "synthetic_redundant_regression" in DATASET_REGISTRY
    assert DATASET_REGISTRY["synthetic_redundant_regression"].name == "synthetic_redundant_regression"


def test_synthetic_redundant_regression_deterministic():
    from rpbench.config import PreprocessSpec, SplitSpec
    from rpbench.datasets.synthetic_redundant_regression import SyntheticRedundantRegressionAdapter

    adapter = SyntheticRedundantRegressionAdapter(
        feature_dim=5,
        n_prototypes=12,
        copies_per_prototype=8,
        generation_seed=2026,
        poisson_diag_trials=3,
    )
    split = SplitSpec(train_fraction=0.8, seed=11)
    pre = PreprocessSpec()
    b1 = adapter.load(split, pre)
    b2 = adapter.load(split, pre)
    import numpy as np

    np.testing.assert_array_equal(b1.X_train, b2.X_train)
    np.testing.assert_array_equal(b1.y_train, b2.y_train)
    assert "geometry_diagnostics" in b1.meta
    assert "poisson_subsampling_diagnostics" in b1.meta
    assert "blocki12_jl_diagnostics" in b1.meta
    assert b1.meta["synthetic_params"]["generation_seed_from_config"] == 2026
    assert b1.meta["synthetic_params"]["generation_seed"] == 2026


def test_synthetic_redundant_regression_random_generation_seed_when_omitted():
    from rpbench.config import PreprocessSpec, SplitSpec
    from rpbench.datasets.synthetic_redundant_regression import SyntheticRedundantRegressionAdapter

    adapter = SyntheticRedundantRegressionAdapter(
        feature_dim=4,
        n_prototypes=8,
        copies_per_prototype=4,
        poisson_diag_trials=2,
    )
    split = SplitSpec(train_fraction=0.8, seed=11)
    pre = PreprocessSpec()
    b = adapter.load(split, pre)
    sp = b.meta["synthetic_params"]
    assert sp["generation_seed_from_config"] is None
    assert isinstance(sp["generation_seed"], int)


def test_synthetic_redundant_regression_mini_benchmark():
    """Small synthetic run: shapes OK for OLSFromRelease."""
    import tempfile

    from rpbench.config import DemoConfig, SplitSpec, PreprocessSpec, SeedBatchSpec
    from rpbench.runners.run_demo import run_demo

    cfg = DemoConfig(
        dataset="synthetic_redundant_regression",
        mechanisms=["Mech_RP", "Mech_RP_Pois"],
        task="OLSFromRelease",
        epsilon_grid=[2.0],
        delta_rule="1e-6",
        seed_batch=SeedBatchSpec(mode="fixed", base_seed=0, count=1),
        split=SplitSpec(train_fraction=0.8, seed=42),
        preprocess=PreprocessSpec(),
        mech_params={"Mech_RP": {"r": 24}, "Mech_RP_Pois": {"r": 24, "q": 0.5}},
        output_root=tempfile.mkdtemp(),
        dataset_params={
            "feature_dim": 6,
            "n_prototypes": 20,
            "copies_per_prototype": 12,
            "generation_seed": 123,
            "poisson_diag_trials": 4,
        },
    )

    records = run_demo(cfg)
    assert len(records) >= 2
    priv = [r for r in records if r["mechanism"] != "NonPrivate"]
    assert priv
    for r in priv:
        assert r["downstream_metrics"]["test_mse"] >= 0.0


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
