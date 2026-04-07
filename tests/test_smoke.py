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


def test_mini_run_with_ptr_wrapper():
    """Run the PTR-based RP wrapper end-to-end."""
    from rpbench.config import DemoConfig, SplitSpec, PreprocessSpec, SeedBatchSpec
    from rpbench.runners.run_demo import run_demo

    # delta_rule 1e-6 → delta = 1e-6; splits must sum exactly to 1e-6.
    cfg = DemoConfig(
        dataset="autompg",
        mechanisms=["Mech_RP_PTR"],
        task="OLSFromRelease",
        epsilon_grid=[2.0],
        delta_rule="1e-6",
        seed_batch=SeedBatchSpec(mode="fixed", base_seed=0, count=1),
        split=SplitSpec(train_fraction=0.8, seed=42),
        preprocess=PreprocessSpec(),
        mech_params={"Mech_RP_PTR": {
            "r": 24,
            "tau": 0.5,
            "delta_r": 5e-7,
            "delta_t": 3e-7,
            "delta_ptr": 2e-7,
        }},
        output_root=tempfile.mkdtemp(),
    )

    records = run_demo(cfg)
    assert len(records) >= 2
    priv = [r for r in records if r["mechanism"] == "Mech_RP_PTR"]
    assert priv, "no Mech_RP_PTR records found"
    rec = priv[0]
    assert "downstream_metrics" in rec
    assert "release_metrics" in rec
    diag = rec["diagnostics"]
    assert "lambda_ptr" in diag
    assert "epsilon_T" in diag
    assert diag["lambda_ptr"] >= 0.0


def test_mini_run_with_sheffet_rp():
    """Run the native Sheffet RP mechanism end-to-end."""
    from rpbench.config import DemoConfig, SplitSpec, PreprocessSpec, SeedBatchSpec
    from rpbench.runners.run_demo import run_demo

    cfg = DemoConfig(
        dataset="synthetic_redundant_regression",
        mechanisms=["Mech_Sheffet_RP"],
        task="OLSFromRelease",
        epsilon_grid=[1.0],
        delta_rule="1e-6",
        seed_batch=SeedBatchSpec(mode="fixed", base_seed=0, count=1),
        split=SplitSpec(train_fraction=0.8, seed=42),
        preprocess=PreprocessSpec(),
        mech_params={"Mech_Sheffet_RP": {"r": 16}},
        output_root=tempfile.mkdtemp(),
        dataset_params={
            "feature_dim": 4,
            "n_prototypes": 8,
            "copies_per_prototype": 4,
            "generation_seed": 123,
            "poisson_diag_trials": 2,
        },
    )

    records = run_demo(cfg)
    assert len(records) >= 2
    priv = [r for r in records if r["mechanism"] == "Mech_Sheffet_RP"]
    assert priv, "no Mech_Sheffet_RP records found"
    rec = priv[0]
    assert "downstream_metrics" in rec
    assert "release_metrics" in rec
    assert "released_clean_sketch" in rec["diagnostics"]
    assert rec["downstream_metrics"]["test_mse"] >= 0.0


def test_synthetic_redundant_regression_registry():
    from rpbench.runners.run_demo import DATASET_REGISTRY

    assert "synthetic_redundant_regression" in DATASET_REGISTRY
    assert DATASET_REGISTRY["synthetic_redundant_regression"].name == "synthetic_redundant_regression"


def test_run_demo_nonprivate_only():
    from rpbench.config import DemoConfig, SplitSpec, PreprocessSpec, SeedBatchSpec
    from rpbench.runners.run_demo import run_demo

    cfg = DemoConfig(
        dataset="synthetic_redundant_regression",
        mechanisms=[],
        task="OLSFromRelease",
        epsilon_grid=[],
        delta_rule="1e-6",
        seed_batch=SeedBatchSpec(mode="fixed", base_seed=0, count=1),
        split=SplitSpec(train_fraction=0.8, seed=42),
        preprocess=PreprocessSpec(),
        mech_params={},
        output_root=tempfile.mkdtemp(),
        dataset_params={
            "feature_dim": 4,
            "n_prototypes": 8,
            "copies_per_prototype": 4,
            "generation_seed": 123,
            "poisson_diag_trials": 2,
        },
    )

    records = run_demo(cfg)
    assert len(records) == 1
    assert records[0]["mechanism"] == "NonPrivate"


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
    assert "prototype_retention_diagnostics" in b1.meta
    assert b1.meta["synthetic_params"]["generation_seed_from_config"] == 2026
    assert b1.meta["synthetic_params"]["generation_seed"] == 2026
    poisson_diag = b1.meta["poisson_subsampling_diagnostics"]["0.2"]
    assert "relative_opnorm_p90" in poisson_diag
    assert "whitened_relative_opnorm_p90" in poisson_diag


def test_synthetic_redundant_regression_supports_explicit_spectrum_and_weak_beta():
    from rpbench.config import PreprocessSpec, SplitSpec
    from rpbench.datasets.synthetic_redundant_regression import SyntheticRedundantRegressionAdapter

    adapter = SyntheticRedundantRegressionAdapter(
        feature_dim=6,
        n_prototypes=15,
        copies_per_prototype=20,
        cluster_noise=0.03,
        generation_seed=2027,
        prototype_eigenvalues=[1.5, 1.2, 0.9, 0.6, 0.4, 0.25],
        beta_mode="weak",
        beta_strength=1.4,
        poisson_diag_qs=[0.3],
        poisson_diag_trials=3,
    )
    bundle = adapter.load(SplitSpec(train_fraction=0.8, seed=7), PreprocessSpec())
    sp = bundle.meta["synthetic_params"]
    assert sp["prototype_eigenvalues"] == [1.5, 1.2, 0.9, 0.6, 0.4, 0.25]
    assert sp["beta_mode"] == "weak"
    assert 0.0 <= sp["beta_weak_half_mass"] <= 1.0
    assert "prototype_retention_per_trial" in bundle.meta["poisson_subsampling_diagnostics"]["0.3"]


def test_synthetic_redundant_regression_exposes_preclip_diagnostics():
    from rpbench.config import PreprocessSpec, SplitSpec
    from rpbench.datasets.synthetic_redundant_regression import SyntheticRedundantRegressionAdapter

    adapter = SyntheticRedundantRegressionAdapter(
        feature_dim=5,
        n_prototypes=20,
        copies_per_prototype=15,
        cluster_noise=0.08,
        label_noise=0.12,
        generation_seed=2028,
        poisson_diag_trials=3,
    )
    bundle = adapter.load(
        SplitSpec(train_fraction=0.8, seed=7),
        PreprocessSpec(clip_x=True, clip_x_bound=2.5, clip_y=True, clip_y_bound=1.2),
    )
    diag = bundle.meta["preprocess_diagnostics"]
    assert 0.0 <= diag["x_row_clip_fraction_train"] <= 1.0
    assert 0.0 <= diag["y_clip_fraction_train"] <= 1.0
    assert diag["x_row_norm_max_train_pre_clip"] >= diag["x_row_norm_p95_train_pre_clip"]
    assert diag["y_abs_max_train_pre_clip"] >= diag["y_abs_p95_train_pre_clip"]


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


def test_flight_registry():
    from rpbench.runners.run_demo import DATASET_REGISTRY

    assert "flight" in DATASET_REGISTRY
    assert DATASET_REGISTRY["flight"].name == "flight"


def test_flight_adapter_load():
    """FlightAdapter loads, preprocesses, and returns a bundle with d=1 and ~327k rows."""
    from unittest.mock import patch
    from pathlib import Path
    import tempfile
    import numpy as np
    import pandas as pd

    from rpbench.config import PreprocessSpec, SplitSpec
    from rpbench.datasets.flight import FlightAdapter

    # Build a tiny synthetic CSV that mimics the nycflights13 structure.
    rng = np.random.default_rng(0)
    n = 200
    dep = rng.normal(12.0, 40.0, size=n)
    arr = dep * 0.9 + rng.normal(0, 5.0, size=n)
    tiny_df = pd.DataFrame({"dep_delay": dep, "arr_delay": arr})

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "nycflights13_flights.csv"
        tiny_df.to_csv(csv_path)  # write with row-index column (mimics Rdatasets)

        adapter = FlightAdapter(cache_dir=tmpdir)
        # Patch _fetch_csv to return our pre-written file without a network call.
        with patch.object(adapter, "_fetch_csv", return_value=csv_path):
            bundle = adapter.load(
                SplitSpec(train_fraction=0.8, seed=7),
                PreprocessSpec(scale_x=True, clip_x=True, clip_x_bound=3.0,
                               clip_y=True, clip_y_bound=3.0),
            )

    assert bundle.X_train.shape[1] == 1, "expected d=1 (dep_delay only)"
    assert bundle.X_train.shape[0] + bundle.X_test.shape[0] == n
    assert bundle.meta["d"] == 1
    assert bundle.meta["feature_columns"] == ["dep_delay"]
    assert bundle.meta["target_column"] == "arr_delay"
    assert bundle.meta["C_X"] == 3.0
    assert bundle.meta["C_Y"] == 3.0

    pub = adapter.public_meta(bundle)
    assert pub["d"] == 1
    assert pub["d_aug"] == 2
    assert pub["n"] == bundle.X_train.shape[0]
    assert pub["l"] > 0


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
