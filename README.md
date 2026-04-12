# rp-benchmark

`rp-benchmark` supports empirical study of random projection (RP) as a
privacy-preserving approach for statistical learning. The repo is intended as a
small, configurable benchmark where users can plug in mechanisms, datasets, and
downstream tasks, then evaluate RP-based algorithms under a common experiment
runner and reporting pipeline.

Current implementation status:

- Mechanism family: RP-based private releases for augmented regression data
  `[X | y]`, plus comparison baselines.
- Datasets: Auto MPG, OpenML Bike Sharing, Kaggle Bike Sharing Demand, NYC
  flight delays, Tecator, and synthetic redundant regression.
- Downstream task: OLS fitted from released sufficient statistics
  (`OLSFromRelease`).
- Release-quality metric: relative Frobenius error of the covariance block
  `X^T X`.
- Reporting: aggregate CSV tables, OLS test-MSE plots, covariance-error plots,
  and markdown summaries.

The framework is broader than the current demos: additional dataset adapters,
mechanisms, metrics, and downstream tasks can be added behind the same config
and runner abstractions.

## Quick start (local)

```bash
# 1. Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Alternative: use the install script
bash scripts/local_scripts/install_local.sh

# 3. Run a benchmark config
python scripts/run_demo.py --config configs/demo/autompg.yaml
python scripts/run_demo.py --config configs/demo/bike_sharing.yaml
python scripts/run_demo.py --config configs/demo/synthetic_redundant_regression.yaml

# 4. Build a report for a run directory
python scripts/build_report.py \
  --input-root data/outputs/runs/synthetic_redundant_regression \
  --output-root reports/synthetic_redundant_regression

# 5. Run tests
pytest -q
```

Some configs require data downloads or local files:

- `autompg`, `bike_sharing`, and `tecator` use OpenML through
  `sklearn.datasets.fetch_openml` and cache under `data/cache/`.
- `flight` downloads the nycflights13 flights CSV from the Rdatasets mirror and
  caches it under `data/cache/`.
- `bike_sharing_kaggle` expects Kaggle's Bike Sharing Demand training CSV at
  `data/raw/bike_sharing_kaggle/bike_train.csv`, unless
  `dataset_params.csv_path` is overridden.

## Quick start (cluster)

The cluster script uses **conda** and places the environment under
`/tmp/python-venv/` to avoid filling the home-directory quota.

```bash
# 1. Install (creates conda env at /tmp/python-venv/rp_benchmark_venv)
bash scripts/local_scripts/cluster_install_bell.sh

# 2. Activate the conda env in subsequent sessions
module load anaconda
eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/rp_benchmark_venv

# 3. Run a benchmark config
python scripts/run_demo.py --config configs/demo/autompg.yaml
python scripts/run_demo.py --config configs/demo/bike_sharing.yaml
python scripts/run_demo.py --config configs/demo/synthetic_redundant_regression.yaml

# 4. Build a report for a run directory
python scripts/build_report.py \
  --input-root data/outputs/runs/synthetic_redundant_regression \
  --output-root reports/synthetic_redundant_regression
```

If you see `CondaToSNonInteractiveError`, accept the Anaconda channel ToS once:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## Notebook walkthrough 

A step-by-step demo notebook is provided at `notebooks/test_demo.ipynb`.

If you registered the kernel via `scripts/local_scripts/cluster_install_bell.sh`,
select the kernel named **Python (rpbench)** in Jupyter and run the cells
top-to-bottom.

## What a benchmark run does

1. Loads the configured dataset adapter and applies that adapter's preprocessing
   contract.
2. Builds public metadata used for calibration, including train size,
   feature dimension, augmented dimension, and public row norm bounds.
3. Records a non-private OLS baseline on the train/test split.
4. For each mechanism, epsilon value, and trial seed:
   - calibrates the mechanism under the config's `(epsilon, delta)` rule,
   - releases private sufficient statistics from augmented data `[X | y]`,
   - fits OLS from the released `X^T X` and `X^T y` blocks,
   - evaluates test MSE,
   - measures relative Frobenius error for the released `X^T X` block,
   - stores runtime, diagnostics, provenance, and metrics as JSONL records.
5. Writes timestamped row-level results plus `demo_results.jsonl` under the
   config's `output_root`.
6. The report builder reads one run directory and writes timestamped CSV, PNG,
   and markdown outputs under the selected report directory.

## Configured Demos

All demo configs live under `configs/demo/`.

| Config | Dataset | Mechanisms | Main purpose |
| --- | --- | --- | --- |
| `autompg.yaml` | Auto MPG from OpenML `data_id=196` | `Mech_RP`, `Mech_RP_Pois`, `Blocki12_JL` | small public-data OLS benchmark |
| `autompg_ptr.yaml` | Auto MPG from OpenML `data_id=196` | `Mech_RP`, `Mech_RP_PTR` | PTR comparison on Auto MPG |
| `bike_sharing.yaml` | OpenML Bike Sharing `data_id=42713` | `Mech_RP`, `Mech_RP_Pois`, `Mech_RP_PTR` | OpenML bike-sharing OLS benchmark |
| `bike_sharing_kaggle.yaml` | local Kaggle Bike Sharing Demand CSV | `Mech_RP`, `Mech_Sheffet_RP`, `Mech_Improved_Sheffet_RP`, `Mech_Modified_GaussMix` | GaussMix-style bike-sharing comparison |
| `flight.yaml` | nycflights13 flights from Rdatasets | `Mech_RP`, `Mech_RP_Pois` | flight-delay regression benchmark |
| `synthetic_redundant_regression.yaml` | synthetic redundant-cluster regression | `Mech_RP`, `Mech_RP_Pois`, `Mech_Sheffet_RP`, `Mech_Improved_Sheffet_RP`, `Mech_Modified_GaussMix` | controlled redundant-regression comparison |
| `synthetic_ptr.yaml` | synthetic redundant-cluster regression | `Mech_RP`, `Mech_RP_PTR` | PTR comparison in a synthetic regime |
| `tecator.yaml` | Tecator from OpenML `name=Tecator`, `version=1` | `Mech_RP`, `Mech_Improved_Sheffet_RP`, `Mech_Modified_GaussMix` | GaussMix-style Tecator comparison |

Current output roots:

| Config | `output_root` |
| --- | --- |
| `autompg.yaml` | `data/outputs/runs` |
| `autompg_ptr.yaml` | `data/outputs/runs/autompg_ptr` |
| `bike_sharing.yaml` | `data/outputs/runs/bike_sharing` |
| `bike_sharing_kaggle.yaml` | `data/outputs/runs/bike_sharing_kaggle` |
| `flight.yaml` | `data/outputs/runs/flight` |
| `synthetic_redundant_regression.yaml` | `data/outputs/runs/synthetic_redundant_regression` |
| `synthetic_ptr.yaml` | `data/outputs/runs/synthetic_ptr` |
| `tecator.yaml` | `data/outputs/runs/tecator` |

## Implemented Components

Datasets:

- `autompg`: UCI Auto MPG via OpenML `data_id=196`.
- `bike_sharing`: OpenML Bike Sharing Demand, `data_id=42713`, numeric
  predictors only.
- `bike_sharing_kaggle`: local Kaggle Bike Sharing Demand CSV with
  GaussMix-style feature engineering and normalization.
- `flight`: nycflights13 flight delays from the Rdatasets mirror; target is
  `arr_delay` and feature is `dep_delay`.
- `synthetic_redundant_regression`: configurable redundant-cluster synthetic
  regression generator.
- `tecator`: OpenML Tecator with GaussMix-style split and normalization.

Mechanisms:

- `Mech_RP`: NDIS-based random projection mechanism.
- `Mech_RP_Pois`: Poisson-subsampled wrapper around `Mech_RP`.
- `Mech_RP_PTR`: PTR-based wrapper around `Mech_RP`.
- `Mech_Sheffet_RP`: Sheffet-style Gaussian RP release.
- `Mech_Improved_Sheffet_RP`: improved Sheffet-style release using the
  GaussMix sigma calibration.
- `Mech_Modified_GaussMix`: native benchmark implementation of the modified
  GaussMix release.
- `Blocki12_JL`: Johnson-Lindenstrauss covariance-estimation baseline from
  Blocki et al. 2012.

Tasks and metrics:

- `OLSFromRelease`: solves an OLS model from private `X^T X` and `X^T y`
  releases, then reports test MSE.
- Release metric: `rel_fro_xtx = ||X^T X - X^T X_hat||_F / ||X^T X||_F`.

## Reports

Build reports from a specific run directory:

```bash
python scripts/build_report.py \
  --input-root <run-output-root> \
  --output-root <report-output-root>
```

Examples:

```bash
python scripts/build_report.py \
  --input-root data/outputs/runs/bike_sharing_kaggle \
  --output-root reports/bike_sharing_kaggle

python scripts/build_report.py \
  --input-root data/outputs/runs/tecator \
  --output-root reports/tecator

python scripts/build_report.py \
  --input-root data/outputs/runs/synthetic_redundant_regression \
  --output-root reports/synthetic_redundant_regression
```

Each report directory contains timestamped artifacts:

- `summary_table_YYYYMMDD-HHMMSS.csv`
- `ols_plot_eps_vs_mse_YYYYMMDD-HHMMSS.png`
- `covariance_plot_eps_vs_error_YYYYMMDD-HHMMSS.png`
- `demo_summary_YYYYMMDD-HHMMSS.md`

## Repo Structure

```text
rp-benchmark/
  configs/demo/                  benchmark configs
  notebooks/                     optional notebook walkthroughs
  src/rpbench/                   core package
    mechanisms/                  RP mechanisms and baselines
    datasets/                    dataset adapters
    tasks/                       downstream tasks
    metrics/                     release and downstream metrics
    reporting/                   tables, figures, summaries
    runners/                     benchmark runner
    utils/                       I/O, seeding, linear algebra
  scripts/                       CLI entry points and install scripts
  tests/                         smoke and release-bundle tests
  data/cache/                    generated dataset cache, ignored by git
  data/raw/                      local raw data, ignored by git
  data/outputs/                  generated run outputs, ignored by git
  reports/                       generated report outputs
```

`docs/` is intentionally ignored by git for local notes and publication-facing
documentation that should not be included when this benchmark repo is pushed.

## References

- NDIS RP mechanism: arXiv:2309.01243.
- Blocki, Blum, Datta, and Sheffet (2012): arXiv:1204.2136.
- GaussMix mechanism: The Gaussian Mixing Mechanism and the reference
  `omrilev1/GaussMix` linear-regression implementation.
- Auto MPG: UCI ML Repository via OpenML, `data_id=196`.
- Bike Sharing: OpenML Bike Sharing Demand, `data_id=42713`.
- Kaggle Bike Sharing Demand: local `bike_train.csv` supplied by the user.
- Flight delays: nycflights13 flights CSV from the Rdatasets mirror.
- Tecator: OpenML `name=Tecator`, `version=1`.
