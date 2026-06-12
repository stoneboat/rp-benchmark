# rp-benchmark

This repository is an anonymized research artifact for a paper under review at
IEEE S&P 2027. It supports the experiments and implementations described in
the full version of the NDIS paper while omitting author and institution
identifiers for review.

## Paper and Artifact Map

The full paper version included with this artifact is:

- [paper/ndis-paper-fullversion.pdf](paper/ndis-paper-fullversion.pdf)

Use the full version for the paper-facing locations of the main results:

| Paper item | Where to look |
| --- | --- |
| Figure 3 | RP mechanism experiment/result for the mechanism defined in Figure 1 of the paper; see the full version PDF. |
| Table 1 | BLR experiment discussed in Appendix C; see the full version PDF and `reports/ndis_gaussian/blr_summary.csv`. |
| Table 2 | GPR experiment discussed in Appendix C; see the full version PDF and `reports/ndis_gaussian/gpr_summary.csv`. |

## Repository Functionality

`rp-benchmark` supports empirical study of privacy mechanisms for statistical
learning. The repository has two complementary experiment paths:

- The original `rpbench` path: RP-style private releases of augmented regression
  data `[X | y]`, evaluated through the shared OLS benchmark runner and report
  builder.
- The `ndis_gaussian` path: a generic wrapper for algorithms that output a
  Gaussian pair `(mu, Sigma)`. The wrapper calibrates an additive covariance
  inflation `tau * I` from the algorithm's NDIS sensitivity bound, then samples
  the private release from `N(mu, Sigma + tau * I)`.

The `ndis_gaussian` package is intentionally separate from `rpbench`: it does
not import from `rpbench`, while the standalone demo scripts may reuse
`rpbench` dataset adapters.

Current implementation status:

- RP benchmark path: RP-based private releases for augmented regression data,
  comparison baselines, OLS-from-release evaluation, covariance-release metrics,
  plots, CSV tables, and markdown summaries.
- NDIS Gaussian-output path: `NDISGaussianWrapper`, shared calibration utilities,
  BLR and scalar-output GPR Gaussian-output algorithms, standalone demo scripts,
  and compact CSV summaries.
- Datasets: Auto MPG, OpenML Bike Sharing, Kaggle Bike Sharing Demand, NYC
  flight delays, Tecator, synthetic redundant regression, Wisconsin Diagnostic
  Breast Cancer, and Linnerud.

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

# 4. Run the NDIS Gaussian-output standalone demos
python scripts/run_blr_ndis_demo.py --config configs/blr_demo/breast_cancer.yaml
python scripts/run_gpr_ndis_demo.py --config configs/gpr_demo/linnerud.yaml

# 5. Build a report for an RP benchmark run directory
python scripts/build_report.py \
  --input-root data/outputs/runs/synthetic_redundant_regression \
  --output-root reports/synthetic_redundant_regression

# 6. Run tests
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

# 4. Run the NDIS Gaussian-output standalone demos
python scripts/run_blr_ndis_demo.py --config configs/blr_demo/breast_cancer.yaml
python scripts/run_gpr_ndis_demo.py --config configs/gpr_demo/linnerud.yaml

# 5. Build a report for an RP benchmark run directory
python scripts/build_report.py \
  --input-root data/outputs/runs/synthetic_redundant_regression \
  --output-root reports/synthetic_redundant_regression
```

If you see `CondaToSNonInteractiveError`, accept the Anaconda channel ToS once:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## Notebook walkthroughs

The `notebooks/` directory contains optional NDIS audit notebooks:

- `notebooks/gaussian_whitebox_audit.ipynb`
- `notebooks/gaussian_parametric_blackbox_audit.ipynb`

If you registered the kernel via `scripts/local_scripts/cluster_install_bell.sh`,
select the kernel named **Python (rpbench)** in Jupyter and run the cells
top-to-bottom.

## What a benchmark run does

The original `scripts/run_demo.py` path is the RP/OLS benchmark path. It:

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

The standalone `scripts/run_blr_ndis_demo.py` and `scripts/run_gpr_ndis_demo.py`
paths exercise the newer generic NDIS Gaussian-output wrapper. They:

1. Load the configured dataset adapter and public preprocessing bounds.
2. Fit a Gaussian-output algorithm, currently BLR or scalar-output GPR.
3. Calibrate `tau_star` with `NDISGaussianWrapper` from the algorithm-specific
   sensitivity formulas.
4. Release samples from `N(mu, Sigma + tau_star * I)` for the configured seeds.
5. Save raw JSONL records under each config's `output_root`.

## Configured Demos

The original RP/OLS benchmark configs live under `configs/demo/`.

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

The NDIS Gaussian-output demos use standalone scripts and configs:

| Config | Script | Dataset | Algorithm | Main purpose |
| --- | --- | --- | --- | --- |
| `configs/blr_demo/breast_cancer.yaml` | `scripts/run_blr_ndis_demo.py` | Wisconsin Diagnostic Breast Cancer | Bayesian logistic regression Gaussian-output algorithm | classification parameter release via the generic NDIS wrapper |
| `configs/gpr_demo/linnerud.yaml` | `scripts/run_gpr_ndis_demo.py` | Linnerud, scalar target `Pulse` by default in this config | single-query scalar-output GPR | regression prediction release via the generic NDIS wrapper |

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
| `configs/blr_demo/breast_cancer.yaml` | `data/outputs/blr_ndis_runs` |
| `configs/gpr_demo/linnerud.yaml` | `data/outputs/gpr_ndis_runs` |

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
- `breast_cancer`: Wisconsin Diagnostic Breast Cancer classification adapter
  for the BLR standalone demo.
- `linnerud`: sklearn Linnerud adapter with configurable scalar target selection
  for the GPR standalone demo.

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

NDIS Gaussian-output framework:

- `GaussianOutputAlgorithm`: interface for algorithms that produce `(mu, Sigma)`
  and expose an NDIS sensitivity bound as a function of covariance inflation
  `tau`.
- `NDISGaussianWrapper`: generic Figure-5-style wrapper that calibrates
  `tau_star` and samples from `N(mu, Sigma + tau_star * I)`.
- `BLRGaussianOutput`: Bayesian logistic regression Gaussian-output algorithm
  for the breast-cancer classification demo.
- `GPRGaussianOutput`: scalar-output, fixed-public-query Gaussian process
  regression algorithm for the Linnerud demo.
- `RBFKernel`: unit-amplitude RBF kernel helper for the GPR path.

Tasks and metrics:

- `OLSFromRelease`: solves an OLS model from private `X^T X` and `X^T y`
  releases, then reports test MSE.
- Release metric: `rel_fro_xtx = ||X^T X - X^T X_hat||_F / ||X^T X||_F`.
- BLR demo metric: non-private and private test accuracy/log-loss.
- GPR demo metric: non-private and private absolute error at one public query.

## Reports

Build RP benchmark reports from a specific run directory:

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

The NDIS Gaussian-output demos currently use compact CSV summaries under:

- `reports/ndis_gaussian/blr_summary.csv`
- `reports/ndis_gaussian/gpr_summary.csv`

## Repo Structure

```text
rp-benchmark/
  configs/demo/                  RP/OLS benchmark configs
  configs/blr_demo/              standalone BLR + NDIS demo config
  configs/gpr_demo/              standalone GPR + NDIS demo config
  notebooks/                     optional notebook walkthroughs
  src/rpbench/                   core package
    mechanisms/                  RP mechanisms and baselines
    datasets/                    dataset adapters
    tasks/                       downstream tasks
    metrics/                     release and downstream metrics
    reporting/                   tables, figures, summaries
    runners/                     benchmark runner
    utils/                       I/O, seeding, linear algebra
  src/ndis_gaussian/             generic NDIS Gaussian-output wrapper package
    calibration.py               delta upper bounds and tau* search
    wrapper.py                   NDISGaussianWrapper
    blr/                         BLR Gaussian-output algorithm
    gpr/                         scalar-output GPR Gaussian-output algorithm
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
- Kaggle Bike Sharing Demand: local `bike_train.csv` supplied outside this
  artifact.
- Flight delays: nycflights13 flights CSV from the Rdatasets mirror.
- Tecator: OpenML `name=Tecator`, `version=1`.
- Wisconsin Diagnostic Breast Cancer: `sklearn.datasets.load_breast_cancer`.
- Linnerud: `sklearn.datasets.load_linnerud`.
