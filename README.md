# rp-benchmark

RP-family differential privacy benchmark — WF-001 Stage 1.

Compares RP-family mechanisms on regression datasets using OLS as the
downstream task. The repo currently includes public-data demos
(`autompg`, `bike_sharing`, `bike_sharing_redundant`) and a synthetic
redundant-regression demo (`synthetic_redundant_regression`).

## Quick start (local)

```bash
# 1. Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Alternative) Use the install script
bash scripts/local_scripts/install_local.sh

# 3. Run the demo benchmark (AutoMPG)
python scripts/run_demo.py --config configs/demo/autompg.yaml

# 3b. Run Bike Sharing benchmark
python scripts/run_demo.py --config configs/demo/bike_sharing.yaml

# 3c. Run Bike Sharing redundant-row benchmark
python scripts/run_demo.py --config configs/demo/bike_sharing_redundant.yaml

# 3d. Run synthetic redundant-regression benchmark
python scripts/run_demo.py --config configs/demo/synthetic_redundant_regression.yaml

# 4. Build the report for a specific run directory
python scripts/build_report.py \
  --input-root data/outputs/runs/synthetic_positive_q03 \
  --output-root reports/synthetic_positive_q03

# 5. Run tests
pytest -q
```

## Quick start (Purdue RCAC Bell cluster)

The cluster script uses **conda** and places the environment under `/tmp/python-venv/`
to avoid filling the home-directory quota.

```bash
# 1. Install (creates conda env at /tmp/python-venv/rp_benchmark_venv)
bash scripts/local_scripts/cluster_install_bell.sh

# 2. Activate the conda env in subsequent sessions
module load anaconda
eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/rp_benchmark_venv

# 3. Run the demo (AutoMPG)
python scripts/run_demo.py --config configs/demo/autompg.yaml

# 3b. Run Bike Sharing benchmark
python scripts/run_demo.py --config configs/demo/bike_sharing.yaml

# 3c. Run Bike Sharing redundant-row benchmark
python scripts/run_demo.py --config configs/demo/bike_sharing_redundant.yaml

# 3d. Run synthetic redundant-regression benchmark
python scripts/run_demo.py --config configs/demo/synthetic_redundant_regression.yaml

# 4. Build the report for a specific run directory
python scripts/build_report.py \
  --input-root data/outputs/runs/synthetic_positive_q03 \
  --output-root reports/synthetic_positive_q03
```

Notes:
- If you see `CondaToSNonInteractiveError`, accept the Anaconda channel ToS once:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## Notebook walkthrough (optional)

A step-by-step demo notebook is provided at `notebooks/test_demo.ipynb`.

If you registered the kernel via `scripts/local_scripts/cluster_install_bell.sh`, select
the kernel named **Python (rpbench)** in Jupyter and run the cells top-to-bottom.

## What the demo does

1. Loads and preprocesses the configured dataset.
2. For each mechanism × epsilon × trial seed:
   - Calibrates the mechanism under (ε, δ)-DP.
   - Uses the same explicit projection dimension `r` for both mechanisms.
   - Produces a private release of X^T X and X^T y.
   - Fits OLS from the release.
   - Evaluates test MSE and relative Frobenius error.
3. Saves row-level JSONL to the config's `output_root`, e.g.
   `data/outputs/runs/` for `autompg` and `bike_sharing`,
   `data/outputs/runs/bike_sharing_redundant/` for `bike_sharing_redundant`,
   and `data/outputs/runs/synthetic_positive_q03/` for
   `synthetic_redundant_regression`.
4. Within that directory, it writes a timestamped JSONL file
   `demo_results_YYYYMMDD-HHMMSS.jsonl` and refreshes
   `demo_results.jsonl` as the latest pointer.
5. The report builder reads one run directory at a time and writes timestamped
   CSV/PNG/MD outputs under the chosen `--output-root`, also refreshing stable
   latest filenames there.

## Configuration

See configs under `configs/demo/`:

- `autompg.yaml`
- `bike_sharing.yaml`
- `bike_sharing_redundant.yaml`
- `synthetic_redundant_regression.yaml`

Current demo outputs:

- `configs/demo/autompg.yaml` writes to `data/outputs/runs`
- `configs/demo/bike_sharing.yaml` writes to `data/outputs/runs`
- `configs/demo/bike_sharing_redundant.yaml` writes to `data/outputs/runs/bike_sharing_redundant`
- `configs/demo/synthetic_redundant_regression.yaml` writes to `data/outputs/runs/synthetic_positive_q03`

Notable config differences:

- `autompg.yaml`: mechanisms `Mech_RP`, `Mech_RP_Pois`, `Blocki12_JL`; `r=50`; `q=0.5`; 5 seeds
- `bike_sharing.yaml`: mechanisms `Mech_RP`, `Mech_RP_Pois`, `Blocki12_JL`; `r=100`; `q=0.8`; 5 seeds
- `bike_sharing_redundant.yaml`: dataset `bike_sharing_redundant`; `copies_per_row=15`; mechanisms `Mech_RP`, `Mech_RP_Pois`; `r=100`; `q=0.8`; 5 seeds
- `synthetic_redundant_regression.yaml`: dataset `synthetic_redundant_regression`; high-redundancy synthetic regime; mechanisms `Mech_RP`, `Mech_RP_Pois`; `r=20`; `q=0.3`; epsilon grid `[0.1, 0.25, 0.5, 1.0]`; 8 seeds

All current demo configs use:

- `delta_rule: 1e-6`
- 80/20 train/test split with seed `42`
- preprocessing with feature scaling and clipping

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
  --input-root data/outputs/runs \
  --output-root reports/autompg

python scripts/build_report.py \
  --input-root data/outputs/runs/bike_sharing_redundant \
  --output-root reports/bike_sharing_redundant

python scripts/build_report.py \
  --input-root data/outputs/runs/synthetic_positive_q03 \
  --output-root reports/synthetic_positive_q03
```

Each report directory contains:

- `summary_table_YYYYMMDD-HHMMSS.csv` and `summary_table.csv`
- `ols_plot_eps_vs_mse_YYYYMMDD-HHMMSS.png` and `ols_plot_eps_vs_mse.png`
- `covariance_plot_eps_vs_error_YYYYMMDD-HHMMSS.png` and `covariance_plot_eps_vs_error.png`
- `demo_summary_YYYYMMDD-HHMMSS.md` and `demo_summary.md`

## Repo structure

```
rp-benchmark/
  configs/demo/                  Config files
  docs/notes/                    Charter, preprocessing contract, references
  src/rpbench/                   Core package
    mechanisms/                  Mech_RP + baselines
    datasets/                    Dataset adapters
    tasks/                       Downstream tasks
    metrics/                     Release + downstream metrics
    reporting/                   Tables, figures, summary
    runners/                     Demo runner
    utils/                       I/O, seeding, linear algebra
  scripts/                       CLI entry points + install scripts
  tests/                         Smoke + release bundle tests
  data/outputs/                  Generated run outputs
  reports/                       Generated report outputs
```

## References

- NDIS paper: arxiv 2309.01243 (`docs/notes/ndis-paper.pdf`)
- Blocki et al. 2012: arxiv 1204.2136
- AutoMPG: UCI ML Repository, data_id=196
- Bike Sharing: OpenML Bike Sharing task (hourly/daily variants)
