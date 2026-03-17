# rp-benchmark

RP-family differential privacy benchmark — WF-001 Stage 1.

Compares **Mech_RP** (NDIS paper, arxiv 2309.01243) and **Blocki12_JL**
(Blocki et al. 2012, arxiv 1204.2136) on the AutoMPG dataset using
OLS as the downstream task.

## Quick start (local)

```bash
# 1. Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Alternative) Use the install script
bash scripts/local_scripts/install_local.sh

# 3. Run the demo benchmark
python scripts/run_demo.py --config configs/demo/autompg.yaml

# 4. Build the report
python scripts/build_report.py \
  --input-root data/outputs/runs \
  --output-root data/outputs/reports

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

# 3. Run the demo
python scripts/run_demo.py --config configs/demo/autompg.yaml

# 4. Build the report
python scripts/build_report.py --input-root data/outputs/runs --output-root data/outputs/reports
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

1. Loads and preprocesses AutoMPG (7 features, `mpg` target).
2. For each mechanism × epsilon × trial seed:
   - Calibrates the mechanism under (ε, δ)-DP.
   - Uses the same explicit projection dimension `r` for both mechanisms.
   - Produces a private release of X^T X and X^T y.
   - Fits OLS from the release.
   - Evaluates test MSE and relative Frobenius error.
3. Saves row-level JSONL to `data/outputs/runs/demo_results.jsonl`.
4. The report builder aggregates results into a CSV table, PNG figure
   (ε vs test MSE), and markdown summary.

## Configuration

See `configs/demo/autompg.yaml` for the demo contract:

- **Mechanisms**: Mech_RP, Blocki12_JL
- **Epsilons**: 0.5, 1.0, 2.0, 4.0
- **Delta**: 1/n²
- **Shared sketch dimension**: r = 96
- **Seed batch**: fixed root seed 0, expanded to 5 trial seeds
- **Split**: 80/20 train/test, seed 42

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
  data/outputs/                  Generated outputs
```

## References

- NDIS paper: arxiv 2309.01243 (`docs/notes/ndis-paper.pdf`)
- Blocki et al. 2012: arxiv 1204.2136
- AutoMPG: UCI ML Repository, data_id=196
