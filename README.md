# rp-benchmark

RP-family differential privacy benchmark — WF-001 Stage 1.

Compares **Mech_RP** (NDIS paper, arxiv 2309.01243) and **Blocki12_JL**
(Blocki et al. 2012, arxiv 1204.2136) on the AutoMPG dataset using
OLS as the downstream task.

## Quick start

```bash
# 1. Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Alternative) Use the install script
bash scripts/local_scripts/install_local.sh

# 3. Run the demo benchmark
python scripts/run_demo.py --config configs/demo/wf001_demo_autompg.yaml

# 4. Build the report
python scripts/build_report.py \
  --input-root data/outputs/runs \
  --output-root data/outputs/reports

# 5. Run tests
pytest -q
```

## What the demo does

1. Loads and preprocesses AutoMPG (7 features, `mpg` target).
2. For each mechanism × epsilon × seed:
   - Calibrates the mechanism under (ε, δ)-DP.
   - Produces a private release of X^T X and X^T y.
   - Fits OLS from the release.
   - Evaluates test MSE and relative Frobenius error.
3. Saves row-level JSONL to `data/outputs/runs/demo_results.jsonl`.
4. The report builder aggregates results into a CSV table, PNG figure
   (ε vs test MSE), and markdown summary.

## Configuration

See `configs/demo/wf001_demo_autompg.yaml` for the demo contract:

- **Mechanisms**: Mech_RP, Blocki12_JL
- **Epsilons**: 0.5, 1.0, 2.0, 4.0
- **Delta**: 1/n²
- **Seeds**: 0–4
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
