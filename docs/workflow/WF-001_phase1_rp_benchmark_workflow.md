# WF-001 — Phase-1 RP Benchmark Repo Workflow

**Workflow ID:** WF-001  
**Title:** Phase-1 RP-family benchmarking repo and execution workflow  
**Project linkage:** PROJ-004 (RQ2-first, RP-family first), using PROJ-099 as the archived predecessor implementation/theory source  
**Status:** draft for registration  
**Date:** 2026-03-17  
**Memo alias:** operationalizes the memo-level workflow idea `WF-RQ3-RP-BENCH` as a canonical phase-1 workflow

---

## 1) Purpose and phase boundary

WF-001 is the **phase-1-only** workflow for building a git repo and execution harness that can answer, in a reproducible way:

1. how `Mech_RP` compares with published RP/sketch DP baselines,
2. whether `Mech_RP^Pois` and `Mech_RP^PTR` improve the privacy–utility tradeoff in the regimes they are designed for,
3. and which mechanism wins under which data geometry.

This workflow is **not** the full “all Gaussian-output mechanisms” agenda.  
It is the RP-family benchmark substrate that should be completed first and then re-used in a later phase.

---

## 2) What this workflow must produce

WF-001 has four concrete deliverables.

### D1. Repo design spec
A repo structure that follows the reusable shape from `REPRODUCTION_MEMO`, but is adapted for Purdue RCAC cluster use rather than a purely local/laptop workflow.

### D2. Plug-and-play benchmark infrastructure
A common interface for:
- mechanisms,
- datasets,
- downstream tasks,
- metrics,
- runners,
- and reporting.

### D3. Inventory and normalization note
A written inventory of:
- mechanisms to benchmark,
- downstream tests,
- datasets,
- synthetic regime studies,
- and testing parameters,
with source URLs.

### D4. Reporting / presentation design
A lightweight but standard report pipeline that converts run outputs into:
- release-level leaderboards,
- downstream-task leaderboards,
- regime maps,
- and figure-ready outputs.

---

## 3) How WF-001 answers the scientific questions

| Scientific question | Repo object that answers it | Primary output |
|---|---|---|
| How does `Mech_RP` compare with RP/sketch DP baselines? | benchmark runner + baseline package + release/downstream leaderboards | privacy–utility curves and leaderboard tables |
| Do `Mech_RP^Pois` and `Mech_RP^PTR` help in the regimes they target? | synthetic regime generators + geometry metrics + mechanism diagnostics | regime-conditioned win/loss tables and heatmaps |
| Which mechanism wins under which data geometry? | geometry metric suite + aggregated report builder | “winner by geometry” summary figure and memo table |

---

## 4) Design principles

1. **Phase 1 only.** RP-family first; broader Gaussian-output benchmarking later.
2. **Repo shape follows `REPRODUCTION_MEMO`.** Keep `src/`, one canonical CLI path, `scripts/local_scripts/`, `Notebook/`, and metadata-rich outputs.
3. **Purdue-ready from day one.** Local setup must be cluster-aware and Slurm-aware, not bolted on later.
4. **Common interfaces first.** Every mechanism should pass through the same benchmark harness.
5. **Demo-first construction.** The first checked-in version is intentionally small.
6. **Two leaderboards.** One for release-level utility, one for downstream-task utility.
7. **Historical vs primary vs optional comparators are labeled explicitly.**
8. **Geometry is a first-class object.** The benchmark should explain *why* a mechanism wins, not just record that it won.

---

## 5) Goal 1 — repo structure design

### 5.1 Base shape inherited from `REPRODUCTION_MEMO`

The reusable pattern we keep:

- importable logic under `src/`,
- one canonical command-line runner,
- machine-specific setup under `scripts/local_scripts/`,
- notebooks as demos/explanations only,
- local state under `data/`,
- outputs saved with metadata.

### 5.2 Purdue/RCAC adaptation

The main adaptation is that this repo should assume:

- environment setup happens on Purdue RCAC systems,
- jobs are submitted with Slurm,
- large temporary inputs/outputs live in scratch,
- and the “local install script” becomes a **cluster install script contract**.

Because the user’s actual Purdue template script is **not yet provided**, WF-001 specifies the contract for that script, but leaves the exact module names / account flags / cluster-specific defaults to be filled in once the template arrives.
 
**Update (template now available):** use `demo_cluster_install_bell.sh` in this folder as the reference implementation style for the Purdue/RCAC environment bootstrap. The workflow below keeps the contract language, but also provides a concrete environment script template derived from that example.

### 5.3 Recommended repo tree

```text
rp-benchmark/
  README.md
  requirements.txt
  .gitignore

  docs/
    workflow/
      WF-001_phase1_rp_benchmark.md
    notes/
      comparator_normalization.md
      preprocessing_contract.md
      benchmark_charter.md
      ndis-paper.pdf
      

  configs/
    demo/
      wf001_demo_autompg.yaml
    benchmark/
      phase1_small.yaml
      phase1_core.yaml
      phase1_full.yaml
    cluster/
      purdue_cpu.yaml
      purdue_gpu_optional.yaml

  src/
    rpbench/
      __init__.py

      config/
        schema.py
        defaults.py
        registry.py

      mechanisms/
        base.py
        rp_ndis.py
        rp_pois.py
        rp_ptr.py

        baselines/
          blocki12_jl.py
          sheffet17_ols.py
          sheffet19_old_tech.py
          adassp.py
          lev25_gaussian_mixing.py
          dd_ssp.py
          dpgd_linear_regression.py
          ihm_2026.py

      datasets/
        base.py
        autompg.py
        communities_crime.py
        tecator.py
        wine_quality.py
        bike_sharing.py
        boston_housing_legacy.py

        synthetic/
          high_leverage.py
          ill_conditioned.py
          redundant_rows.py
          favorable_eigenvalue.py
          high_residual.py
          small_theta_star.py
          low_leverage_well_conditioned.py

      tasks/
        base.py
        ols_from_release.py
        ridge_from_release.py

      metrics/
        release.py
        downstream.py
        geometry.py
        runtime.py

      runners/
        single_run.py
        sweep.py
        manifest.py

      reporting/
        tables.py
        figures.py
        dashboard.py

      utils/
        io.py
        seeding.py
        linear_algebra.py
        privacy_conversion.py
        logging.py

  scripts/
    run_demo.py
    run_benchmark.py
    make_manifest.py
    build_report.py

    local_scripts/
      install_purdue_cluster.sh
      smoke_test_env.sh
      sync_results_from_scratch.sh

    slurm/
      submit_demo.sbatch
      submit_grid.sbatch
      submit_array.sh

  Notebook/
    demo_phase1.ipynb

  tests/
    unit/
    integration/
    smoke/

  data/
    raw/
    cache/
    processed/
    outputs/
      runs/
      tables/
      figures/
      logs/

  reports/
    examples/
      wf001_demo_privacy_utility.png
      wf001_demo_summary.md
```

### 5.4 File responsibilities

### Top level
- `README.md`: one-page description, install path, demo command, benchmark command, output locations.
- `requirements.txt`: direct dependencies only; keep short and explicit.
- `.gitignore`: ignore `data/`, caches, logs, large outputs.

### `docs/`
Use this for workflow-level and benchmark-level frozen notes.  
Minimum required docs:
- benchmark charter,
- comparator normalization note,
- preprocessing contract.

### `configs/`
All benchmark settings live here, never hard-coded in scripts.

Suggested config split:
- `demo/`: the tiny end-to-end smoke path,
- `benchmark/`: real experiment grids,
- `cluster/`: RCAC resource presets.

### `src/rpbench/`
This is the reusable codebase.

### `scripts/`
This is the executable surface.
- `run_demo.py`: one command that proves the repo works.
- `run_benchmark.py`: canonical experiment entry point.
- `make_manifest.py`: materializes sweep grids into a manifest file for Slurm arrays.
- `build_report.py`: consumes results and produces tables/figures.

### `scripts/local_scripts/`
Purdue-specific environment and syncing logic only.
No benchmarking logic should live here.

### `scripts/slurm/`
RCAC job wrappers only.
No privacy math here.

### `Notebook/`
A pure-Python walkthrough that imports from `src/`.
No notebook-only logic.

### `tests/`
Three layers:
- `unit/`: mechanism calibration / metric functions,
- `integration/`: one full run through the harness,
- `smoke/`: cluster-safe tiny tests.

### 5.5 Purdue-specific script contract

The repo should define `scripts/local_scripts/install_purdue_cluster.sh` as a **contract-driven** script with the following responsibilities:

1. load the cluster-supported Python/Conda module,
2. create or reuse the benchmark environment,
3. install `requirements.txt`,
4. verify Python imports for `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`,
5. create the repo’s scratch-side working directories under `$RCAC_SCRATCH`,
6. register a Jupyter kernel (optional but recommended),
7. print canonical demo and Slurm submission commands,
8. be idempotent on rerun.

The script should **not** hard-code a specific cluster name until the Purdue template is supplied.

#### 5.5.1 Reference example (Bell-style cluster install)

Use `artifacts/projects/PROJ-004/RQ2/demo_cluster_install_bell.sh` as the *shape* reference:
- module-based conda bootstrap (`module load anaconda` then `eval "$(conda shell.bash hook)"`),
- env-as-prefix (not env-by-name) for reproducibility in non-interactive shells,
- retry logic / SSL mitigation hooks (cluster networks often break TLS),
- optional Jupyter kernel registration with a kernel.json that points to the prefix Python.

#### 5.5.2 Purdue/RCAC environment script template (derived from the reference)

This template is meant to live in the benchmark repo as `scripts/local_scripts/install_purdue_cluster.sh`.
It is intentionally parameterized (module names, scratch path) but mirrors the reference script’s behavior.

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[install] RP benchmark environment bootstrap (Purdue/RCAC)"

# ---- 0) Cluster modules (edit for your RCAC system) ----
# Examples you might swap in depending on cluster:
#   module load anaconda
module load anaconda

# Initialize conda in this non-interactive shell
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "[install] conda not found after loading module" >&2
  exit 1
fi

# ---- 1) Prefix-based env location (prefer scratch) ----
: "${RCAC_SCRATCH:=$HOME}"
ENV_ROOT="${RCAC_SCRATCH}/python-venv"
mkdir -p "${ENV_ROOT}"

# Choose a stable prefix path (idempotent)
CONDA_ENV_PATH="${ENV_ROOT}/rpbench_venv"

if [[ -d "${CONDA_ENV_PATH}" ]]; then
  echo "[install] Using existing env prefix: ${CONDA_ENV_PATH}"
else
  echo "[install] Creating env prefix: ${CONDA_ENV_PATH}"
  conda create --prefix "${CONDA_ENV_PATH}" python=3.11 -y
fi

conda activate "${CONDA_ENV_PATH}"

# ---- 2) Optional: mitigate SSL issues (uncomment only if needed) ----
# conda update -n base -c defaults conda --yes 2>/dev/null || true
# conda config --set ssl_verify false
# export REQUESTS_CA_BUNDLE=""
# export CURL_CA_BUNDLE=""

# ---- 3) Install dependencies ----
python -m pip install --upgrade pip
pip install -r requirements.txt

# ---- 4) Smoke imports (fast fail) ----
python - <<'PY'
import numpy, scipy, pandas, sklearn, matplotlib
print("[install] imports ok")
PY

# ---- 5) Scratch-side working dirs ----
RUN_ROOT="${RCAC_SCRATCH}/rpbench"
mkdir -p "${RUN_ROOT}/cache" "${RUN_ROOT}/manifests" "${RUN_ROOT}/outputs"
echo "[install] scratch run root: ${RUN_ROOT}"

# ---- 6) Jupyter kernel ----
python -m ipykernel install --user --name=rpbench --display-name "rpbench (rcac)" || true

echo "[install] done"
echo "[next] Demo (interactive): python scripts/run_demo.py --config configs/demo/wf001_demo_autompg.yaml"
echo "[next] Slurm: sbatch scripts/slurm/submit_demo.sbatch"
```

Notes:
- The benchmark repo should treat this as **the** canonical entrypoint for interactive setup.
- Keep it idempotent; reruns should never break an existing working env.

### 5.6 RCAC storage contract

Recommended storage split:

- repo + configs + code: home/depot (persistent),
- raw temporary downloads + cache + manifests + run outputs: `$RCAC_SCRATCH`,
- final frozen result bundles copied back to persistent storage.

This prevents the repo from assuming that scratch is durable.

### 5.7 Slurm contract

Recommended execution pattern:

1. `make_manifest.py` writes a row-wise CSV/JSONL manifest.
2. `submit_array.sh` submits a Slurm array job.
3. each array row calls `run_benchmark.py --manifest ... --row-id $SLURM_ARRAY_TASK_ID`
4. outputs are written to one run directory per row.

Recommended run directory shape:

```text
data/outputs/runs/<run_id>/
  config_snapshot.yaml
  metadata.json
  metrics.json
  release_bundle_meta.json
  stdout.log
  stderr.log
```

---

## 6) Goal 2 — plug-and-play benchmark infrastructure

### 6.1 Core interfaces

### `DatasetAdapter`
Responsibilities:
- load raw data,
- apply the shared preprocessing contract,
- expose public metadata `(n, d, C_X, C_Y, split seed, dataset id)`,
- compute geometry diagnostics used by regime analysis.

Suggested interface:

```python
class DatasetAdapter:
    name: str

    def load(self, split_spec):
        ...
    def public_meta(self):
        ...
    def geometry(self, train_data):
        ...
```

### `Mechanism`
Responsibilities:
- receive `PrivacySpec` + public metadata,
- calibrate internal parameters,
- release a private artifact,
- emit diagnostics for later analysis.

```python
class Mechanism:
    name: str

    def calibrate(self, privacy_spec, public_meta):
        ...
    def release(self, train_data, seed):
        ...
    def diagnostics(self):
        ...
```

### `ReleaseBundle`
A common return object so every mechanism can be post-processed uniformly.

Required fields:

```python
@dataclass
class ReleaseBundle:
    mechanism_name: str
    release_kind: str
    xtx_hat: object | None
    xty_hat: object | None
    sketch: object | None
    beta_hat: object | None
    calibration: dict
    diagnostics: dict
    runtime_sec: float
```

### `Task`
Consumes a `ReleaseBundle` and produces a downstream fitted object and predictions.

```python
class Task:
    name: str

    def fit_from_release(self, release_bundle, train_meta):
        ...
    def evaluate(self, fitted_obj, test_data):
        ...
```

### `MetricSuite`
Should expose:
- release-level metrics,
- downstream metrics,
- geometry metrics,
- runtime/memory diagnostics.

### `ExperimentRunner`
Owns:
- seed expansion,
- grid construction,
- logging,
- result saving,
- checkpoint-safe execution.

### 6.2 Required config objects

At minimum define:

- `PrivacySpec(epsilon, delta, adjacency, notion)`
- `SplitSpec(train_fraction, seed, stratify=None)`
- `PreprocessSpec(scale_x, scale_y, clip_x, clip_y, missing_policy)`
- `MechanismSpec(name, params)`
- `TaskSpec(name, params)`
- `RunnerSpec(num_seeds, output_root, fail_policy)`
- `ReportSpec(metric_primary, metric_secondary, figure_kinds)`

### 6.3 Result schema

Each completed run should log:

```json
{
  "run_id": "...",
  "dataset": "...",
  "mechanism": "...",
  "task": "...",
  "epsilon": 1.0,
  "delta": 1e-6,
  "seed": 3,
  "release_metrics": {...},
  "downstream_metrics": {...},
  "geometry": {...},
  "runtime": {...},
  "provenance": {...}
}
```

The aggregate format should be `parquet` or line-delimited JSON, plus a human-readable markdown summary.

---

## 7) Demo-first construction (the first git version)

The first checked-in version should be intentionally small.

### 7.1 Demo scope (v0)

### Mandatory mechanisms
- `Mech_RP`
- `Blocki12_JL` (historical baseline)

### Mandatory dataset
- `AutoMPG`

### Mandatory release-level metric
- relative Frobenius error of released Gram estimate  
  `||X^T X - X^T X_hat||_F / ||X^T X||_F`

### Mandatory downstream task
- `OLSFromRelease`

### Mandatory downstream metric
- test MSE

### Mandatory report artifact
- one privacy–utility figure:
  - x-axis: `epsilon`
  - y-axis: test MSE
  - one line per mechanism

### 7.2 Why this is the right demo

This demo is small enough to prove:
- the repo imports correctly,
- config loading works,
- mechanisms share one interface,
- one dataset adapter works,
- one downstream task works,
- one report can be generated,
- and the Slurm pathway is viable.

It is **not** intended to answer the entire phase-1 research program.

### 7.3 Demo config

Suggested demo benchmark contract:

| Parameter | Demo value |
|---|---|
| adjacency | add/remove row |
| dataset | AutoMPG |
| task | OLSFromRelease |
| release metric | relative Frobenius error on `X^T X` |
| downstream metric | test MSE |
| epsilon grid | `{0.5, 1, 2, 4}` |
| delta | `1 / n^2` |
| sketch ratio grid | `r/d in {2, 4}` |
| seeds | `0,1,2,3,4` |
| split | 80/20 train/test, fixed seed |
| preprocessing | shared scaling + clipping |
| report | one line figure + one markdown summary |

### 7.4 Demo acceptance criteria

The demo milestone is complete when the repo can produce all of the following from one command path:

1. one run manifest,
2. one successful local/interactive run,
3. one successful Slurm-submitted run,
4. aggregated metrics for both mechanisms,
5. one saved figure,
6. one markdown summary,
7. one notebook that replays the same demo without shelling out.

---

## 8) Goal 3 — mechanism inventory, downstream tests, datasets, and parameter inventory

### 8.1 Mechanism inventory

| ID                      | Role in WF-001                         | Phase-1 status                 | Notes for normalization                                       | Details URL                                          |
| ----------------------- | -------------------------------------- | ------------------------------ | ------------------------------------------------------------- | ---------------------------------------------------- |
| `Mech_RP`               | primary project mechanism              | mandatory                      | main RP-family reference point                                | `docs/notes/ndis-paper.pdf` (local)                  |
| `Mech_RP^Pois`          | project mechanism                      | mandatory after demo           | evaluate when redundancy / subsampling helps                  | `docs/notes/ndis-paper.pdf` (local)                  |
| `Mech_RP^PTR`           | project mechanism                      | mandatory after demo           | evaluate when data geometry permits smaller ridge             | `docs/notes/ndis-paper.pdf` (local)                  |
| `Blocki12_JL`           | historical RP/JL baseline              | mandatory in demo              | historical reference; not the main modern OLS baseline        | https://arxiv.org/abs/1204.2136                      |
| `Sheffet17_OLS`         | primary RP/sketch OLS baseline         | mandatory                      | core modern comparator                                        | https://proceedings.mlr.press/v70/sheffet17a.html    |
| `Sheffet19_OldTech`     | primary OLS baseline                   | mandatory                      | explicitly requested comparator family                        | https://proceedings.mlr.press/v98/sheffet19a.html    |
| `AdaSSP`                | strong non-sketch regression baseline  | mandatory secondary ring       | include in phase 1 after primary RP/sketch ring is stable     | https://arxiv.org/abs/1803.02596                     |
| `Lev25_GaussianMixing`  | recent Gaussian sketch baseline        | recommended primary ring       | document RDP-to-`(epsilon,delta)` conversion clearly          | https://arxiv.org/abs/2505.24603                     |
| `DD_SSP`                | improved SSP baseline                  | optional secondary ring        | useful later; not required for first runnable phase           | https://arxiv.org/abs/2405.15002                     |
| `DPGD_LinearRegression` | gradient-based baseline                | optional secondary ring        | useful for broader practical comparison                       | https://proceedings.mlr.press/v235/brown24a.html     |
| `IHM_2026`              | latest sketch baseline                 | optional “latest-result” track | relevant, but keep explicitly labeled as preprint/optional    | https://arxiv.org/abs/2601.07545                     |
| `PROJ099_upstream`      | provenance source, not a benchmark row | reference only                 | archived predecessor toolkit and pinned implementation source | https://github.com/stoneboat/NDIS-paper-RP-toolkits- |

### Comparator normalization policy

- **Primary ring:** RP/sketch comparators closest to the RP-family story.
- **Secondary ring:** strong non-sketch DP regression baselines.
- **Optional/latest ring:** high-value but higher-implementation-cost or preprint comparators.

This normalization should be frozen in `docs/notes/comparator_normalization.md`.

### 8.2 Downstream tasks and utility tests

| Task / metric | Status | Use in workflow | Details URL |
|---|---|---|---|
| OLS prediction (`LinearRegression`) | mandatory | first downstream task | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html |
| Ridge regression (`Ridge`) | mandatory after demo | second downstream task; useful when conditioning matters | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html |
| Test MSE | mandatory | primary downstream metric | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html |
| Train/test split (`train_test_split`) | mandatory | first benchmark split helper | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html |
| Relative Frobenius error of `X^T X` | mandatory | first release-level metric | formula defined in this workflow |
| Spectral norm error of `X^T X` | mandatory after demo | release-quality robustness metric | formula defined in this workflow |
| Relative error of `X^T y` | mandatory after demo | release-quality metric tied to regression | formula defined in this workflow |
| Parameter error `||beta_hat - beta_nonprivate||_2` | mandatory after demo | secondary downstream metric | formula defined in this workflow |
| Runtime / memory | mandatory | cost accounting | implementation-defined |
| Failure / instability rate | mandatory | detects singular or numerically unstable outcomes | implementation-defined |

### 8.3 Dataset plan

#### 8.3.1 Demo dataset

| Dataset | Status | Why it is in demo | Details URL |
|---|---|---|---|
| Auto MPG | mandatory demo dataset | small regression dataset; easy smoke test for OLS + released sufficient statistics | https://archive.ics.uci.edu/ml/datasets/auto%2Bmpg |

#### 8.3.2 Core real-data phase-1 panel

| Dataset | Status | Notes | Details URL |
|---|---|---|---|
| Communities & Crime | core | literature-aligned social-science regression dataset | https://archive.ics.uci.edu/ml/datasets/communities%2Band%2Bcrime |
| Tecator | core | appears in recent Gaussian-sketch benchmarking | https://www.openml.org/search?id=505&type=data |
| Boston Housing (legacy) | core but disabled by default | use only if historical comparability is needed; ethics caveat should be documented | https://fairlearn.org/v0.8/user_guide/datasets/boston_housing_data.html |
| Wine Quality | core | compact regression benchmark | https://archive.ics.uci.edu/dataset/186/wine%2Bquality |
| Bike Sharing | core | medium-size regression benchmark | https://archive.ics.uci.edu/ml/datasets/bike%2Bsharing%2Bdataset |
| Auto MPG | core | also used for demo | https://archive.ics.uci.edu/ml/datasets/auto%2Bmpg |

#### 8.3.3 Extension real-data panel (phase-1 backlog / scale-up)

| Dataset | Status | Suggested source for implementation | Details URL |
|---|---|---|---|
| Energy Efficiency | extension | direct adapter or `uci_datasets` | https://archive.ics.uci.edu/ml/datasets/energy%2Befficiency |
| Concrete Compressive Strength | extension | direct adapter or `uci_datasets` | https://archive.ics.uci.edu/ml/datasets/concrete%2Bcompressive%2Bstrength |
| Elevators | extension | OpenML adapter | https://www.openml.org/d/44134 |
| Gas | extension | use the concentration/regression version | https://archive.ics.uci.edu/ml/datasets/Gas%2BSensor%2BArray%2BDrift%2BDataset%2Bat%2BDifferent%2BConcentrations |
| Airfoil Self-Noise | extension | direct UCI adapter | https://archive.ics.uci.edu/dataset/291/airfoil%2Bself%2Bnoise |
| Parkinsons Telemonitoring | extension | direct UCI adapter | https://archive.ics.uci.edu/ml/datasets/parkinsons%2Btelemonitoring |
| Kin40k | extension | standardized benchmark loader strongly recommended | https://github.com/treforevans/uci_datasets |
| TamiElectric | extension | standardized benchmark loader strongly recommended | https://github.com/treforevans/uci_datasets |
| Buzz | extension | OpenML / standardized benchmark loader | https://www.openml.org/search?id=4549&type=data |
| Slice (CT slice localization) | extension | direct UCI adapter or standardized loader | https://archive.ics.uci.edu/ml/datasets/Relative%2Blocation%2Bof%2BCT%2Bslices%2Bon%2Baxial%2Baxis |

#### 8.3.4 Synthetic regime panel

These are benchmark generators rather than public datasets.

| Synthetic regime | Status | Purpose | Details source |
|---|---|---|---|
| High-leverage outlier regime | mandatory | test leverage-sensitive privacy/utility behavior | this workflow + NDIS RP motivation |
| Ill-conditioned regime | mandatory | stress small `lambda_min(X^T X)` | this workflow |
| Redundant-row regime | mandatory | target the intended benefit of Poisson wrapper | this workflow |
| Favorable-eigenvalue regime | mandatory | target the intended benefit of PTR | this workflow |
| High-residual regime | mandatory | separate release quality from downstream fit difficulty | this workflow |
| Small-`||theta*||_2` regime | recommended | aligns with recent DP-OLS analyses | this workflow |
| Low-leverage well-conditioned control | mandatory | sanity-control “friendly data” regime | this workflow |

### 8.4 Dataset sourcing recommendation

For implementation convenience, dataset adapters should support **two source modes**:

1. **Direct source mode**  
   Pull from UCI/OpenML pages directly.

2. **Standardized-split mode**  
   Use `uci_datasets` when the benchmark needs legacy standardized train/test splits matching prior regression benchmarks.

This dual-mode design avoids overfitting the repo to one data source while still making historical comparisons easier.

### 8.5 Testing parameter inventory

#### Frozen benchmark contract (phase 1)

| Parameter family | Recommended phase-1 default |
|---|---|
| adjacency | add/remove row |
| privacy notion | `(epsilon, delta)`-DP |
| epsilon grid | `{0.1, 0.25, 0.5, 1, 2, 4, 8, 10}` |
| primary delta | `1 / n^2` |
| continuity study | fixed `delta = 1e-6` |
| sketch size grid | `r/d in {1.5, 2.5, 4.5, 8}` |
| repetitions | 20 seeds for development, 50 seeds for final figures |
| split protocol | fixed train/test split for first benchmark release |
| preprocessing | one shared documented scaling/clipping recipe |
| row clipping | public bound `C_X` |
| label clipping | public bound `C_Y` |
| Poisson wrapper grid | sweep subsampling rate `q` |
| PTR wrapper grid | record selected ridge vs default ridge |
| runtime accounting | wall-clock runtime + peak memory if feasible |
| report aggregation | mean, std, and 95% CI over seeds |

#### Demo-specific reduced grid

| Parameter family | Demo value |
|---|---|
| epsilon grid | `{0.5, 1, 2, 4}` |
| delta | `1 / n^2` |
| sketch size grid | `r/d in {2, 4}` |
| seeds | 5 |
| dataset count | 1 |
| mechanism count | 2 |
| release metric count | 1 |
| downstream metric count | 1 |

### 8.6 Geometry metrics to log for regime studies

The repo should compute and log at least:

- maximum leverage,
- leverage concentration score (e.g. top-k share),
- `lambda_min(X^T X)`,
- condition number of `X^T X`,
- duplicate / near-duplicate row ratio,
- residual variance of the non-private fit,
- `||beta_nonprivate||_2`,
- effective sketch ratio `r/d`,
- selected `q` for Poisson wrapper,
- selected ridge for PTR wrapper.

These geometry columns are necessary for the “which mechanism wins where?” report.

---

## 9) Goal 4 — presentation / report design

### 9.1 Reporting objects

The reporting layer should generate four standard objects.

#### A. Release-level leaderboard
One row per:
`dataset x mechanism x epsilon x sketch ratio`

Columns:
- relative Frobenius error,
- spectral error,
- relative `X^T y` error,
- runtime,
- mechanism diagnostics.

#### B. Downstream leaderboard
One row per:
`dataset x mechanism x epsilon x task`

Columns:
- test MSE,
- train MSE,
- parameter error,
- runtime,
- failure rate.

#### C. Regime map
One row per synthetic regime configuration:
- regime label,
- geometry diagnostics,
- winning mechanism,
- margin to second-best.

#### D. Report-ready figure set
Minimum figure families:
1. privacy–utility curve,
2. release-quality curve,
3. regime heatmap,
4. runtime-vs-utility scatter,
5. mechanism-diagnostic plot (e.g. chosen ridge or `q` vs geometry).

### 9.2 Figure generation plan

All figures should be generated from saved aggregate tables, never from ad-hoc notebook code only.

Suggested pipeline:

1. `run_benchmark.py` writes row-level metrics.
2. `build_report.py` reads all rows into one aggregate dataframe.
3. `reporting/tables.py` writes CSV/Parquet summary tables.
4. `reporting/figures.py` writes PNG/SVG figures.
5. `reporting/dashboard.py` writes one markdown/HTML report.

### 9.3 Naming convention for figures

```text
reports/
  figures/
    eps_vs_test_mse__dataset=autompg__task=ols.png
    eps_vs_rel_fro_xtx__dataset=autompg.png
    regime_heatmap__winner_by_geometry.png
    runtime_vs_test_mse__dataset=tecator.png
```

### 9.4 Suggested first presentation deck (6 slides)

1. **Why phase 1 is RP-family only**
2. **Benchmark contract**
3. **Mechanisms and comparator rings**
4. **Release-level results**
5. **Downstream OLS/ridge results**
6. **Regime map: where Poisson/PTR help**

### 9.5 Minimum sample figure for the first git version

The first git version should ship one **sample report figure** under `reports/examples/` so that downstream contributors know the expected reporting style.

The figure should be:
- small,
- reproducible,
- based on the demo config,
- and clearly labeled if it uses mock/demo data.

---

## 10) Workflow task breakdown

| Task ID | Task | Main output | Acceptance criterion |
|---|---|---|---|
| T0 | Freeze benchmark charter | `docs/notes/benchmark_charter.md` | adjacency, preprocessing, grids, comparator rings frozen |
| T1 | Freeze comparator normalization | `docs/notes/comparator_normalization.md` | every mechanism labeled primary / secondary / optional |
| T2 | Build repo skeleton | checked-in directory tree | imports work and tests discover modules |
| T3 | Define Purdue install script contract | `scripts/local_scripts/install_purdue_cluster.sh` stub + README note | script is idempotent and documents required template substitutions |
| T4 | Implement common interfaces | `DatasetAdapter`, `Mechanism`, `ReleaseBundle`, `Task`, `ExperimentRunner` | one toy run works end-to-end |
| T5 | Implement demo mechanisms | `rp_ndis.py`, `blocki12_jl.py` | both produce valid `ReleaseBundle`s |
| T6 | Implement demo dataset + task | `autompg.py`, `ols_from_release.py` | one full demo run succeeds |
| T7 | Implement report builder | `build_report.py`, `reporting/*` | sample figure and summary markdown generated |
| T8 | Expand to RP-family wrappers | `rp_pois.py`, `rp_ptr.py` | all three project mechanisms share the same interface |
| T9 | Add primary RP/sketch baselines | Sheffet 2017, Sheffet 2019, Lev 2025 | primary ring runnable on core panel |
| T10 | Add secondary baselines | AdaSSP first, others later | secondary leaderboard runnable |
| T11 | Add core real-data panel | six dataset adapters | each passes one smoke test |
| T12 | Add synthetic regime panel | regime generators + geometry metrics | regime report generated |
| T13 | Run main sweep | aggregate result tables | all core curves generated with fixed seeds |
| T14 | Write analysis memo | `reports/phase1_summary.md` | answers all three phase-1 scientific questions |

---

## 11) Immediate recommended build order

### Stage A — skeleton + demo
Do first:
- T0
- T1
- T2
- T3
- T4
- T5
- T6
- T7

### Stage B — full RP-family phase 1
Then:
- T8
- T9
- T11
- T12
- T13
- T14

### Stage C — optional strengtheners
Finally:
- T10 optional baselines (`DD_SSP`, `DPGD`, `IHM_2026`)
- extension real-data panel

---

## 12) Open decisions / risk notes

1. **Purdue template not yet supplied.**  
   The install script can only be finalized after the user provides the cluster template.

2. **RDP comparators need a normalization note.**  
   At least one comparator (e.g. Gaussian Mixing) may require explicit conversion into a common `(epsilon, delta)` reporting frame.

3. **Boston Housing should be treated carefully.**  
   Keep it disabled by default and document the ethical caveat.

4. **Dataset source mode should be frozen early.**  
   Decide whether a comparison is with direct-source splits or standardized legacy splits before generating leaderboards.

5. **Do not blur the demo with the full benchmark.**  
   The first git version should be visibly minimal.

---

## 13) Recommendation

Register **WF-001** as the canonical **phase-1 RP benchmark workflow** and treat the first implementation milestone as:

- one repo,
- one demo config,
- `Mech_RP` + `Blocki12_JL`,
- one dataset,
- one release metric,
- one downstream metric,
- one sample figure,
- one Slurm path,
- and one report builder.

That is the right minimal substrate for the broader RP-family benchmark, and later for phase-2 general Gaussian-output benchmarking.

---

## 14) Source basis and URLs

### Internal source basis
- `2026-03-17_proj-004-phase-1-rp-benchmark-workflow.md`
- `memo_mar-9_proj-004-rq3_benchmark-plan.md`
- `REPRODUCTION_MEMO.md`

### Verified external URLs used above
- NDIS / RP-family paper: https://arxiv.org/abs/2309.01243
- PROJ-099 upstream repo: https://github.com/stoneboat/NDIS-paper-RP-toolkits-
- Blocki et al. 2012: https://arxiv.org/abs/1204.2136
- Sheffet 2017: https://proceedings.mlr.press/v70/sheffet17a.html
- Sheffet 2019: https://proceedings.mlr.press/v98/sheffet19a.html
- AdaSSP / Wang 2018: https://arxiv.org/abs/1803.02596
- Gaussian Mixing / Lev et al. 2025: https://arxiv.org/abs/2505.24603
- DD-SSP / Ferrando & Sheldon: https://arxiv.org/abs/2405.15002
- DP-GD for linear regression / Brown et al. 2024: https://proceedings.mlr.press/v235/brown24a.html
- IHM 2026 preprint: https://arxiv.org/abs/2601.07545
- Purdue RCAC Slurm docs: https://www.rcac.purdue.edu/knowledge/slurm?all=true
- Purdue RCAC job submission example: https://www.rcac.purdue.edu/knowledge/scholar/run/slurm/submit
- Purdue RCAC scratch guide: https://www.rcac.purdue.edu/knowledge/scholar/storage/options/scratch
- Purdue RCAC storage overview: https://www.rcac.purdue.edu/storage/scratch
- scikit-learn OLS: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- scikit-learn Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- scikit-learn MSE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
- scikit-learn train/test split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- `uci_datasets` standardized regression splits repo: https://github.com/treforevans/uci_datasets