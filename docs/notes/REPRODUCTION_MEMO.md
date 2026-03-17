# Reproduction Memo

This memo captures the reusable project structure and workflow patterns that worked well in this reproduction. The concrete algorithm here is specific to private synthetic text generation, but the repo layout, implementation split, experiment flow, and notebook style are intended to be reusable for future reproduction projects.

## Goal Of This Memo

Use this as a template for future paper-reproduction repos where:

- there is one core method or algorithm to implement,
- there is a small number of environment-specific setup steps,
- experiments need a clean command-line entry point,
- notebooks are helpful for explanation and demos,
- outputs should be easy to inspect and compare.

The exact model, datasets, privacy math, evaluation, and runtime constraints can change across projects. The reusable part is the structure.

## Recommended Repo Structure

```text
project_root/
  README.md
  requirements.txt
  .gitignore

  src/
    __init__.py
    config.py
    <core_method>.py
    <supporting_math_or_utils>.py
    <evaluation>.py

  scripts/
    run_experiment.py
    local_scripts/
      install_<machine>.sh

  Notebook/
    demo.ipynb

  data/
    models/
    datasets/
    outputs/
```

## Why This Structure Works

### `src/`

`src/` should contain all reusable Python logic, with each file owning one responsibility.

Typical split:

- `config.py`
  - dataclasses for runtime configuration,
  - default hyperparameters,
  - prompt templates or dataset-specific constants,
  - parameter grids for sweeps if needed.
- one or more core implementation files
  - the actual algorithm,
  - supporting transforms,
  - privacy/accounting/math helpers,
  - sampling / selection / scoring logic.
- `evaluate.py`
  - save/load outputs,
  - compute summary metrics,
  - helper functions for downstream evaluation.

This split makes the code importable from both:

- `scripts/run_experiment.py`, and
- notebooks such as `Notebook/demo.ipynb`.

### `scripts/run_experiment.py`

Keep one clear CLI entry point that:

1. parses arguments,
2. loads data,
3. builds configs,
4. loads the model,
5. runs generation or training,
6. saves outputs with metadata,
7. prints summary stats.

This is the most important executable surface in the repo. It should be usable without notebooks.

### `scripts/local_scripts/`

Put machine-specific setup here, not in `README.md` commands alone.

This keeps environment logic separate from research logic:

- GPU/module loading,
- venv or conda creation,
- dependency installation,
- model download,
- notebook kernel registration,
- cloud-machine quirks.

If you later need support for a cluster, laptop, or another cloud image, add another install script instead of overloading the main code.

### `Notebook/`

Keep notebooks as examples and explanations, not as the primary implementation.

Best pattern:

- notebooks should import from `src/` and `scripts/`,
- notebooks should mirror the CLI flow in smaller steps,
- notebooks should avoid containing the only copy of important logic.

This project's `Notebook/demo.ipynb` follows that pattern after being rewritten to call the Python APIs directly instead of shelling out to the CLI.

### `data/`

Treat `data/` as local state and keep it out of git.

Suggested subfolders:

- `data/models/`: downloaded models,
- `data/datasets/`: cached datasets,
- `data/outputs/`: generated outputs, metrics, logs.

This makes the repo lightweight and reproducible while still giving a predictable place for large artifacts.

## Recommended Implementation Pattern

The most reusable software pattern from this project is:

### 1. Put all tunable values in config dataclasses

Use small config objects instead of long chains of loose parameters. This makes it easier to:

- reuse code across CLI and notebooks,
- log metadata,
- compare runs,
- sweep parameters later.

### 2. Keep pure logic separate from I/O

For example:

- math/accounting code should not download data,
- generation code should not parse CLI arguments,
- save/load logic should not run the model.

This separation made it easy here to expose the same workflow in both `scripts/run_experiment.py` and `Notebook/demo.ipynb`.

### 3. Save outputs with metadata

Every generated output file should contain enough metadata to reconstruct the run:

- dataset,
- key hyperparameters,
- seed,
- runtime,
- privacy or evaluation summary values.

This project stores metadata in the first line of the JSONL output. That pattern is worth reusing.

### 4. Keep evaluation helpers lightweight

Even if full evaluation is not ready yet, include:

- output saving/loading,
- basic stats,
- small inspection utilities.

This makes debugging and sanity-checking much easier early in the project.

## Experiment Workflow Pattern

The workflow that worked well here:

1. Start from one paper and define the minimum reproducible core.
2. Implement the core method in `src/`.
3. Add one CLI script that runs end-to-end.
4. Run a tiny sanity-check experiment first.
5. Add one notebook that replays the same pipeline step by step in Python.
6. Only then broaden dataset support, parameter sweeps, or evaluation.

This keeps the project from becoming notebook-only or script-only.

## Practical Lessons From This Reproduction

These are not paper-specific and are worth reusing in future repos.

### 1. The initial plan should allow pragmatic deviations

The original plan expected one setup style, but the actual implementation changed where necessary:

- the setup ended up using a Python venv in `/tmp/python-venv/` rather than the originally sketched conda flow,
- HuggingFace authentication had to be made explicit with `HF_TOKEN`,
- the install script had to absorb cloud-machine quirks that were not obvious up front.

Takeaway: keep the plan as a guide, not a contract.

### 2. Environment setup should be idempotent

The install script worked better once it:

- tolerated reruns,
- reused existing environments,
- skipped existing model downloads,
- repaired broken env directories,
- printed actionable failure messages.

This is a strong pattern for any reproduction repo.

### 3. Real-world infra issues belong in setup scripts

Important fixes that ended up mattering here:

- disabling IPv6 to avoid HuggingFace hangs on EC2,
- checking/loading NVIDIA kernel modules,
- verifying PyTorch CUDA visibility,
- handling gated-model authentication,
- registering a Jupyter kernel that points at the project venv.

These are not algorithmic, but they heavily affect reproducibility. Capture them in scripts, not tribal knowledge.

### 4. Memory constraints often require implementation changes

The initial algorithmic implementation was correct but not practical at full batch size. The main fix was:

- micro-batching model forward passes in `src/generate.py`,
- plus setting `PYTORCH_ALLOC_CONF=expandable_segments:True`.

Takeaway: in reproduction work, "paper-correct" and "machine-runnable" are different milestones. The repo structure should make room for runtime adaptations without muddying the algorithm.

### 5. Keep one pure-Python path in addition to the CLI

The notebook was most useful after being changed from:

- "run the CLI via subprocess"

to:

- "import the same Python functions and execute the workflow cell by cell."

That pattern is much better for teaching, debugging, and future reuse.

## Reusable File Responsibilities

If building a new reproduction project, this mapping is a good starting point:

- `README.md`
  - short project description,
  - quick install,
  - one canonical run command,
  - minimal structure overview.
- `requirements.txt`
  - direct dependencies only,
  - keep it short unless pinning is truly needed.
- `src/config.py`
  - runtime configs and defaults,
  - templates/constants,
  - sweep grids if relevant.
- `src/<algorithm>.py`
  - main core method implementation.
- `src/<math_or_utils>.py`
  - isolated helper logic with clear inputs/outputs.
- `src/evaluate.py`
  - save/load/stats/eval helpers.
- `scripts/run_experiment.py`
  - end-to-end entry point.
- `scripts/local_scripts/install_<env>.sh`
  - environment and machine setup only.
- `Notebook/demo.ipynb`
  - educational, step-by-step Python walkthrough.
- `data/outputs/*.jsonl`
  - outputs with metadata for later comparison.

## Good Defaults To Reuse

- Use dataclasses for config.
- Keep outputs machine-readable.
- Save metadata with every run.
- Make install scripts rerunnable.
- Use one CLI as the canonical execution path.
- Make notebooks import real code instead of duplicating logic.
- Put local artifacts under `data/`.
- Keep top-level structure small and predictable.

## Things To Adapt Per Project

These should change freely in future work:

- model family,
- dataset sources,
- prompt format,
- privacy or optimization math,
- evaluation protocol,
- output schema,
- hardware assumptions,
- install script details.

The point is not to copy this repo exactly. The point is to reuse the shape:

- clean `src/`,
- one runner script,
- one environment setup area,
- one notebook demo,
- one outputs area,
- metadata-first experiments.

## Suggested Use In Future Projects

When starting a new reproduction repo:

1. copy the folder skeleton,
2. rename the algorithm-specific files,
3. keep `run_experiment.py` as the main entry point,
4. write the install script early,
5. add metadata-rich output saving before large experiments,
6. add a pure-Python notebook after the CLI works,
7. document the deviations from the original paper or plan as they arise.

That combination gives a repo that is easier to run, explain, debug, and extend.
