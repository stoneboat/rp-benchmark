# Memo (Mar-9): Benchmark and Workflow Plan for PROJ-004-RQ3

**Project:** PROJ-004  
**Research question:** RQ3 — *How do other NDIS-derived Gaussian-output mechanisms compare empirically with state-of-the-art DP mechanisms for the same tasks, in terms of privacy–utility tradeoff and robustness?*  
**Prepared for:** Yu Wei  
**Status:** planning memo (pre-WF)

## Purpose
This memo turns the current motivation around the project-099 random-projection mechanisms
$\Mech_{RP}$, $\Mech^{\mathsf{Pois}}_{RP}$, and $\Mech^{\mathsf{PTR}}_{RP}$ into a concrete execution plan.

The immediate objective is **not** to answer the full umbrella RQ3 in one step. The immediate objective is to set up a benchmark program that can cleanly answer:

1. how strong the project-099 RP-family mechanisms are relative to prior **RP/sketch-based** DP methods, and  
2. how competitive they are relative to strong non-sketch DP regression baselines when the downstream task is least squares / ridge-style regression.

My recommendation is to create a workflow for this, but to scope the first workflow narrowly and deliberately.

---

## Executive recommendation

### Recommendation 1 — create a workflow, but make it **phase 1 only**
Create a workflow for **RP-family benchmarking first**, not for the entire “all Gaussian-output mechanisms” agenda.

The first workflow should answer:

- how $\Mech_{RP}$ compares with published RP/sketch DP baselines,
- whether $\Mech^{\mathsf{Pois}}_{RP}$ and $\Mech^{\mathsf{PTR}}_{RP}$ improve the privacy–utility tradeoff in the data regimes they are designed for,
- and which mechanism wins under which data geometry.

### Recommendation 2 — treat broader Gaussian-output benchmarking as **phase 2**
Once the RP benchmark infrastructure is stable, re-use it for broader NDIS-derived Gaussian-output mechanisms. That second phase can target examples such as Bayesian logistic regression and Gaussian process regression, which already appear naturally in the NDIS draft as Gaussian-output algorithms that can be privatized through NDIS-based calibration [1].

### Recommendation 3 — build the benchmark around **common interfaces**
The first deliverable should be a plug-and-play codebase with common interfaces for:
- mechanisms,
- datasets,
- downstream tasks,
- metrics,
- and experiment runners.

This matters because the benchmark is meant to be split across agents. Without a common interface, implementation effort will fragment immediately.

---

## Why this scope is aligned with the current draft

The uploaded NDIS draft already gives the main technical reasons to stage the work in this way [1].

First, for Gaussian RP the draft identifies **leverage** as the privacy-critical quantity, shows how to calibrate ridge regularization through an explicit $\delta^{gRP}_r(\varepsilon; p)$ curve, and proves a utility statement via unbiased Gram recovery from the sketch. This makes RP an unusually well-specified first benchmark target [1].  
Second, the draft already introduces the two practical wrappers $\Mech^{\mathsf{Pois}}_{RP}$ and $\Mech^{\mathsf{PTR}}_{RP}$ and explains the regimes in which each can improve utility: Poisson subsampling helps when the data is redundant, while PTR helps when $\lambda_{\min}(D^\top D)$ is large enough to justify less conservative regularization [1].  
Third, the same draft explicitly positions NDIS as a **general lifting tool** for Gaussian-output algorithms beyond RP, which makes an RP-first workflow a natural first milestone rather than a detour [1].  

So the clean interpretation is:

- **Phase 1:** benchmark the RP-family mechanisms thoroughly.
- **Phase 2:** port the same benchmark scaffold to other NDIS-derived Gaussian-output mechanisms.

---

## Recommended benchmark questions for Phase 1

I would operationalize the first workflow around the following questions.

### Q1. RP-family comparison
How do $\Mech_{RP}$, $\Mech^{\mathsf{Pois}}_{RP}$, and $\Mech^{\mathsf{PTR}}_{RP}$ compare with prior RP/sketch-based DP mechanisms in terms of privacy–utility tradeoff?

### Q2. Downstream least-squares comparison
When the private release is used for OLS or ridge-style downstream learning, how do the project-099 mechanisms compare with strong DP regression baselines?

### Q3. Regime characterization
Which mechanism wins in which regime, and which geometric quantities explain the winner?

The regime variables should include at least:
- leverage / leverage concentration,
- $\lambda_{\min}(D^\top D)$ or conditioning,
- redundancy / duplicate-cluster structure,
- residual size,
- and sketch dimension relative to ambient dimension.

---

## Proposed scope of the first workflow

### In scope
- RP-family mechanisms from project-099:
  - $\Mech_{RP}$
  - $\Mech^{\mathsf{Pois}}_{RP}$
  - $\Mech^{\mathsf{PTR}}_{RP}$
- Downstream tasks:
  - OLS prediction
  - ridge regression
- Utility views:
  - release-level utility
  - downstream-task utility
- Real data + synthetic regime studies
- Plug-and-play benchmark infrastructure

### Out of scope for the first workflow
- Bayesian logistic regression
- Gaussian process regression
- full “all Gaussian-output mechanisms vs all SOTA mechanisms” benchmark
- confidence-interval / inference benchmarking unless it becomes essential later

This keeps the first workflow disciplined and finishable.

---

## Comparator set

I recommend organizing comparators into two rings.

### Ring A — primary RP / sketch baseline ring
This is the **main** comparison ring for the first workflow.

1. **Blocki et al. (2012)** — historical JL-transform privacy baseline [2].  
   This should be included as a historical reference point, but not treated as the main modern OLS baseline.

2. **Sheffet (2017)** — DP ordinary least squares via Gaussian sketching [3].  
   This is a mandatory baseline.

3. **Sheffet (2019)** — *Old Techniques in Differentially Private Linear Regression* [4].  
   This is also mandatory, especially because you explicitly want comparison against it.

4. **Lev et al. (2025), Gaussian Mixing / Gaussian Sketches** [6].  
   This is the most relevant recent RP/sketch comparator to include by default.

5. **Iterative Hessian Mixing (2026 preprint)** [9].  
   I would treat this as an **optional latest-result track**, clearly labeled as a preprint. It is highly relevant, but I would not make it mandatory for the first round if implementation cost is high.

### Ring B — strong non-sketch DP regression ring
This is a **secondary** comparison ring. It answers the broader practical question:
can the RP-family mechanisms compete with strong DP regression methods even when those methods are not sketch-based?

1. **AdaSSP (Wang, 2018)** — mandatory secondary baseline [5].  
   This is still the most standard adaptive sufficient-statistics baseline for private linear regression.

2. **Data-dependent SSP (Ferrando & Sheldon, 2025)** — optional secondary baseline [7].  
   This is worth including later because it improves on data-independent SSP in practice, but it expands implementation scope.

3. **DP-GD for linear regression (Brown et al., 2024)** — optional secondary baseline [8].  
   This is useful as a gradient-based comparator for downstream regression, but not essential for the initial RP-first release-level study.

### Not recommended as default phase-1 baselines
- **Objective perturbation** should not be a default OLS baseline for this first workflow. It is more useful later if the benchmark expands to logistic regression or broader ERM tasks.
- Mechanisms analyzed under **different privacy notions** (for example MI-DP rather than standard $(\varepsilon,\delta)$-DP) should not go into the main leaderboard unless the comparison is explicitly labeled.

---

## Recommended dataset plan

The benchmark should be staged rather than launched on a massive panel all at once.

### Real-data core panel
Start with a compact, literature-aligned panel:

1. **Communities & Crime**
2. **Tecator**
3. **Boston Housing**
4. **Wine**
5. **Bike Sharing**
6. **AutoMPG**

This core panel is a good starting point because:
- Communities & Crime and Tecator already appear in recent Gaussian-sketch benchmarking [6],
- and the broader recent IHM study includes Boston Housing, Tecator, Wine, Bike Sharing, and AutoMPG among its real-data suite [9].

### Real-data extension panel
Only after the core panel is stable, add:
- Energy
- Concrete
- Elevators
- Gas
- Airfoil
- Parkinsons
- Kin40k
- TamiElectric
- Buzz
- Slice

I would **not** begin with all available datasets. Start with the overlap panel, then extend.

### Synthetic regime panel
The synthetic suite should not be generic isotropic Gaussian data only. It should be targeted to the quantities your mechanisms are actually about.

The synthetic families should include:

1. **High-leverage outlier regime**  
   One or a few rows dominate leverage.

2. **Ill-conditioned regime**  
   Small $\lambda_{\min}(D^\top D)$ / badly conditioned design.

3. **Redundant-row regime**  
   Many rows are nearly duplicated or strongly clustered. This is especially relevant for the Poisson wrapper.

4. **Favorable-eigenvalue regime**  
   Large enough $\lambda_{\min}(D^\top D)$ so that PTR can plausibly reduce regularization.

5. **High-residual regime**  
   Useful for understanding when downstream fitting behaves differently across methods.

6. **Small-$\|\theta^\*\|_2$ regime**  
   A common axis used in recent DP-OLS analyses.

7. **Well-conditioned low-leverage regime**  
   A “friendly data” control case where all reasonable mechanisms should look strong.

The point of this synthetic panel is not only to get curves. It is to explain *why* a mechanism wins.

---

## Common experimental setting

The first workflow should freeze a single public benchmark contract.

### Adjacency
Use **add/remove row adjacency** throughout.

### Public bounds
Use a public row-norm bound and a public label bound whenever the method requires them.

### Preprocessing
Use a single documented preprocessing contract across all mechanisms. A pragmatic default is:

- remove rows with missing targets,
- train/test split fixed in advance,
- feature normalization under one shared benchmark recipe,
- row clipping to a public $C_X$,
- label clipping to a public $C_Y$.

Important note: if later you want the benchmark to reflect strict end-to-end DP accounting, then any data-dependent normalization step must either be made public, replaced by a private estimate, or explicitly declared outside the privacy budget. For the first workflow, consistency is more important than full end-to-end accounting, but the preprocessing contract must be written down.

### Privacy grid
I recommend:

- $\varepsilon \in \{0.1, 0.25, 0.5, 1, 2, 4, 8, 10\}$
- primary $\delta = 1/n^2$
- plus one fixed-$\delta$ study at $\delta = 10^{-6}$ for direct continuity with the RP calibration analysis already present in the NDIS draft [1]

### Sketch-size grid
For methods with sketch dimension $r$ or $k$, sweep ratios such as

- $r/d \in \{1.5, 2.5, 4.5, 8\}$

This is more interpretable than using raw sketch dimension only.

### Repetitions
Use at least 20 random seeds for the first sweep and 50 seeds for the final figures.

---

## What to measure

I strongly recommend maintaining **two leaderboards**.

### Leaderboard A — release-level utility
This treats the mechanism as a reusable private release, not just a one-off predictor.

Measure:
- error of the released Gram / covariance estimate
- Frobenius norm error
- spectral norm error
- relative error
- effective regularization level
- sketch dimension used
- runtime
- memory
- mechanism-specific control quantities:
  - leverage cap or estimated leverage proxy
  - subsampling rate $q$ for $\Mech^{\mathsf{Pois}}_{RP}$
  - PTR-selected ridge versus baseline ridge for $\Mech^{\mathsf{PTR}}_{RP}$

For the project-099 family, this leaderboard is important because the NDIS RP mechanism naturally supports a sketch-based release view.

### Leaderboard B — downstream-task utility
This treats each mechanism as a component in a regression pipeline.

Measure:
- test MSE
- train MSE / empirical risk
- excess empirical risk relative to the non-private solver
- parameter error $\|\hat\beta - \beta_{\mathrm{nonprivate}}\|_2$
- runtime
- failure or instability rate

The downstream tasks should begin with:
- OLS
- ridge regression

I would not introduce additional downstream tasks until these two are stable.

---

## Plug-and-play code architecture

The benchmark codebase should be built around a small set of interfaces.

### 1. `DatasetAdapter`
Responsibilities:
- load raw dataset
- apply the benchmark preprocessing contract
- expose public metadata: $(n,d)$, row-norm bound, label bound, split information
- provide synthetic-neighbor helpers for debugging and audits

### 2. `Mechanism`
Responsibilities:
- accept a privacy specification and public metadata
- calibrate internal parameters
- release a private object from training data
- export mechanism metadata needed for analysis

Suggested conceptual API:

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

### 3. `ReleaseBundle`
A common object that can store any of:
- private sketch
- $\widehat{X^\top X}$
- $\widehat{X^\top y}$
- private coefficient vector
- calibration metadata
- audit metadata

This prevents each mechanism from inventing its own incompatible return type.

### 4. `Task`
Responsibilities:
- consume a `ReleaseBundle`
- produce a fitted downstream model under a common solver contract
- evaluate against held-out data

Suggested tasks:
- `OLSFromRelease`
- `RidgeFromRelease`

### 5. `MetricSuite`
Responsibilities:
- release-level metrics
- downstream metrics
- runtime / memory
- stability summaries

### 6. `ExperimentRunner`
Responsibilities:
- grid construction
- seed control
- result logging
- figure generation
- checkpointed execution

This architecture is the main precondition for multi-agent implementation.

---

## Proposed workflow

**Suggested workflow name:** `WF-RQ3-RP-BENCH`

**Goal:** establish a reproducible benchmark for project-099 RP-family DP mechanisms against RP/sketch baselines and selected strong non-sketch baselines.

### Workflow tasks

| ID | Task | Main output | Acceptance criterion |
|---|---|---|---|
| T0 | Benchmark charter | Frozen scope note: adjacency, preprocessing, privacy grid, mechanism list, dataset core panel, metrics | One written spec; no unresolved scope ambiguity |
| T1 | Comparator normalization note | Short note on which baselines are directly comparable, which are historical only, and which are optional/preprint | Every baseline assigned to primary / secondary / optional ring |
| T2 | Benchmark infra skeleton | `DatasetAdapter`, `Mechanism`, `ReleaseBundle`, `Task`, `ExperimentRunner` | One minimal end-to-end run works on a toy dataset |
| T3 | Project-099 mechanism package | Implement $\Mech_{RP}$, $\Mech^{\mathsf{Pois}}_{RP}$, $\Mech^{\mathsf{PTR}}_{RP}$ under one common interface | All three mechanisms run through the same harness |
| T4 | Baseline package A | Implement Sheffet 2017, Sheffet 2019, and one recent Gaussian-sketch comparator | Reproduces basic sanity checks and benchmark outputs |
| T5 | Baseline package B | Implement AdaSSP, then optionally data-dependent SSP and/or DP-GD | Secondary leaderboard becomes runnable |
| T6 | Real-data adapters | Implement the core dataset panel | Each dataset passes preprocessing and one smoke test |
| T7 | Synthetic regime generator | High-leverage / ill-conditioned / redundant / favorable-eigenvalue / high-residual generators | Each regime produces interpretable diagnostics |
| T8 | Metric and audit package | Release-level metrics, downstream metrics, runtime/memory, optional white-box pair checks | Metrics are logged uniformly for all mechanisms |
| T9 | Main sweep | Full experiment grid over privacy and sketch dimension | All core curves produced with fixed seeds |
| T10 | Analysis memo | Regime map and recommendations | Clear answer to “which mechanism wins where?” |

---

## Suggested agent split

If this gets parallelized, I would split the work as follows.

### Agent A — benchmark charter and literature normalization
Owns:
- T0
- T1

### Agent B — benchmark infrastructure
Owns:
- T2
- part of T8

### Agent C — project-099 mechanisms
Owns:
- T3

### Agent D — external baselines
Owns:
- T4
- T5

### Agent E — datasets and synthetic generators
Owns:
- T6
- T7

### Agent F — experiment execution and analysis
Owns:
- T9
- T10

This split keeps theory-heavy and engineering-heavy work decoupled.

---

## Decision points that still need to be frozen

Before finalizing the workflow, I would resolve the following items.

### D1. Is IHM in the default benchmark?
My recommendation: **optional, not default** for round 1, because it is a 2026 preprint and adds implementation complexity.

### D2. Is Ring B mandatory in phase 1?
My recommendation: **AdaSSP yes**, everything else optional for the first round.

### D3. Do we optimize for release quality or downstream OLS first?
My recommendation: keep both leaderboards, but execute **downstream OLS/ridge first** after the release bundle is stable. That is the clearest story for the first results.

### D4. How many datasets in the first sweep?
My recommendation: **six real datasets + five synthetic regimes**. Expand only after the harness is stable.

### D5. Do we include private auditing in the first workflow?
My recommendation: only a light white-box diagnostic layer at first; full auditing can wait.

---

## Expected outcome of the first workflow

A successful first workflow should produce:

1. a clean answer to whether the project-099 RP mechanisms are competitive with prior RP/sketch DP mechanisms,
2. a regime map explaining when Poisson and PTR variants help,
3. a reusable benchmark codebase,
4. and a principled basis for deciding whether to launch phase 2 on broader NDIS-derived Gaussian-output mechanisms.

In other words, the first workflow should not merely generate plots. It should create the **benchmark substrate** for the rest of RQ3.

---

## My recommendation for the next move

The next move I would take is:

1. freeze the **phase-1 scope** as “RP-family benchmark for OLS/ridge,”
2. register a workflow equivalent to `WF-RQ3-RP-BENCH`,
3. and immediately launch T0, T1, and T2 in parallel.

Only after that should we finalize the broader “other Gaussian-output algorithms vs SOTA” phase.

---

## References

1. **Uploaded NDIS draft (local file).**  
   *The Normal Distributions Indistinguishability Spectrum and its Application to Privacy-Preserving Machine Learning.*  
   Local copy: [NDIS_paper__RP_.pdf](sandbox:/mnt/data/NDIS_paper__RP_.pdf)

2. **Blocki, Blum, Datta, Sheffet (2012).**  
   *The Johnson-Lindenstrauss Transform Itself Preserves Differential Privacy.*  
   <https://arxiv.org/abs/1204.2136>

3. **Sheffet (2017).**  
   *Differentially Private Ordinary Least Squares.*  
   <https://proceedings.mlr.press/v70/sheffet17a.html>

4. **Sheffet (2019).**  
   *Old Techniques in Differentially Private Linear Regression.*  
   <https://proceedings.mlr.press/v98/sheffet19a.html>

5. **Wang (2018).**  
   *Revisiting Differentially Private Linear Regression: Optimal and Adaptive Prediction & Estimation in Unbounded Domain.*  
   <https://www.auai.org/uai2018/proceedings/papers/40.pdf>

6. **Lev, Srinivasan, Shenfeld, Ligett, Sekhari, Wilson (2025).**  
   *The Gaussian Mixing Mechanism: Rényi Differential Privacy via Gaussian Sketches.*  
   <https://openreview.net/forum?id=IjqTJELKU1>

7. **Ferrando, Sheldon (2025).**  
   *Private Regression via Data-Dependent Sufficient Statistic Perturbation.*  
   <https://openreview.net/forum?id=gtCfDKm9ME>

8. **Brown, Dvijotham, Evans, Liu, Smith, Guha Thakurta (2024).**  
   *Private Gradient Descent for Linear Regression: Tighter Error Bounds and Instance-Specific Uncertainty Estimation.*  
   <https://proceedings.mlr.press/v235/brown24a.html>

9. **Near-Optimal Private Linear Regression via Iterative Hessian Mixing (2026 preprint).**  
   <https://arxiv.org/abs/2601.07545>

10. **Chaudhuri, Monteleoni, Sarwate (2011).**  
    *Differentially Private Empirical Risk Minimization.*  
    <https://www.jmlr.org/papers/v12/chaudhuri11a.html>

