# EHC-SN Architecture Specification

> Canonical source of truth for component boundaries, vocabulary, and composition
> patterns. See `spec/spec-manifest.toml` for precedence rules.

## 1 Project Identity

**EHC-SN** (_Entorhinal-Hippocampal Circuit — Spatial Navigation_) is a research
library for biologically-inspired spatial navigation models built on PyTorch and
Lightning.

**Goal**: Resolve complex navigation tasks (mazes) using advanced neural models
(i.e. TEM, HRM) that capture how the hippocampal formation (HPC) and
prefrontal cortex (PFC) interact for goal-directed generalization. The project is
inspired by the work of Zheng, Wolf, Ranganath, O'Reilly & McKee (_"Flexible
Prefrontal Control over Hippocampal Episodic Memory for Goal-Directed
Generalization"_).

**TEM** (_Tolman-Eichenbaum Machine_) is a multi-scale spatial memory model that
composes LEC, MEC, and HPC modules.

**HRM** (_Hierarchical Reasoning Model_) is a PFC-based recurrent reasoning architecture
with striatal gating (STR) for adaptive computation time.

**Target users**: Computational neuroscience researchers and ML practitioners
studying hippocampal/entorhinal spatial models.

---

## 2 Canonical Import Namespace

The sole canonical import namespace is **`ehc_sn`**.

Legacy namespaces, e.g. `torch_tem` and `hrm_sn` are retired; legacy code exist under
`temp/` (archived legacy code, not on the Python path). No new code may import from
`torch_tem` or `hrm_sn`.

---

## 3 Component Taxonomy

Every top-level package under `src/ehc_sn/` maps to exactly one component below.
If a new package is created, this table must be updated.

### 3.1 Brain-Region Modules

These implement neuroscience-grounded circuit components. Each is a `nn.Module`
(or collection of modules) that can be composed by a top-level model.

| Component | Path           | Biological Role                                                                                                                                                          | Computational Responsibility                                                                                                                                                                                                                                                            |
| --------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LEC**   | `modules/lec/` | Lateral entorhinal cortex (LEC): sensory encoding and temporal frequency filtering.                                                                                      | Transforms raw observations into multi-scale feature codes.                                                                                                                                                                                                                             |
| **MEC**   | `modules/mec/` | Medial entorhinal cortex (MEC): path integration, grid-cell spatial coding, and object-vector cell (OVC) representations.                                                | Grid-cell dynamics, abstract-location projections, and OVC encoding.                                                                                                                                                                                                                    |
| **HPC**   | `modules/hpc/` | Hippocampus (HPC): episodic memory formation, pattern completion via attractor dynamics, and place-cell spatial coding.                                                  | Hebbian associative memory, attractor retrieval, and grounded-location inference.                                                                                                                                                                                                       |
| **PFC**   | `modules/pfc/` | Prefrontal cortex (PFC): working memory maintenance and goal-directed reasoning over episodic memory.                                                                    | Two-level recurrent architecture ($z_H$, $z_L$) with transformer blocks and alternating update cycles.                                                                                                                                                                                  |
| **STR**   | `modules/str/` | Striatum (STR): action selection and gating via Go/NoGo (D1/D2-like) pathways. Receives projections from PFC and (optionally) HPC; modulates when to act vs. deliberate. | `nn.Module`(s) and protocols defining STR's public contract (`HaltingHead`, `ACTBackbone`). Current scope: binary halt/continue head consuming PFC features. Migration target: multi-input interface with separate PFC/HPC projections, external reward signal, and N-action selection. |

### 3.2 Shared Neural-Network Building Blocks

Model-agnostic modules reused across brain-region components.

| Component     | Path                                                            | Responsibility                                                                          |
| ------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Shared NN** | `modules/{attention.py, autoencoder.py, mlp.py, projection.py}` | Attention, autoencoder, MLP (SwiGLU), and projection layers. No brain-region semantics. |

### 3.3 Models

Top-level composed `LightningModule` wrappers. Each model composes brain-region
modules and shared NN blocks, manages explicit recurrent state via dataclasses,
and exposes a step-level forward interface.

| Model      | File               | Status         | Composes                                       |
| ---------- | ------------------ | -------------- | ---------------------------------------------- |
| **TEM v1** | `models/tem_v1.py` | Needs refactor | LEC + MEC + HPC + Autoencoder + Projections    |
| **HRM v1** | `models/hrm_v1.py` | Needs refactor | PFC + STR + partial resets                     |
| **EHC v1** | `models/ehc_v1.py` | Pending        | LEC + MEC + HPC + PFC + STR + shared NN blocks |

### 3.4 Loss

Generic, composable loss primitives operating on flat tensors. Each module
provides stateless functions as the primary API, with optional thin
`nn.Module` wrappers that co-locate a Pydantic config. No model-specific
names, no multi-scale iteration, no orchestration logic.

| Module              | Responsibility                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `cross_entropy.py`  | Token-/observation-level cross-entropy: `stablemax_cross_entropy`, `softmax_cross_entropy`. `(logits, labels, ignore_index) → (*)`.  |
| `consistency.py`    | Representation consistency: `mse_consistency(pred, target) → (B,)`, `nll_consistency(pred, mean, std) → (B,)`. Flat `(B, D)` inputs. |
| `regularization.py` | Activation penalties: L1 sparsity, L2 norm on flat `(B, D)` codes.                                                                   |
| `decision.py`       | Gating losses: BCE for halt/continue. Extensible to N-action selection.                                                              |

### 3.5 Training

Shared training infrastructure. Components are named by algorithmic
function, not by consuming model. Models compose these building blocks in
their `LightningModule.training_step` / `configure_optimizers`.

#### 3.5.1 Components

| Component         | Path(s)             | Responsibility                                                                                          |
| ----------------- | ------------------- | ------------------------------------------------------------------------------------------------------- |
| **Step-Loop**     | `step_loop.py`      | Generic step iteration: `StepLoop`, `StepModule` protocol, `StepContext`.                               |
| **Loss Heads**    | `act_head.py`       | `StepModule` implementations that wire a controller + `loss/` primitives into a step-level contract.    |
| **ACT**           | `act_controller.py` | Adaptive Computation Time (Graves 2016): `ACTController`, `ACTState`, `ACTOutput`, protocol interfaces. |
| **Partial-Reset** | `partial_reset.py`  | Stateful batch assembly: replace completed rows with fresh examples from a buffer.                      |
| **Collector**     | `collector.py`      | Per-step state collection for partial-reset pipelines.                                                  |
| **Buffers**       | `buffers.py`        | Bounded FIFO storage for batch examples.                                                                |
| **Optimizers**    | `optim.py`          | Typed optimizer configs and wrappers (currently `AdamATan2`).                                           |
| **Schedulers**    | `schedules.py`      | LR schedules: `CosineAnnealingLRWithWarmup`, `SequentialLR`, `SchedulerConfig`.                         |

#### 3.5.2 Design Rules

1. **Named by function, not by model.** No model-specific imports inside
   `training/`. If a training module needs model output types, those types
   must be defined in a neutral location (`types.py` or `training/` itself).

2. **Loss heads implement `StepModule`.** Each loss head composes a controller
   (or model callable) with stateless primitives from `loss/` and returns
   `(outputs, carry, done)`. The concrete output dataclass is head-specific
   (e.g., `ACTStepOutput`); `StepLoop` treats it as opaque. Models
   instantiate and wire loss heads in their `LightningModule.__init__`.

3. **Models own the wiring, not the algorithms.** The `LightningModule`
   instantiates modules, controllers, and loss heads, then calls `StepLoop`
   in `training_step`. The training package provides the building blocks.

4. **Extend by addition.** New training algorithms (e.g., advantage estimation,
   replay buffers, policy optimization) are added as new modules in
   `training/`. Existing modules are not modified to accommodate new models.

### 3.6 Data

Environments, datasets, DataModules, and the data processing pipeline.

| Component               | Path                                     | Responsibility                                                             |
| ----------------------- | ---------------------------------------- | -------------------------------------------------------------------------- |
| **Data (package)**      | `data/` (under `ehc_sn`)                 | DataModules, datasets, environment validation, maze vocabulary, collation. |
| **Mazes (package)**     | `mazes/` (under `src/`)                  | Auxiliary maze-environment package.                                        |
| **Data (project root)** | `data/{raw,interim,processed,external}/` | On-disk data directories. Not committed (see data pipeline below).         |

### 3.7 Evaluation

Metrics, trace/rollout collection, and publication-ready visualization.

| Component    | Path        | Responsibility                                                                                                                                                    |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Metrics**  | `metrics/`  | TorchMetrics-based evaluation (accuracy, loss ratios, halting stats). Adapter pattern for model-output → metric update.                                           |
| **Rollouts** | `rollouts/` | Trace collection (`TraceCollector`, `TraceSpec`) and tree-structured rollout data (`TraceTree`). Feeds both training diagnostics and figures.                     |
| **Figures**  | `figures/`  | Publication-ready plotting. Registry pattern (`FigureSpec`, `REGISTRY`), plot modules, sinks (PDF/show), and axis utilities. Uses SciencePlots + pub-ready-plots. |

### 3.8 Infrastructure

Cross-cutting support that wraps external frameworks or provides generic
utilities.

| Component       | Path           | Responsibility                                                                                                                                                                                  |
| --------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Callbacks**   | `callbacks/`   | Lightning `Callback` wrappers (checkpoint, figures). Used by experiments only.                                                                                                                  |
| **Logging**     | `logging/`     | Lightning `TensorBoardLogger` wrapper. Used by experiments only.                                                                                                                                |
| **Activations** | `activations/` | Custom activation functions (e.g., `stablemax`). Consumed by `loss/cross_entropy.py`. Small (single file); lives here rather than in `loss/` because activations are a general-purpose concern. |
| **Utils**       | `utils/`       | Genuine cross-cutting helpers only: tensor operations, normalization, seeding, symmetry transforms, pytree registration, logging config. See §5 Utils Cap.                                      |

---

## 4 Data Pipeline

```
maze-nd (external)          scripts/data-gen/
       │                         │
       ▼                         ▼
  data/raw/  ──────────►  data/processed/  ──────►  ehc_sn.data DataModules
       │                                                │
       └── data/interim/ (optional intermediate)        ▼
                                                   Training loop
```

- **maze-nd**: External tool/library used to generate raw maze structures.
  Dev/script-only dependency (not declared in package `[project.dependencies]`).
- **`scripts/data-gen/build_maze.py`**: CLI script that processes raw mazes
  (add labels, goals, start positions, solve). Uses `typer` (see known
  dependency bug in `spec-requirements.md`).
- **`data/raw/`**: Unprocessed maze-nd output. Not committed to version control.
- **`data/processed/`**: Labeled, solved mazes ready for DataModule consumption.

---

## 5 Utils Cap

`utils/` may contain only genuine cross-cutting helpers with no domain semantics.
Current modules (as of spec creation):

1. `__init__.py` — tensor ops, Gaussian sampling, Hebbian helpers, connection
   logic
2. `logging.py` — logging configuration
3. `norms.py` — normalization functions (RMS norm)
4. `seed.py` — random seed management
5. `symmetry.py` — dihedral symmetry transforms
6. `torch_pytree.py` — PyTorch pytree registration

**Hard cap: 8 modules maximum.** Any addition beyond the cap, or any module
that encodes domain-specific logic (model, loss, data), requires a spec
amendment tracked in `.copilot-tracking/plans/`.

---

## 6 Model Composition Pattern

Each model (`TEM v1`, `HRM v1`) follows this pattern:

1. **Config**: A Pydantic `BaseModel` tree that composes sub-configs for each
   brain-region module.
2. **State**: An explicit frozen/mutable `dataclass` holding per-module recurrent
   state. No hidden state in module attributes.
3. **Step interface**: A `forward(observation, action, state) → (output, state)`
   signature (or equivalent) that makes the recurrent contract explicit.
4. **LightningModule wrapper**: The `models/*.py` file wraps the core `nn.Module`
   with Lightning hooks (`training_step`, `configure_optimizers`, etc.).

---

## 7 Configuration Pattern

- **Component configs**: `pydantic.BaseModel(extra="forbid")` for strictness.
- **CLI entry points**: `pydantic_settings.BaseSettings(cli_parse_args=True)`.
- **Static defaults**: TOML files under `config/`.
- **Experiment configs**: Composed in `experiments/*.py` by nesting component
  configs inside a `RunArguments` settings class.

---

## 8 Forbidden Architectural Patterns

1. **Web-framework layering**: No controllers, views, routers, serializers,
   schemas (in the web sense), or similar patterns from Django/FastAPI/Flask.
2. **Flat utils dump**: `utils/` has a hard cap (§5). Domain logic must live in
   its owning component.
3. **`core`/`common`/`shared` mega-packages**: Do not create catch-all packages
   that aggregate unrelated code.
4. **Wildcard star imports** in non-`__init__` files: `from X import *` is
   forbidden outside `__init__.py` re-exports.
5. **Legacy namespace imports**: No `from torch_tem` or `from hrm_sn` in any
   file under `src/ehc_sn/`.

---

## 9 Migration Status (as of 2026-02-26)

| Item                                   | Status                                                                                                                        |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Namespace consolidation (`ehc_sn`)     | **Complete**. Legacy code archived in `temp/`.                                                                                |
| STR module (`modules/str/`)            | **Minimal**. `LinearHaltingHead` implemented. Protocols (`HaltingHead`, `ACTBackbone`) still in `training/act_controller.py`. |
| STR protocol migration                 | **Pending**. Move `HaltingHead` and `ACTBackbone` protocols from `training/` into `modules/str/`.                             |
| PFC subcomponent generalization        | **Pending**. HRModel is monolithic; target is separated subcomponents.                                                        |
| Training infrastructure generalization | **Pending**. ACT-specific code mixed with shared protocols.                                                                   |
| `config/defaults_ehc.toml`             | **Empty**. Needs population for default experiment configs.                                                                   |
| `README.md`                            | **Populated**. Project overview, install, quick start, layout.                                                                |
