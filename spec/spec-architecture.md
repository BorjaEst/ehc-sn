# EHC-SN Architecture Specification

> Canonical source of truth for component boundaries, vocabulary, and composition
> patterns. See `spec/spec-manifest.toml` for precedence rules.

## 1 Project Identity

**EHC-SN** (_Entorhinal-Hippocampal Complex — Spatial Navigation_) is a research
library for biologically-inspired spatial navigation models built on PyTorch and
Lightning.

**Goal**: Resolve complex navigation tasks (mazes) using advanced neural models
(TEM, HRM, future TRM) that capture how the hippocampal formation (HPC) and
prefrontal cortex (PFC) interact for goal-directed generalization. The project is
inspired by the work of Zheng, Wolf, Ranganath, O'Reilly & McKee (_"Flexible
Prefrontal Control over Hippocampal Episodic Memory for Goal-Directed
Generalization"_).

**Target users**: Computational neuroscience researchers and ML practitioners
studying hippocampal/entorhinal spatial models.

---

## 2 Canonical Import Namespace

The sole canonical import namespace is **`ehc_sn`**.

Legacy namespaces `torch_tem` and `hrm_sn` are retired. The migration is
complete; remnants exist only under `temp/` (archived legacy code, not on the
Python path). No new code may import from `torch_tem` or `hrm_sn`.

---

## 3 Component Taxonomy

Every top-level package under `src/ehc_sn/` maps to exactly one component below.
If a new package is created, this table must be updated.

### 3.1 Brain-Region Modules

These implement neuroscience-grounded circuit components. Each is a `nn.Module`
(or collection of modules) that can be composed by a top-level model.

| Component | Path           | Responsibility                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LEC**   | `modules/lec/` | Sensory encoding and temporal frequency filtering. Transforms raw observations into multi-scale feature codes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **MEC**   | `modules/mec/` | Path integration, grid-cell dynamics, object-vector cells (OVC), and abstract-location projections.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **HPC**   | `modules/hpc/` | Hebbian associative memory, attractor dynamics, grounded-location inference, and place-code maintenance.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **PFC**   | `modules/pfc/` | Biological working-memory and adaptive computation time (ACT) reasoning. Models the prefrontal cortex's role in goal-directed control over episodic memory. The current implementation (`HRModel`) is a two-level recurrent architecture with high-level state $z_H$ and low-level state $z_L$ updated in alternating cycles, composed of transformer blocks, SwiGLU MLPs, and a linear halting head. This is the first implementation approach; the target is to generalize into clearly separated subcomponents (e.g., working-memory buffer, reasoning stack, halting mechanism) while preserving the single biological design. PFC is **not** intended to hold a family of alternative architectures. |

### 3.2 Shared Neural-Network Building Blocks

Model-agnostic modules reused across brain-region components.

| Component     | Path                                                            | Responsibility                                                                          |
| ------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Shared NN** | `modules/{attention.py, autoencoder.py, mlp.py, projection.py}` | Attention, autoencoder, MLP (SwiGLU), and projection layers. No brain-region semantics. |

### 3.3 Models

Top-level composed `LightningModule` wrappers. Each model composes brain-region
modules and shared NN blocks, manages explicit recurrent state via dataclasses,
and exposes a step-level forward interface.

| Model      | File               | Status              | Composes                                        |
| ---------- | ------------------ | ------------------- | ----------------------------------------------- |
| **TEM v1** | `models/tem_v1.py` | Active              | LEC + MEC + HPC + Autoencoder + Projections     |
| **HRM v1** | `models/hrm_v1.py` | Active              | PFC (HRModel) + ACT controller + partial resets |
| **TRM v1** | `models/trm_v1.py` | Placeholder (empty) | TBD                                             |

### 3.4 Loss

Per-model loss computation. Lives as a top-level sibling (`loss/`), not nested
under `training/`.

| Component | Path    | Responsibility                                                                                                                |
| --------- | ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Loss**  | `loss/` | Loss heads: sensory reconstruction loss ($L_x$, $L_g$, $L_p$) for TEM; ACT loss head for HRM; shared cross-entropy utilities. |

### 3.5 Training

Shared training infrastructure and model-specific training extensions.

| Component    | Path        | Responsibility                                                                                                                                                                                                                           |
| ------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Training** | `training/` | Shared: step-loop protocol (`StepLoop`, `StepModule`), optimizer configs (`AdamATan2Config`), LR schedulers (`CosineAnnealingLRWithWarmup`, `SequentialLR`). Model-specific: ACT controller, partial-reset batch assembler, FIFO buffer. |

**Migration target**: Model-specific training code (ACT controller, partial
resets, FIFO buffer) currently lives alongside shared infrastructure. The target
is a pluggable/registerable pattern where model-specific training extensions are
clearly namespaced (e.g., `training/hrm/`) while shared protocols remain at the
`training/` root.

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

Each model (`TEM v1`, `HRM v1`, future `TRM v1`) follows this pattern:

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

## 9 Migration Status (as of 2025-02-25)

| Item                                   | Status                                                                 |
| -------------------------------------- | ---------------------------------------------------------------------- |
| Namespace consolidation (`ehc_sn`)     | **Complete**. Legacy code archived in `temp/`.                         |
| `trm_v1.py`                            | **Placeholder** (empty file).                                          |
| PFC subcomponent generalization        | **Pending**. HRModel is monolithic; target is separated subcomponents. |
| Training infrastructure generalization | **Pending**. ACT-specific code mixed with shared protocols.            |
| `config/defaults_ehc.toml`             | **Empty**. Needs population for default experiment configs.            |
| `README.md`                            | **Populated**. Project overview, install, quick start, layout.         |
