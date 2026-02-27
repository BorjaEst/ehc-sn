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
Legacy namespaces (`torch_tem`, `hrm_sn`) are retired and archived under
`temp/` (not on the Python path).

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

Each model is a self-contained unit living in `models/` as a flat file (one
file per model version). A model file co-locates:

- A **pure `nn.Module`** — framework-agnostic forward pass.
- One or more **`LightningModule` trainers** — model-specific training wiring.
- **Pydantic configs** — architecture config and training config.
- **State dataclasses** — explicit recurrent state.

| Model      | `nn.Module`  | `LightningModule` | Config      | State      | Composes                                       | Status         |
| ---------- | ------------ | ----------------- | ----------- | ---------- | ---------------------------------------------- | -------------- |
| **TEM v1** | `TEMModelV1` | `TEMTrainerV1`    | `TEMConfig` | `TEMState` | LEC + MEC + HPC + Autoencoder + Projections    | Needs refactor |
| **HRM v1** | `HRMModelV1` | `HRMTrainerV1`    | `HRMConfig` | `HRMState` | PFC + STR                                      | Needs refactor |
| **EHC v1** | `EHCModelV1` | `EHCTrainerV1`    | `EHCConfig` | `EHCState` | LEC + MEC + HPC + PFC + STR + shared NN blocks | `NOT_STARTED`  |

**Multiple trainers per model.** E.g., `EHCModelV1` might have both
`EHCTrainerV1` (RL) and `EHCPretrainV1` (supervised). All live in the
same model file. Models compose modules; they do not subclass them.
See §6.6 for import-direction rules.

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

100% generic algorithmic building blocks — no model-specific code,
no `LightningModule` implementations, no model imports.

| Component         | Path(s)             | Paradigm   | Responsibility                                                                                                |
| ----------------- | ------------------- | ---------- | ------------------------------------------------------------------------------------------------------------- |
| **Step-Loop**     | `step_loop.py`      | Generic    | Generic step iteration: `StepLoop`, `StepModule` protocol, `StepContext`.                                     |
| **Loss Heads**    | `act_head.py`       | Generic    | `StepModule` implementations that wire a controller + `loss/` primitives into a step-level contract.          |
| **ACT**           | `act_controller.py` | Generic    | Adaptive Computation Time (Graves 2016): `ACTController`, `ACTState`, `ACTOutput`, protocol interfaces.       |
| **Partial-Reset** | `partial_reset.py`  | Generic    | Stateful batch assembly: replace completed rows with fresh examples from a buffer.                            |
| **Collector**     | `collector.py`      | Generic    | Per-step state collection for partial-reset pipelines.                                                        |
| **Buffers**       | `buffers.py`        | Generic    | Bounded FIFO storage for batch examples.                                                                      |
| **Optimizers**    | `optim.py`          | Generic    | Typed optimizer configs and wrappers (currently `AdamATan2`).                                                 |
| **Schedulers**    | `schedules.py`      | Generic    | LR schedules: `CosineAnnealingLRWithWarmup`, `SequentialLR`, `SchedulerConfig`.                               |
| **Supervised**    | `supervised.py`     | Supervised | Curriculum scheduling, label-smoothing helpers, supervised step patterns.                                     |
| **RL**            | `rl.py`             | RL         | `compute_gae()`, `policy_gradient_loss()`, advantage estimation, rollout buffer utils, discount calculations. |
| **ELBO**          | `elbo.py`           | VAE / ELBO | KL divergence utilities, ELBO loss aggregation, reconstruction + KL balancing, annealing schedules.           |

> **RULE**: Nothing in `training/` may import from `models/` or `modules/`.

**Named by function.** Root-level files are named by algorithmic function
(`act_controller.py`, `buffers.py`). Regime files are named by paradigm
(`supervised.py`, `rl.py`, `elbo.py`).

### 3.6 Data

Data pipeline: maze generation, on-disk storage, dataset loading,
gymnasium environments, and Lightning DataModules. Three sub-layers
with strict top-down imports (no reverse dependency):
`data/*.py` (torch + lightning) → `data/envs/` (gymnasium) → `data/mazes/` (numpy only).

#### 3.6.1 Mazes (`data/mazes/`)

Pure maze infrastructure with **no ML or framework dependencies**. This
sub-package defines maze structures, generation wrappers, augmentation
operations, and I/O for the canonical on-disk format.

| Module        | Responsibility                                                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `types.py`    | `MazeGraph`, `Cell`, `Wall`, grid metadata. No neural types.                                                                                           |
| `generators/` | Thin wrappers around external generators: `maze_nd.py`, `dungeongen.py`, `huggingface.py`.                                                             |
| `ops.py`      | Pure augmentation functions: `solve()`, `add_start_goal()`, `generate_observations()`, `add_landmarks()`. Operate on `MazeGraph` or raw channel dicts. |
| `io.py`       | Read/write canonical channel NPZ files and JSONL index entries.                                                                                        |

**Dependency rule:** `ehc_sn.data.mazes` may import only stdlib, `numpy`, and
the declared generator libraries (`maze-nd`, `dungeongen`, `huggingface_hub`).
It must **not** import `torch`, `gymnasium`, `lightning`, or any other
`ehc_sn` subpackage.

#### 3.6.2 Canonical On-Disk Format

Processed maze data lives in `data/processed/` as **one NPZ file per maze**
plus a JSONL index. Each NPZ contains a `dict[str, numpy.ndarray]` of named
channels with heterogeneous dtypes (channels vary between `bool` and `int32`).
Optional channels are represented by key absence, not by zero-filled arrays.

**Mandatory channel:**

| Channel  | Name       | dtype  | Shape    | Description                                |
| -------- | ---------- | ------ | -------- | ------------------------------------------ |
| Topology | `topology` | `bool` | `(H, W)` | Passable cells (`True`) vs walls (`False`) |

**Optional channels:**

| Channel      | Name           | dtype   | Shape    | Description                                   | Primary consumers                                                |
| ------------ | -------------- | ------- | -------- | --------------------------------------------- | ---------------------------------------------------------------- |
| Observations | `observations` | `int32` | `(H, W)` | Unique observation ID per cell                | TEM, EHC                                                         |
| Start        | `start`        | `bool`  | `(H, W)` | Start position(s)                             | HRM, EHC                                                         |
| Goals        | `goals`        | `bool`  | `(H, W)` | Goal position(s)                              | HRM, EHC                                                         |
| Landmarks    | `landmarks`    | `int32` | `(H, W)` | Special object IDs ("shiny")                  | TEM, EHC                                                         |
| Solution     | `solution`     | `int32` | `(H, W)` | Shortest-path distance or step labels         | HRM (supervision)                                                |
| Regions      | `regions`      | `int32` | `(H, W)` | Room/region ID                                | Future                                                           |
| Valid mask   | `mask_valid`   | `bool`  | `(H, W)` | Legal agent positions (explicit reachability) | All (when topology alone is insufficient, e.g. dungeongen voids) |

**JSONL index** (`data/processed/index.jsonl`): one JSON object per line.

| Field            | Type        | Description                                |
| ---------------- | ----------- | ------------------------------------------ |
| `id`             | `str`       | Unique maze identifier                     |
| `file`           | `str`       | Relative path to NPZ file                  |
| `source`         | `str`       | Generator that produced the raw maze       |
| `split`          | `str`       | Dataset split: `train`, `val`, or `test`   |
| `height`         | `int`       | Grid height                                |
| `width`          | `int`       | Grid width                                 |
| `channels`       | `list[str]` | Channel names present in the NPZ           |
| `n_observations` | `int`       | Observation vocabulary size (0 if absent)  |
| `n_goals`        | `int`       | Number of goal cells (0 if absent)         |
| `difficulty`     | `str`       | Source-defined difficulty label (optional) |

**On-disk layout:**

```text
data/
├── raw/                          # Untouched generator output
│   ├── maze-nd/
│   ├── dungeongen/
│   └── huggingface/
├── interim/                      # Source-specific augmented output
│   ├── maze-nd/                  #   (solved, start/goal added, etc.)
│   ├── dungeongen/
│   └── huggingface/
└── processed/                    # Canonical channel NPZ format
    ├── index.jsonl
    ├── train/
    │   ├── maze_00001.npz
    │   └── ...
    ├── val/
    └── test/
```

#### 3.6.3 Data Pipeline Modules (`data/`)

ML data infrastructure: datasets, DataModules, environments, and collation.

| Module           | Responsibility                                                                                       |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| `schema.py`      | Channel name constants, dtype contracts, and validation for the canonical format.                    |
| `index.py`       | JSONL index parsing, dataset splitting, channel-availability queries.                                |
| `datasets.py`    | Map-style `torch.utils.data.Dataset` subclasses loading NPZ → tensors.                               |
| `datamodules.py` | Lightning `DataModule` implementations. One per model or a unified module with mode selection.       |
| `collation.py`   | Model-specific batch assembly (`WalkBatch` for TEM, `Dict[str, Tensor]` for HRM, RL episode format). |
| `vocabulary.py`  | Observation token mappings (current `maze_vocab`).                                                   |
| `transforms.py`  | Channel → tensor conversions, normalization, augmentation at load time.                              |

#### 3.6.4 Gymnasium Environments (`data/envs/`)

Gymnasium wrappers over processed maze NPZs. Base environment returns
a rich observation dict; model-specific `ObservationWrapper` subclasses
adapt it to each model's expected input.

| Module        | Responsibility                                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `maze_env.py` | Base `MazeEnv(gymnasium.Env)`: loads canonical NPZ, manages agent position, computes reward from goals. Returns rich obs dict. |
| `wrappers.py` | `ObservationWrapper` subclasses: `TEMObsWrapper` (one-hot vectors), `HRMObsWrapper` (token sequences), etc.                    |

#### 3.6.5 Model–Data Consumption Paths

| Model   | Data path                                         | Interaction mode  |
| ------- | ------------------------------------------------- | ----------------- |
| **TEM** | NPZ → `MazeEnv` + `TEMObsWrapper` → runtime walks | Online (env.step) |
| **HRM** | NPZ → `PuzzleDataset` → `DataLoader`              | Offline (static)  |
| **EHC** | NPZ → `MazeEnv` → RL episodes                     | Online (env.step) |

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

```text
Generators (maze-nd, dungeongen, HF)      scripts/data-gen/
          │                                      │
          ▼                                      │ augmentation calls
     data/raw/                                   │ (ehc_sn.data.mazes.ops)
          │                                      │
          ▼                                      ▼
     data/interim/  ◄────── source-specific augmentation
          │                 (solve, add start/goal, generate obs IDs)
          ▼
     data/processed/  ◄──── canonical NPZ + index.jsonl
          │
          ├──► ehc_sn.data.datasets    (static)  ──► HRM DataModule
          │
          └──► ehc_sn.data.envs.MazeEnv (runtime) ──► TEM DataModule
                    │                               ──► EHC RL training
                    └──► ObservationWrappers
```

### 4.1 Raw Sources

| Source          | Output                       | Dependency        |
| --------------- | ---------------------------- | ----------------- |
| `maze-nd`       | Grid graph with connectivity | `maze-nd`         |
| `dungeongen`    | Room-based layouts           | `dungeongen`      |
| HuggingFace Hub | Pre-built maze datasets      | `huggingface_hub` |

All three are declared runtime dependencies in `pyproject.toml`. Raw output
is stored in `data/raw/<source>/` and is **not** committed to version control.

### 4.2 Augmentation (raw → interim)

Source-specific augmentation scripts in `scripts/data-gen/` call pure
functions from `ehc_sn.data.mazes.ops`:

- **dungeongen** → add start/goal positions, optionally compute shortest paths.
- **maze-nd** → solve maze, generate observation IDs, assign landmarks.
- **HuggingFace** → relabel goals, extract topology from images, normalize
  format.

Scripts are thin CLI orchestrators; all logic lives in `ehc_sn.data.mazes.ops`.
Output is stored in `data/interim/<source>/`.

### 4.3 Canonicalization (interim → processed)

A final processing step converts augmented interim data into the canonical
channel NPZ format defined in §3.6.2. This step is source-agnostic: it reads
whatever channels are available and writes a conformant NPZ with a
corresponding JSONL index entry.

Output is stored in `data/processed/{train,val,test}/`.

---

## 5 Utils

Path: `utils/`

### 5.1 Dependency Rule

`utils/` must **not** import from `ehc_sn` or any of its subpackages.
External dependencies (PyTorch, NumPy, SciPy, stdlib) are allowed.

A function that requires an `ehc_sn` type or module is domain logic and
belongs in the component that owns that type.

### 5.2 Scope

Generic, reusable helpers with no brain-region, model, or training semantics.
A function belongs here only if it could be moved to an unrelated ML project
unchanged.

### 5.3 Allowed Concerns

| Concern           | Examples                                              |
| ----------------- | ----------------------------------------------------- |
| Tensor ops        | Shape manipulation, one-hot encoding, reductions.     |
| Initialization    | Weight init (truncated normal, Xavier, etc.).         |
| Linear algebra    | Projection matrices, downsampling, encoding tables.   |
| Normalization     | Stateless norms (RMS norm, layer norm).               |
| Reproducibility   | Seeding, deterministic mode.                          |
| Filesystem / IO   | Path resolution, directory creation, file validation. |
| Logging config    | Logger setup and formatting.                          |
| Framework helpers | PyTree registration, device/dtype utilities.          |
| Geometry          | Symmetry transforms, coordinate conversions.          |

### 5.4 Exclusions

The following do **not** belong in `utils/`:

- Functions referencing brain regions (HPC, MEC, LEC, PFC, STR).
- Functions operating on project types (`LocationBelief`, `MultiScaleCode`, etc.).
- Loss or metric computation (→ `loss/`, `metrics/`).
- Activation functions (→ `activations/`).
- Data loading or dataset logic (→ `data/`).

---

## 6 Model Composition Pattern

Each model is composed of co-located artifacts in `models/`: a **config**,
a **pure `nn.Module`**, a **state dataclass**, and one or more **trainers**
(`LightningModule`). Generic training infrastructure lives in `training/`.

### 6.1 Separation of Concerns

| Layer          | Contains                                      | Knows about                                               |
| -------------- | --------------------------------------------- | --------------------------------------------------------- |
| `experiments/` | CLI parsing, Trainer construction, seed       | Everything (top of the DAG)                               |
| `models/`      | nn.Module + LightningModule + configs + state | `modules/`, `training/`, `loss/`, `rollouts/`, `types.py` |
| `training/`    | Generic algorithms + paradigm building blocks | `loss/`, `types.py`, peers in `training/`                 |
| `modules/`     | Brain-region and shared nn.Modules            | Peers in `modules/`, `types.py`, `utils/`                 |

### 6.2 Configuration Hierarchy

Configuration is split by concern. Architectural parameters live with the
model; training parameters live with the regime.

| Config type           | Scope                                                    | Lives in                        | Example                  |
| --------------------- | -------------------------------------------------------- | ------------------------------- | ------------------------ |
| **Model config**      | Architecture only: dimensions, layer counts, activations | `models/*.py` or `modules/*.py` | `TEMConfig`, `HRMConfig` |
| **Training config**   | Optimizer, LR schedule, loss weights, buffer sizes       | `models/*.py` (with trainer)    | `HRMTrainingConfig`      |
| **Data config**       | Dataset paths, batch size, workers, augmentation         | `data/*.py`                     | `PuzzleDatamoduleConfig` |
| **Experiment config** | Composes all of the above + Trainer knobs                | `experiments/*.py`              | `RunArguments`           |

Model configs use `pydantic.BaseModel(extra="forbid")`. Architectural
dimensions that must not change after construction use `frozen=True`.

Training configs are kept separate from model configs. A trainer receives
both but passes only the architecture config to the `nn.Module` constructor.

### 6.3 State Management

Each model defines an explicit `dataclass` for its recurrent state.

- State is passed into and returned from `forward()`.
- No hidden state is stored in module attributes between calls.
- State dataclasses provide `detach()` (for TBPTT truncation, etc.).
- Composed models nest sub-states (e.g., `TEMState` contains `LECState`,
  `MECState`, `HPCState`; `EHCState` would contain `TEMState` +
  `HRMState` or their components).

### 6.4 Step Interface

Each model exposes a step-level `forward()` that processes one timestep:

```python
def forward(self, ..., state: ModelState) -> tuple[ModelState, Logits, Features]:
    ...
```

The trainer's `training_step` iterates over `StepLoop`, yielding
`(t, step_output)` pairs per timestep.

### 6.5 Module Reuse Protocol

Module configs use stable, module-oriented field names across all
models that share that module.

- The Pydantic type behind the field is the same class in every model
  that uses that module.
- Model-specific fields (fields that only one model needs) are named
  freely, but must not collide with shared module field names.

Pre-trained module weights can be loaded from a standalone model's
checkpoint into a composite model. The protocol is:

1. Module state_dicts are extractable from a parent checkpoint
   by parameter name prefix (e.g., model.hpc.\*).
2. When a checkpoint path is provided, the model's **init** loads
   the extracted weights into the corresponding sub-module.
3. Composite model configs include optional checkpoint path fields:

```python
hpc_checkpoint: Optional[Path] = None
pfc_checkpoint: Optional[Path] = None
```

Both forms of reuse require the same prerequisite: module configs and
module parameter names must be consistent across all models that share
those modules.

### 6.6 Import-Direction Rules

External libraries declared in `pyproject.toml` may be imported freely.
Internal imports follow a strict top-down DAG — import from your own
layer or below, **never** upward.

| Layer | Components                                                                                               |
| ----- | -------------------------------------------------------------------------------------------------------- |
| **4** | `experiments/`                                                                                           |
| **3** | `models/`                                                                                                |
| **2** | `modules/`, `training/`, `loss/`, `metrics/`, `rollouts/`, `figures/`, `callbacks/`, `logging/`, `data/` |
| **1** | `activations/`, `utils/`, `types.py`                                                                     |

#### Additional constraints

| Rule | Constraint                                                              |
| ---- | ----------------------------------------------------------------------- |
| R1   | `training/` must not import from `modules/` (same layer, but forbidden) |
| R2   | `modules/` must not import from `training/` (same layer, but forbidden) |
| R3   | `data/` must not import from `modules/` or `training/`                  |
| R4   | `utils/` must not import from any `ehc_sn` subpackage (layer 1 rule)    |
| R5   | Peer imports within a component (e.g., `modules/hpc/` → `modules/mec/`) |
|      | are allowed                                                             |

---

## 7 Configuration Pattern

### 7.1 Config Types

| Type                    | Base class                                            | Scope                                         | Lives in                     | Example                  |
| ----------------------- | ----------------------------------------------------- | --------------------------------------------- | ---------------------------- | ------------------------ |
| **Component config**    | `pydantic.BaseModel(extra="forbid")`                  | Single-component settings                     | Same file as the `nn.Module` | `AttractorSettings`      |
| **Model config**        | `pydantic.BaseModel(extra="forbid")`                  | Architecture: dimensions, layers, activations | `models/*.py`                | `TEMConfig`, `HRMConfig` |
| **Training config**     | `pydantic.BaseModel(extra="forbid")`                  | Optimizer, LR schedule, loss weights, buffers | `models/*.py` (with trainer) | `HRMTrainingConfig`      |
| **Data config**         | `pydantic.BaseModel(extra="forbid")`                  | Dataset paths, batch size, workers            | `data/*.py`                  | `PuzzleDatamoduleConfig` |
| **Experiment settings** | `pydantic_settings.BaseSettings(cli_parse_args=True)` | Composes all above + Trainer knobs            | `experiments/*.py`           | `RunArguments`           |

Architectural dimensions use `frozen=True`. Training configs are separate
from model configs; the trainer passes only the architecture config to
the `nn.Module` constructor.

### 7.2 Static Defaults

TOML files under `config/` provide default values. The experiment
entry point loads defaults first, then CLI arguments override:

```python
defaults = tomllib.load(Path("config/defaults_ehc.toml").open("rb"))
settings = RunArguments(**defaults)  # CLI overrides via pydantic_settings
```

### 7.3 Leaf Composition Pattern

Experiment settings are structured as a flat tree of **leaf configs**.
Composed configs for downstream consumers (model, datamodule, regime) are
assembled via `@property` methods using `model_validate(self, from_attributes=True)`.

```text
RunArguments(BaseSettings)                    ← CLI + TOML
  ├── architecture: HRMConfig                 ← model architecture (frozen dims)
  ├── loss: ACTLossConfig                     ← loss weights / targets
  ├── optimizer: AdamATan2Config              ← optimizer hyperparams
  ├── scheduler: SchedulerConfig              ← LR schedule
  ├── dataset: PuzzleDatasetSettings          ← dataset paths, seed
  ├── global_batch_size: int                  ← shared by model + data
  ├── logger: LoggerSettings                  ← TensorBoard config
  ├── checkpoint: CheckpointSettings          ← checkpoint config
  │
  └── composed via @property:
      ├── .model → ModelConfig_HRM_V1         (architecture + loss + optimizer + ...)
      └── .datamodule → PuzzleDatamoduleConfig (dataset + batch_size + workers + ...)
```

This pattern ensures:

- **CLI ergonomics**: Any leaf field can be overridden from the command line
  (e.g., `--optimizer.lr=1e-4`, `--epochs=100`).
- **No duplication**: Shared fields (e.g., `global_batch_size`) are defined
  once and composed into multiple downstream configs.
- **Reproducibility**: The full `RunArguments` can be serialized to
  reproduce any experiment.

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
