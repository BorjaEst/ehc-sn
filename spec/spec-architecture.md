# EHC-SN Architecture Specification

> Canonical source of truth for component boundaries, vocabulary, and
> composition patterns.

## 1 Project Identity

**EHC-SN** (_Entorhinal-Hippocampal Circuit — Spatial Navigation_) is a research
library for biologically-inspired spatial navigation models built on PyTorch and
Lightning.

**Goal**: Study navigation architectures built from TEM-like episodic-memory
backbones, HRM-like recurrent reasoning modules, and explicit
controller/policy layers for goal-directed evaluation. The project is inspired
by the work of Zheng, Wolf, Ranganath, O'Reilly & McKee (_"Flexible
Prefrontal Control over Hippocampal Episodic Memory for Goal-Directed
Generalization"_).

**TEM** (_Tolman-Eichenbaum Machine_) is a multi-scale spatial memory backbone
that composes LEC, MEC, and HPC modules. By itself it updates and queries
structured latent state; it does not define benchmark-time action selection.

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

## 3 Import and Dependency Rules

Any component may import external dependencies declared in `pyproject.toml`.
Internal imports follow a top-down DAG: import from your own layer or below,
**never** upward.

Layer 4: `experiments/`, `scripts/benchmarks/`

Layer 3: `models/`, `benchmarks/`, `lightning/`

Layer 2: `modules/`, `controllers/`, `policies/`, `heads/`, `training/`, `loss/`, `metrics/`, `rollouts/`, `figures/`, `callbacks/`, `logging/`, `data/`, `envs/`

Layer 1: `activations/`, `utils/`, `types.py`

**Additional constraints:**

R1: `training/` must not import from `modules/`, `models/`, or `lightning/`

R2: `modules/` must not import from `training/`, `controllers/`, `heads/`, `models/`, or `lightning/`

R3: `controllers/` and `heads/` must not import from `models/` or `lightning/`

R4: `data/` must not import from `modules/`, `training/`, `controllers/`, `heads/`, `models/`, or `lightning/`

R5: `utils/` must not import from any `ehc_sn` subpackage

R6: Peer imports within a component (for example, `modules/hpc/` → `modules/mec/`) are allowed

R7: `policies/` must not import from `models/`, `lightning/`, `controllers/`, `heads/`, `training/`, or `modules/`

R8: `envs/` must not import from `models/`, `lightning/`, `controllers/`, `heads/`, or `training/`

R9: `benchmarks/` is a reusable benchmark component in Layer 3. Its benchmark-semantic subpackages (`b0/`-`b3/`), `_capabilities/`, and `_infra/` must remain model-agnostic and must not import from `models/`, `lightning/`, `controllers/`, `heads/`, or `training/`. Its internal `_bindings/` subpackage may import from `models/` plus lower reusable layers to adapt pure models to benchmark capability protocols. No code outside `benchmarks/` may import from `benchmarks/_*`. Layer 4 entrypoints may import benchmark evaluator and builder surfaces from `benchmarks/`.

R10: `lightning/` is a reusable training-execution component in Layer 3. It may import from `models/` and lower reusable layers. It owns model-aware Lightning training surfaces, optimizer assembly, training hooks, and training-time checkpoint-restore glue. `lightning/` must not own canonical benchmark definitions, benchmark bindings, or benchmark job scheduling. Layer 4 entrypoints may import `lightning/`. Components in Layers 1-2 and `benchmarks/` must not import upward from `lightning/`.

---

## 4 Component Taxonomy

Every top-level package under `src/ehc_sn/` maps to exactly one component below.
If a new package is created, this table must be updated.

### 4.1 Brain-Region Modules

These implement neuroscience-grounded circuit components. Each is a `nn.Module`
(or collection of modules) that can be composed by a top-level model. The
canonical table may include planned modules required by the target
architecture even before implementation lands in `src/ehc_sn/`; such rows must
state that status explicitly.

| Component | Path           | Biological Role                                                                                                                                                          | Computational Responsibility                                                                                                                                                                                                                                                                   |
| --------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LEC**   | `modules/lec/` | Lateral entorhinal cortex (LEC): sensory encoding and temporal frequency filtering.                                                                                      | Transforms raw observations into multi-scale feature codes.                                                                                                                                                                                                                                    |
| **MEC**   | `modules/mec/` | Medial entorhinal cortex (MEC): path integration, grid-cell spatial coding, and object-vector cell (OVC) representations.                                                | Grid-cell dynamics, abstract-location projections, and OVC encoding.                                                                                                                                                                                                                           |
| **HPC**   | `modules/hpc/` | Hippocampus (HPC): episodic memory formation, pattern completion via attractor dynamics, and place-cell spatial coding.                                                  | Hebbian associative memory, attractor retrieval, and grounded-location inference.                                                                                                                                                                                                              |
| **PFC**   | `modules/pfc/` | Prefrontal cortex (PFC): working memory maintenance and goal-directed reasoning over episodic memory.                                                                    | Two-level recurrent architecture ($z_H$, $z_L$) with transformer blocks and alternating update cycles.                                                                                                                                                                                         |
| **BG**    | `modules/bg/`  | Basal-ganglia arbitration module (planned): loop-level arbitration across admissible internal-operation, motor, modifier, and submit channels within broader cortico-basal-ganglia-thalamo-cortical control. | Planned `nn.Module`(s) owning explicit bundle-score integration and arbitration. BG combines cortical candidate scores, explicit admissibility structure, and optional striatal reward-sensitive bias inputs into policy-ready channel or bundle scores. BG does not sample actions, step environments, or replace reusable policy layers. |
| **STR**   | `modules/str/` | Striatum (STR): reward-learning and reward-sensitive bias interface inside broader basal-ganglia arbitration. Receives summary projections from PFC and may later receive richer cortical or hippocampal inputs. | `nn.Module`(s) defining STR's public contract as reward prediction and optional reward-sensitive bias signals. Current scope: scalar immediate-reward prediction from detached PFC summary features plus cortical action/policy scores. STR must not own admissibility logic, bundle arbitration, generic policy sampling, or the full action ontology. Migration target: recurrent reward-learning state and bias heads consumed by a separate BG arbitration module. |

### 4.2 Shared Neural-Network Building Blocks

Model-agnostic modules reused across brain-region components.

| Component     | Path                                                                  | Responsibility                                                                          |
| ------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Shared NN** | `modules/{attention.py, autoencoder.py, mlp.py, projection.py, etc.}` | Attention, autoencoder, MLP (SwiGLU), and projection layers. No brain-region semantics. |

### 4.3 Models

Each model is a self-contained architecture unit living in `models/` as a flat
file (one file per model version). A model file co-locates:

- A **pure `nn.Module`** — framework-agnostic forward pass.
- **Pydantic architecture config(s)** — dimensions, structural options, and
  module-composition settings required to construct the pure model.
- **State dataclasses** — explicit recurrent state.
- Optional model-local helper functions required to construct or adapt the pure
  model, provided they do not introduce training, benchmark, checkpoint-loading,
  or environment-wiring orchestration.

A model file does **not** define Lightning trainers, benchmark evaluators,
benchmark bindings, checkpoint-loading surfaces, dataset/env assembly, or
task-specific policy-factory resolution. Those executable surfaces live outside
`models/`, primarily in `lightning/` and `benchmarks/`.

| Model      | `nn.Module`  | Architecture Config | State      | Composes                                       | Status         |
| ---------- | ------------ | ------------------- | ---------- | ---------------------------------------------- | -------------- |
| **TEM v1** | `TEMModelV1` | `ModelSettings_V1`  | `TEMState` | LEC + MEC + HPC + Autoencoder + Projections    | Needs refactor |
| **HRM v1** | `HRMModelV1` | `PFCSettings`       | `HRMState` | PFC + STR                                      | Needs refactor |
| **EHC v1** | `EHCModelV1` | `EHCConfig`         | `EHCState` | LEC + MEC + HPC + PFC + BG + STR + shared NN blocks | `NOT_STARTED`  |

**Multiple execution surfaces per model.** A model may be consumed by multiple
training or benchmark surfaces. For example, one model may have both RL and
supervised Lightning training surfaces, plus one or more benchmark bindings
consumed by benchmark evaluators, all defined outside `models/`. Models compose
modules; they do not subclass them.

Canonical EHC separation: PFC maintains and scores candidate control content,
STR supplies reward-learning and optional reward-sensitive bias signals, BG owns
loop-level arbitration over admissible channel bundles, policies sample from
declared score tensors, and controllers execute rollout/environment mechanics.
Direct PFC-to-policy shortcuts are interim implementation paths, not the target
architecture boundary.

### 4.4 Lightning

`lightning/` owns model-aware training execution surfaces that adapt pure
models plus lower-layer reusable infrastructure into concrete training
workflows.

Responsibilities:

- Adapt `models/` plus lower-layer reusable primitives into Lightning training
  surfaces.
- Own model-aware training wiring, including optimizer/scheduler assembly,
  training/validation hooks, and trainer-local checkpoint-restore glue.
- Own training-local configuration that is not part of the pure architecture
  contract.
- Remain reusable across entrypoints that need the same training surface.

Constraints:

- `lightning/` may import from `models/` and lower reusable layers.
- `lightning/` must not own canonical benchmark definitions, benchmark
  bindings, or benchmark job scheduling; that belongs in `benchmarks/`.
- `lightning/` must not own generic training primitives; that belongs in
  `training/`.
- `lightning/` must not own reusable scripted or learned action-selection
  policies; that belongs in `policies/`.

### 4.5 Loss

Generic, composable loss primitives operating on flat tensors. Each module
provides stateless functions as the primary API, with optional thin
`nn.Module` wrappers that co-locate a Pydantic config. No model-specific
names, no multi-scale iteration, no orchestration logic.

| Module              | Responsibility                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `cross_entropy.py`  | Token-/observation-level cross-entropy: `stablemax_cross_entropy`, `softmax_cross_entropy`. `(logits, labels, ignore_index) → (*)`.  |
| `consistency.py`    | Representation consistency: `mse_consistency(pred, target) → (B,)`, `nll_consistency(pred, mean, std) → (B,)`. Flat `(B, D)` inputs. |
| `regularization.py` | Activation penalties: L1 sparsity, L2 norm on flat `(B, D)` codes.                                                                   |
| `decision.py`       | Decision and selection losses over admissible actions or channel bundles. Legacy halt/continue BCE remains allowed as a narrow special case, but canonical designs should support structured policy-logit supervision and RL-compatible selection losses. |

### 4.6 Runtime Orchestration

Runtime orchestration is split into controllers and heads so action/halting
mechanics and per-step loss composition can evolve independently while staying
outside `models/`.

#### 4.6.1 Controllers

Controllers own recurrent rollout carry, slot refresh/reset behavior, and the
algorithm-specific step policy for a model backbone. They are model-agnostic:
no imports from `models/`, no Lightning code, and no dataset ownership.

`ACT`: `controllers/act.py`.
Adaptive Computation Time rollout control: halting policy, recurrent carry, TD bootstrap targets.

`RL`: `controllers/rl.py`.
Environment-coupled rollout control: policy invocation over admissibility-aware action views, env stepping, recurrent carry, and done propagation.

Controllers execute declared model outputs. They must not absorb loop-level
bundle-score integration that belongs to an explicit BG or arbitration module.

Shared rollout-state helpers and thin controller bases live in the same
component when they exist only to support these controllers.

#### 4.6.2 Heads

Heads adapt a controller to the `StepModule` contract and assemble per-step
losses, metrics, and diagnostic signals. They may import from `controllers/`,
`loss/`, `metrics/`, and `training/`, but not from `models/`.

`ACT`: `heads/act.py`.
Supervised ACT losses, halted/token aggregation, and ACT-specific diagnostics.

`RL`: `heads/rl.py`.
Actor-critic losses, supervised token loss, halted/token aggregation, diagnostics.

Shared head utilities and thin base heads live in the same component when they
serve multiple head variants without introducing model semantics.

#### 4.6.3 Policies

Policies own reusable action-selection logic over rollout-state views. They are
not environments and not controllers: they do not advance environment state,
manage rollout carry, or compute losses. Their job is limited to selecting an
action from an explicit policy input and owning policy-local randomness.

`policies/`
Reusable scripted or learned action-selection strategies.

Responsibilities:

- Define a narrow public policy protocol for action selection.
- Define typed policy-input/config contracts when reuse across controllers is intended.
- Own policy-local RNG and exploration semantics.
- Apply explicit admissibility information supplied by controller or environment, including hard feasibility masks and documented logical-compatibility constraints.
- Remain reusable across TEM, RL, scripted evaluation, and data-generation workflows.

Constraints:

- Policies consume typed rollout-state views or documented tensor mappings.
- Policies may sample from score tensors emitted by PFC, BG, or other declared model modules, but they must not absorb loop-level bundle-score integration or broader arbitration responsibilities.
- Policies must treat hard constraints as explicit inputs rather than inferring them only from downstream losses or controller internals.
- Structured action selection should prefer documented channel families or admissible bundle mappings over opaque flat encodings when the task semantics depend on parallel internal-operation and motor channels.
- Policies must not depend on controller internals or model classes.
- Controllers invoke policies and pass resulting actions to environments.
- Environments validate and apply actions but do not silently define policy behavior.

Examples:

- Random walk over a valid-action mask.
- Deterministic stay / no-op policy.
- Region-biased or novelty-biased scripted walk policies.
- Learned selection over admissible internal-operation, motor, modifier, or submit-channel bundles.

#### 4.6.4 Training Primitives

Generic algorithmic building blocks — no model-specific code, no model imports,
and no executable training surfaces. Lightning trainers and other model-aware
training surfaces live in `lightning/`.

| Component         | Path(s)            | Paradigm   | Responsibility                                                                                                |
| ----------------- | ------------------ | ---------- | ------------------------------------------------------------------------------------------------------------- |
| **Step-Loop**     | `step_loop.py`     | Generic    | Generic step iteration: `StepLoop`, `StepModule` protocol, `StepContext`.                                     |
| **Partial-Reset** | `partial_reset.py` | Generic    | Stateful batch assembly: replace completed rows with fresh examples from a buffer.                            |
| **Collector**     | `collector.py`     | Generic    | Per-step state collection for partial-reset pipelines.                                                        |
| **Buffers**       | `buffers.py`       | Generic    | Bounded FIFO storage for batch examples.                                                                      |
| **Optimizers**    | `optim.py`         | Generic    | Typed optimizer configs and wrappers (currently `AdamATan2`).                                                 |
| **Schedulers**    | `schedules.py`     | Generic    | LR schedules: `CosineAnnealingLRWithWarmup`, `SequentialLR`, `SchedulerConfig`.                               |
| **Supervised**    | `supervised.py`    | Supervised | Curriculum scheduling, label-smoothing helpers, supervised step patterns.                                     |
| **RL**            | `rl.py`            | RL         | `compute_gae()`, `policy_gradient_loss()`, advantage estimation, rollout buffer utils, discount calculations. |
| **ELBO**          | `elbo.py`          | VAE / ELBO | KL divergence utilities, ELBO loss aggregation, reconstruction + KL balancing, annealing schedules.           |

**Named by function.** Root-level files are named by algorithmic function
(`buffers.py`). Regime files are named by paradigm
(`supervised.py`, `rl.py`, `elbo.py`).

### 4.7 Data

Data covers on-disk processed format contracts, index parsing, dataset loading,
channel transforms, and Lightning DataModules. Source-specific generation and
canonicalization currently live in `scripts/data-gen/`, not in a first-party
`ehc_sn.data.mazes` package.

#### 4.7.1 Data Modules (`data/`)

ML data infrastructure for processed mazes.

| Module           | Responsibility                                                                                      |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| `schema.py`      | Channel name constants, dtype contracts, and validation for the canonical on-disk format.           |
| `index.py`       | JSONL index parsing, dataset splitting, channel-availability queries.                               |
| `datasets.py`    | Map-style `torch.utils.data.Dataset` returning the canonical per-sample channel dict.               |
| `datamodules.py` | Generic Lightning `DataModule`. Model-specific adaptation is external and belongs in `lightning/` or benchmark bindings under `benchmarks/_bindings/`. |
| `vocabulary.py`  | Canonical maze semantic enum (SEM IDs: PAD, WALL, EMPTY, START, GOAL) and debug character mappings. |
| `transforms.py`  | Model-agnostic channel transforms such as augmentation and semantic-grid derivation.                |

#### 4.7.2 Canonical On-Disk Format

Processed maze data lives in `data/processed/` as a dataset root with a
single JSONL index and per-split stacked channel arrays. Each split stores one
`.npy` file per declared channel with shape `(N, H, W)`, where all samples in
the split share the same stored spatial shape. Optional channels are optional
at split scope, not per-sample scope.

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

**Split metadata** (`data/processed/<split>/dataset.json`): one JSON object per split.

| Field       | Type        | Description                                       |
| ----------- | ----------- | ------------------------------------------------- |
| `source`    | `str`       | Generator or source dataset name                  |
| `split`     | `str`       | Split name                                        |
| `n_samples` | `int`       | Number of samples stored in the split             |
| `shape`     | `list[int]` | Stored normalized maze shape as `[height, width]` |
| `channels`  | `list[str]` | Channel names present for every sample in split   |

**JSONL index** (`data/processed/index.jsonl`): one JSON object per line.

| Field            | Type              | Description                                  |
| ---------------- | ----------------- | -------------------------------------------- |
| `id`             | `str`             | Unique maze identifier                       |
| `source`         | `str`             | Generator that produced the raw maze         |
| `split`          | `str`             | Dataset split: `train`, `val`, or `test`     |
| `shape`          | `tuple[int, int]` | Stored normalized grid shape                 |
| `channels`       | `list[str]`       | Channel names present for the sample's split |
| `n_observations` | `int`             | Observation vocabulary size (0 if absent)    |
| `n_goals`        | `int`             | Number of goal cells (0 if absent)           |
| `difficulty`     | `str`             | Source-defined difficulty label (optional)   |

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
└── processed/                    # Canonical split-uniform channel storage
  ├── index.jsonl
  ├── train/
  │   ├── dataset.json
  │   ├── topology.npy
  │   ├── observations.npy
  │   └── ...
  ├── val/
  └── test/
```

#### 4.6.3 Pipeline

```text
Generators (maze-nd, dungeongen, HF)      scripts/data-gen/
    │
    ▼
  data/raw/
    │
    ▼
  data/interim/  ◄────── source-specific normalization / augmentation scripts
    │
    ▼
    data/processed/  ◄──── root index + split-uniform stacked arrays
    │
    ├──► ehc_sn.data.datasets / datamodules   ──► model adapters
    │
    └──► ehc_sn.envs.*                        ──► controller-owned runtime stepping
```

**Raw sources:**

| Source          | Output                       | Dependency        |
| --------------- | ---------------------------- | ----------------- |
| `maze-nd`       | Grid graph with connectivity | `maze-nd`         |
| `dungeongen`    | Room-based layouts           | `dungeongen`      |
| HuggingFace Hub | Pre-built maze datasets      | `huggingface_hub` |

All three are declared runtime dependencies in `pyproject.toml`. Raw output
is stored in `data/raw/<source>/` and is **not** committed to version control.

**Augmentation (raw → interim):**
Source-specific augmentation and canonicalization scripts currently live in
`scripts/data-gen/`:

- **dungeongen** → add start/goal positions, optionally compute shortest paths.
- **maze-nd** → solve maze, generate observation IDs, assign landmarks.
- **HuggingFace** → relabel goals, extract topology from images, normalize
  format.

Scripts are the current source of truth for generator-specific preprocessing.
Output is stored in `data/interim/<source>/`.

**Canonicalization (interim → processed):**
A final source-agnostic step reads whatever channels are available, normalizes
each split to a uniform stored shape, stacks channels into `(N, H, W)` arrays,
and writes split metadata plus corresponding JSONL index entries.
Output is stored in `data/processed/` with one root index and one storage
partition per split.

#### 4.6.4 Data Loading Responsibilities

`data/` owns persisted processed-data contracts and static loading. Runtime
interaction is handled by the top-level `envs/` component.

#### 4.6.5 Canonical Dataset Output

`MazeDataset.__getitem__` returns a model-agnostic per-sample channel mapping:

| Key              | Type     | Shape    | Description                                         |
| ---------------- | -------- | -------- | --------------------------------------------------- |
| `<channel_name>` | `Tensor` | `(H, W)` | Raw tensor for one canonical channel in the sample. |

The dataset layer preserves the stored processed channels for one sample and
does not synthesize a nested `grid/channels/metadata` structure. Model-specific
input construction such as semantic grids, tokenization, or environment reset
batches is handled by downstream adapters, transforms, or runtime components.

#### 4.6.6 TorchRL Environments (`envs/`)

Runtime environments live in the top-level `envs/` package and are stepped by
controllers, not by dataloaders.

- `dungeon_walk.py`: `DungeonWalk`, a TorchRL environment for controller- and policy-driven dungeon walks from processed maze tensors.
- `mazehard.py`: `MazeHardEnv`, a TorchRL environment for batched token-prediction deliberation on maze-hard style data.

#### 4.6.7 Model–Data Consumption Paths

- **TEM**: processed split arrays → `MazeDataset` / `DataModule` → TEM backbone + controller + explicit policy layer → `DungeonWalk` or benchmark runtime.
- **HRM**: processed split arrays → `MazeDataset` → `MazeHardEnv` or model adapters, depending on training regime.
- **EHC**: processed split arrays → planned dataset/env adapters → RL episodes.

### 4.8 Evaluation

Metrics, trace/rollout collection, and publication-ready visualization.

| Component    | Path        | Responsibility                                                                                                                                                                                                        |
| ------------ | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Lightning** | `lightning/` | Model-aware Lightning training surfaces: `LightningModule` wrappers, optimizer assembly, training hooks, and training-time checkpoint-restore glue. Consumes `models/` and lower reusable layers, but does not own canonical benchmark definitions or benchmark bindings. |
| **Benchmarks** | `benchmarks/` | Canonical benchmark component. Benchmark semantics live in `b0/`-`b3/`; `_capabilities/` defines narrow evaluator contracts; `_bindings/` adapts pure models to those contracts; `_infra/` owns mechanical benchmark support such as seeding, artifact writing, timing, and result types. Benchmark-semantic packages remain model-agnostic; `_bindings/` is internal and model-aware. |
| **Metrics**  | `metrics/`  | TorchMetrics-based evaluation (accuracy, loss ratios, halting stats). Adapter pattern for model-output → metric update.                                                                                               |
| **Rollouts** | `rollouts/` | Trace collection (`TraceCollector`, `TraceSpec`) and tree-structured rollout data (`TraceTree`). Feeds both training diagnostics and figures.                                                                         |
| **Figures**  | `figures/`  | Publication-ready plotting. Public API centers on `FigureContext`, `FigureSpec`, `REGISTRY`, built-in registration, figure modules, sinks, and reusable plotting/layout helpers. Uses SciencePlots + pub-ready-plots. |

Repository-root `scripts/benchmarks/` contains canonical benchmark entrypoints and thin CLIs that resolve config and checkpoint paths, construct benchmark bindings, and delegate into `ehc_sn.benchmarks`. Repository-root `experiments/` is reserved for exploratory, paper-specific, or non-canonical research runners and must not become a second home for shared benchmark wrapper logic.

#### 4.8.1 Canonical Research Benchmark Suite

The canonical research claim for EHC-SN is navigation-centered: EHC is
evaluated as an architecture for partially observable navigation that combines
strong within-episode reasoning with across-episode one-shot adaptation while
preserving benchmark-relevant component capabilities inherited from HRM-style
deliberation and TEM-style episodic memory. Broader general-reasoning claims
are interpretive extrapolations from these benchmarks unless an explicit
non-navigation benchmark suite is added.

The canonical benchmark family is divided into two bridge benchmarks and three
primary navigation benchmarks. Bridge benchmarks validate that EHC preserves
important parent-family capabilities in isolation. Primary benchmarks evaluate
complete agents under shared navigation contracts.

| Benchmark | Purpose | Canonical split / protocol | Current implementation status |
| --------- | ------- | -------------------------- | ----------------------------- |
| **B0 HRM Deliberation Bridge** | Optional bridge benchmark for HRM-style deliberative batch prediction over full-maze token targets. | Use the existing `maze-30x30-hard-1k` processed split: 1000 train / 1000 val / 1000 test 30x30 mazes. Report full test plus a preregistered hard subset derived from the `difficulty` field in the processed index. | Partially scaffolded today by `envs/mazehard.py`, the HRM experiment entrypoints, and `scripts/benchmarks/b0_bridge.py`. Canonical benchmark semantics belong in `benchmarks/`; model-aware HRM bridge bindings belong in `benchmarks/_bindings/`. |
| **M0 Episodic Memory Bridge** | Optional bridge benchmark for TEM-style structural memory, sensory binding, episodic write/read, and one-shot memory behavior under fixed trajectory contracts. | Reuse the B1 processed dungeon split and OOD corpora together with the B2/B3 six-goal / three-start contract. Evaluate benchmark-owned shortest-path exposure traces from start 1 to held-out goals 5-6 and fixed probe traces from starts 2-3 with learned weights frozen. Report current-location localization, held-out-goal recall, write/read consistency from exposure to probe, and interference under sequential held-out-goal exposures. | Not yet implemented. Requires benchmark-owned trajectory contracts and evaluator semantics that measure memory/state capability without treating a memory backbone as a full navigation agent. |
| **B1 Dungeon Navigation Reasoning** | Main within-episode reasoning benchmark for navigation. | Train on the existing processed dungeon split: 800 train / 100 val / 100 test layouts generated by `scripts/data-gen/build-dungeons.py`. Evaluate both in-distribution and on OOD generated test corpora: 100 `medium/classic`, 100 `large/classic`, 100 `small/temple`, and 100 `small/cavern` layouts. | Requires a goal-reaching reward/binding layer on top of the dungeon processed-data contract plus an explicit rollout-agent contract for complete benchmark-time agents; `DungeonWalk` alone is a zero-reward walk surface. |
| **B2 One-Shot Goal Relocation** | Main across-episode one-shot adaptation benchmark. | Reuse the B1 layouts. For each layout precompute a six-goal / three-start contract over the largest connected component: canonical start 1 plus probe starts 2-3. Train with goals 1-4. Reserve goals 5-6 for held-out one-shot evaluation. Evaluation uses one rewarded exposure episode from start 1 followed immediately by probe episodes from starts 2-3 with all learned weights frozen. | Requires the same goal-reaching reward/binding layer as B1 plus explicit frozen-weight evaluation support owned by the B2 benchmark semantics. |
| **B3 Interference and Control** | Main mechanism benchmark for complementary-memory claims. | Reuse the B1 layouts with the same precomputed 6-goal / 3-start contract. Train on goals 1-4 under equal-budget blocked and interleaved schedules. Test on rapid goal-switch sequences over held-out starts 2-3. | Requires the same goal-reaching reward/binding layer as B1 plus retrieval/control diagnostics on top of the rollout traces. |

Status note: `MazeHardEnv` already implements the fixed-horizon deliberation
surface for HRM-style token-prediction tasks. `DungeonWalk` is currently a
policy-driven walk environment with zero reward and therefore does not by
itself satisfy the B1-B3 goal-reaching benchmark contract. Canonical EHC
benchmark work must introduce either a reward-owning dungeon environment or a
clearly documented benchmark binding that adds goal-reaching reward,
termination, and the one-shot exposure/probe protocol without changing the
processed dataset contract. For B1-B3, the evaluated rollout surface must be a
complete agent: a recurrent backbone may update latent or memory state, but an
explicit policy or planner layer must own action selection over legal actions.
`M0` is intentionally not a navigation benchmark; it measures memory/state
capability under fixed benchmark-owned trajectory semantics and must not be
reported as a substitute for B1-B3.

M0 protocol note: M0 owns the action sequence and evaluates only memory/state
behavior. It reuses the B1 layout corpora and the B2/B3 six-goal / three-start
contract so bridge results stay aligned with the primary navigation suite.
Canonical M0 evaluation uses benchmark-owned shortest-path exposure traces from
start 1 to held-out goals 5-6, followed by fixed probe traces from starts 2-3.
No benchmark-time action selection is permitted. During M0 evaluation, only
declared fast-memory state and other recurrent state may change.

#### 4.8.2 Figures Internal Layers

The `figures/` component is internally split into four layers:

- `figures/figures/`: figure authoring framework (`BaseFigureTemplate`, panel decorators, grouped colorbars).
- `figures/modules/`: public figure definitions. Each public module owns one primary figure class and may expose a thin `plot(...)` wrapper.
- `figures/plots/`: Axes-first plotting primitives reused across figure modules.
- `figures/utils/`, `registry.py`, `register.py`, `sinks.py`: layout/data helpers, discovery, and output persistence.

Dependency direction inside the component is one-way: `modules/` may depend on the authoring framework, `plots/`, `utils/`, and registry contracts; `plots/` and `utils/` must not depend on `modules/`.

### 4.9 Utils

Path: `utils/` — generic, reusable helpers with no brain-region, model, or
training semantics. Must **not** import from any `ehc_sn` subpackage.
External dependencies are allowed when declared in `pyproject.toml`; keep
`utils/` broadly reusable and dependency-light.
A function belongs here only if it could be moved to an unrelated ML project
unchanged. Anything requiring an `ehc_sn` type is domain logic and belongs
in its owning component.

**Allowed concerns:**

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

**Exclusions:**

The following do **not** belong in `utils/`:

- Functions referencing brain regions (HPC, MEC, LEC, PFC, STR).
- Functions operating on project types (`LocationBelief`, `MultiScaleCode`, etc.).
- Loss or metric computation (→ `loss/`, `metrics/`).
- Activation functions (→ `activations/`).
- Data loading or dataset logic (→ `data/`).

### 4.9 Infrastructure

Cross-cutting support that wraps external frameworks.

| Component       | Path           | Responsibility                                                                                                                                                                                  |
| --------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Callbacks**   | `callbacks/`   | Lightning `Callback` wrappers (checkpoint, figures). Used by experiments only.                                                                                                                  |
| **Logging**     | `logging/`     | Lightning `TensorBoardLogger` wrapper. Used by experiments only.                                                                                                                                |
| **Activations** | `activations/` | Custom activation functions (e.g., `stablemax`). Consumed by `loss/cross_entropy.py`. Small (single file); lives here rather than in `loss/` because activations are a general-purpose concern. |

---

## 5 Model Composition

Each model co-locates a **config**, a **pure `nn.Module`**, and a **state
dataclass** in `models/`. Executable training surfaces such as
`LightningModule` trainers live in `lightning/`. Executable benchmark surfaces
live in `benchmarks/`, with model-aware adaptation and benchmark-time
checkpoint hydration in `benchmarks/_bindings/`. Generic training
infrastructure lives in `training/`.

### 5.1 State Management

Each model defines an explicit `dataclass` for its recurrent state.

- State is passed into and returned from `forward()`.
- No hidden state is stored in module attributes between calls.
- State dataclasses provide `detach()` (for TBPTT truncation, etc.).
- Composed models nest sub-states (e.g., `TEMState` contains `LECState`,
  `MECState`, `HPCState`; `EHCState` would contain `TEMState` +
  `HRMState` or their components).

### 5.2 Step Interface

Each model exposes a step-level `forward()` that processes one timestep:

```python
def forward(self, ..., state: ModelState) -> tuple[ModelState, Logits, Features]:
    ...
```

The training runtime's `training_step` may iterate over `StepLoop`, yielding
`(t, step_output)` pairs per timestep.

### 5.3 Module Reuse Protocol

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

---

## 6 Configuration

### 6.1 Config Types

| Type                    | Base class                                            | Scope                                         | Lives in                     | Example                           |
| ----------------------- | ----------------------------------------------------- | --------------------------------------------- | ---------------------------- | --------------------------------- |
| **Component config**    | `pydantic.BaseModel(extra="forbid")`                  | Single-component settings                     | Same file as the `nn.Module` | `AttractorSettings`               |
| **Model config**        | `pydantic.BaseModel(extra="forbid")`                  | Architecture: dimensions, layers, activations | `models/*.py`                | `ModelSettings_V1`, `PFCSettings` |
| **Lightning training config** | `pydantic.BaseModel(extra="forbid")`              | Optimizer, LR schedule, loss weights, buffers, trainer-local training settings | `lightning/**/*.py` | `ModelConfig_HRM_V1`               |
| **Data config**         | `pydantic.BaseModel(extra="forbid")`                  | Dataset paths, batch size, workers            | `data/*.py`                  | `DatamoduleConfig`                |
| **Experiment settings** | `pydantic_settings.BaseSettings(cli_parse_args=True)` | Composes all above + Trainer knobs            | `experiments/*.py`           | `RunArguments`                    |

Architectural dimensions use `frozen=True`. Training runtime configs are
separate from model configs; the runtime passes only the architecture config to
the pure `nn.Module` constructor.

### 6.2 Static Defaults

TOML files under `config/` provide default values. The experiment
entry point loads defaults first, then CLI arguments override:

```python
defaults = tomllib.load(Path("config/defaults_ehc.toml").open("rb"))
settings = RunArguments(**defaults)  # CLI overrides via pydantic_settings
```

### 6.3 Leaf Composition Pattern

Experiment settings are structured as a flat tree of **leaf configs**.
Composed configs for downstream consumers (model, datamodule, regime) are
assembled via `@property` methods using `model_validate(self, from_attributes=True)`.

```text
RunArguments(BaseSettings)                    ← CLI + TOML
  ├── architecture: PFCSettings                 ← model architecture (frozen dims)
  ├── loss: ACTLossConfig                     ← loss weights / targets
  ├── optimizer: AdamATan2Config              ← optimizer hyperparams
  ├── scheduler: SchedulerConfig              ← LR schedule
  ├── dataset: DatasetSettings                ← dataset paths, seed
  ├── global_batch_size: int                  ← shared by model + data
  ├── logger: LoggerSettings                  ← TensorBoard config
  ├── checkpoint: CheckpointSettings          ← checkpoint config
  │
  └── composed via @property:
      ├── .model → ModelConfig_HRM_V1         (architecture + loss + optimizer + ...)
      └── .datamodule → DatamoduleConfig (dataset + batch_size + workers + ...)
```

This pattern ensures:

- **CLI ergonomics**: Any leaf field can be overridden from the command line
  (e.g., `--optimizer.lr=1e-4`, `--epochs=100`).
- **No duplication**: Shared fields (e.g., `global_batch_size`) are defined
  once and composed into multiple downstream configs.
- **Reproducibility**: The full `RunArguments` can be serialized to
  reproduce any experiment.

---

## 7 Forbidden Architectural Patterns

1. **Web-framework layering**: No controllers, views, routers, serializers,
   schemas (in the web sense), or similar patterns from Django/FastAPI/Flask.
2. **Flat utils dump**: Domain logic must live in its owning component, not
   in `utils/`.
3. **`core`/`common`/`shared` mega-packages**: Do not create catch-all packages
   that aggregate unrelated code.
4. **Wildcard star imports** in non-`__init__` files: `from X import *` is
   forbidden outside `__init__.py` re-exports.
5. **Legacy namespace imports**: No `from torch_tem` or `from hrm_sn` in any
   file under `src/ehc_sn/`.

---
