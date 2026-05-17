# EHC-SN Architecture Specification

> Canonical source of truth for component boundaries, public vocabulary, and
> top-level taxonomy.

## 1 Project Identity

**EHC-SN** (_Entorhinal-Hippocampal Circuit — Spatial Navigation_) is a
research library for biologically-inspired spatial cognition and navigation models built on
PyTorch.

The canonical architecture is **multi-task** and **multi-model**:

- pure reusable architectures live in `models/`;
- task semantics live in `tasks/`;
- the only canonical model-task seam lives in `adapters/`.

---

## 2 Canonical Import Namespace

The sole canonical import namespace is **`ehc_sn`**.
Legacy namespaces (`torch_tem`, `hrm_sn`) are retired and archived under
`temp/` (not on the Python path).

---

## 3 Public Vocabulary

TODO:

---

## 4 Layered Dependency Model

Internal imports follow a top-down DAG: import from your own layer or below,
never upward.

| Layer | Components                                                                                                                                                               |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **4** | `scripts/training/`, `scripts/haicore/`, `scripts/benchmarks/`                                                                                                           |
| **3** | `models/`, `tasks/`, `adapters/`, `benchmarks/`, `lightning/`                                                                                                            |
| **2** | `modules/`, `controllers/`, `objectives/`, `policies/`, `training/`, `loss/`, `metrics/`, `rollouts/`, `traces/`, `figures/`, `callbacks/`, `logging/`, `data/`, `envs/` |
| **1** | `activations/`, `utils/`, `types.py`                                                                                                                                     |

### 4.1 Boundary Rules

- `utils/` must not import from any `ehc_sn` subpackage.
- `modules/` may depend on peer modules and Layer 1 only. They must not import
  from `models/`, `tasks/`, `adapters/`, `lightning/`, or training/runtime
  execution surfaces.
- `models/` may import `modules/` and lower reusable layers. They must not
  import from `tasks/`, `adapters/`, benchmark-semantic packages, or
  `lightning/`.
- `tasks/` may import lower reusable layers such as `data/`, `envs/`,
  `metrics/`, `rollouts/`, `traces/`, `policies/`, and Layer 1. They must not
  import from `models/`, `modules/`, `adapters/`, or `lightning/`.
- `adapters/` may import from `models/`, `tasks/`, and lower reusable layers.
  They must not own canonical benchmark semantics, generic training primitives,
  or CLI orchestration.
- `controllers/`, `objectives/`, and `policies/` are reusable lower-layer
  runtime primitives. Controllers own rollout-state transitions, objectives own
  rollout scoring, and policies own action selection. They must remain
  model-agnostic and task-agnostic.
- `objectives/` is the sole canonical public surface for objective modules.
- `training/`, `rollouts/`, and `traces/` are reusable lower-layer execution
  primitives. They must remain model-agnostic.
- `data/` and `envs/` are reusable infrastructure. They must not import from
  `models/`, `tasks/`, `adapters/`, or `lightning/`.
- `benchmarks/` semantic packages remain model-agnostic. Only
  `benchmarks/_bindings/` may be model-aware. No code outside `benchmarks/` may
  import from `benchmarks/_*`.
- `lightning/` may import from `models/`, `tasks/`, `adapters/`, and lower
  reusable layers. It trains through adapters and does not own canonical
  benchmark semantics or generic training primitives.
- `scripts/training/`, `scripts/haicore/`, and `scripts/benchmarks/` are thin
  entry points. Shared logic must live below them.

### 4.2 Ownership Invariants

- **Models are task-agnostic.** No reward logic, dataset logic, environment
  stepping, puzzle semantics, or task-visible observation/action schemas belong
  in `models/`.
- **Predictive models are not navigation agents by default.** A model trained
  on spatial trajectories may remain a predictive cognitive-map model when it
  does not itself own goal selection or overt action choice.
- **Tasks own semantics.** Observation formats, action ontology, workspace slot
  geometry, episode structure, reward or supervision rules, and evaluation
  semantics belong in `tasks/`.
- **Navigation grounding lives in tasks.** Spatial transitions, revisit
  structure, action ontology, and episode organization belong to the task even
  when the bound model family is purely predictive.
- **Adapters are the only canonical seam.** Observation normalization,
  tokenization, workspace packing, model input assembly, output decoding, loss
  construction, reward shaping, and rollout binding belong in `adapters/`.
- **Goal-directed navigation claims require action selection above the task.**
  A complete navigation agent exposes action selection through a native action-
  selection head or an explicit attached policy/controller layer.
- **Lightning executes through adapters.** Training surfaces instantiate
  `task -> model -> adapter -> optimizer/scheduler`.
- **Benchmarks own evaluation semantics.** Benchmark-semantic packages define
  benchmark meaning; benchmark bindings adapt models or adapters to benchmark
  capability contracts.
- **Data and environments are reusable infrastructure.** `data/` owns persisted
  static contracts; `envs/` owns reusable runtime kernels; `tasks/` own the task
  meaning attached to those resources.
  <<<<<<< HEAD
- **`tasks/*/environment.py` is the canonical live-env shell.** Each task's
  `environment.py` is the authoritative live sequential-decision shell for
  task-owned RL interaction via `RLTaskRuntime`. Reward, `terminated`,
  `truncated`, and semantic episode horizon belong here, not in `envs/` or in
  online RL controller configs. Tasks whose live-env implementation is deferred
  must declare that status explicitly in their `environment.py` rather than
  leaving an ambiguous empty file. Online RL controllers must not own a
  `max_steps` or equivalent semantic budget field in their public configs.
  =======
  > > > > > > > dev0.2.11
- **Shared substrate versus task corpus.** `data/` owns shared-substrate
  persistence — geometry and spatial channels reused across task corpora.
  Each `tasks/<task>/` owns the schema, validation, and materialization of its
  own task-corpus channels layered over that substrate. Adapters own no
  persistence.

---

## 5 Component Taxonomy

Every top-level package under `src/ehc_sn/` maps to exactly one component below.
If a new top-level package is created, this table must be updated in the same
change.

### 5.1 Brain-Region Modules

| Component | Path           | Responsibility                                              |
| --------- | -------------- | ----------------------------------------------------------- |
| **LEC**   | `modules/lec/` | Sensory encoding and temporal filtering.                    |
| **MEC**   | `modules/mec/` | Path integration, grid-cell dynamics, and OVC coding.       |
| **HPC**   | `modules/hpc/` | Episodic memory write/read and grounded-location inference. |
| **PFC**   | `modules/pfc/` | Working-memory maintenance and recurrent reasoning.         |
| **BG**    | `modules/bg/`  | Planned arbitration over admissible control bundles.        |
| **STR**   | `modules/str/` | Reward prediction and reward-sensitive bias signals.        |

### 5.2 Top-Level Packages

| Path           | Responsibility                                                                         |
| -------------- | -------------------------------------------------------------------------------------- |
| `activations/` | Reusable activation functions.                                                         |
| `adapters/`    | Explicit model-task bindings.                                                          |
| `benchmarks/`  | Canonical benchmark semantics and internal benchmark bindings.                         |
| `callbacks/`   | Training-time framework callbacks.                                                     |
| `controllers/` | Reusable rollout-state transition primitives.                                          |
| `data/`        | Persisted processed-data contracts and static loading.                                 |
| `envs/`        | Reusable runtime environment kernels.                                                  |
| `figures/`     | Visualization and figure authoring.                                                    |
| `lightning/`   | Executable training surfaces (Lightning modules) built on tasks, models, and adapters. |
| `logging/`     | Logging wrappers and logger setup.                                                     |
| `loss/`        | Reusable loss primitives.                                                              |
| `metrics/`     | Reusable evaluation metrics and signal keys.                                           |
| `models/`      | Pure architectures such as TEM, HRM, and EHC families.                                 |
| `modules/`     | Pure reusable neural-network building blocks.                                          |
| `objectives/`  | Canonical public objective API.                                                        |
| `policies/`    | Reusable action-selection logic only.                                                  |
| `rollouts/`    | Temporal execution drivers and rollout records.                                        |
| `tasks/`       | Canonical task semantics and task-local contracts.                                     |
| `traces/`      | Passive trace observation and storage.                                                 |
| `training/`    | Generic training primitives with no model or task semantics.                           |
| `utils/`       | Generic helpers with no `ehc_sn` imports.                                              |
| `types.py`     | Lightweight shared types and aliases.                                                  |

---

## 6 Companion Reference Specs

The following specs are **not** part of the always-read required set in
`spec/spec-manifest.toml`. They hold lower-frequency reference material that
should not bloat the core hot path.

- `spec/spec-data-contracts.md`: processed-data format, pipeline, and dataset
  output contracts.
- `spec/spec-benchmark-suite.md`: detailed benchmark definitions, protocols,
  and benchmark-specific constraints.
- `spec/spec-configuration-patterns.md`: configuration taxonomy, default
  loading, and composition patterns.
- `spec/spec-model-interfaces.md`: detailed state, step, and adapter interface
  patterns.
- `spec/spec-controller-runtime-contracts.md`: controller-to-learner,
  runtime, and family-specific execution contracts.

---

## 7 Forbidden Boundary Violations

- No task-visible observation, action, reward, puzzle, or dataset semantics in
  `modules/` or `models/`.
- No imports from `models/` inside `tasks/`.
- No benchmark-semantic logic in `adapters/` or `lightning/`.
- No code outside `benchmarks/` may import benchmark internals under
  `benchmarks/_*`.
- No trainer or benchmark entry point may bypass adapters to call task-shaped
  model payloads directly.
