# EHC-SN Requirements Specification

> Canonical source of truth for repo-level gating rules, dependency inventory,
> runtime constraints, and change policy.

## 1 Spec Gate (Mandatory for Agents)

Before generating any design or code artifact, automated agents **MUST**:

1. Verify `spec/spec-manifest.toml` exists.
2. Verify every file listed under `required.files` in that manifest exists and
   is non-placeholder.
3. If any are missing, **STOP** and output exactly:
   `Blocking: missing required specs: <comma-separated list of missing paths>`

Spec maintenance and conflict-resolution workflow are owned by
`spec/spec-process-spec-maintenance.md`.

---

## 2 Namespace and Schema Constraints

- All new code under `src/ehc_sn/` **MUST** use `ehc_sn` as the sole import
  namespace. Forbidden: `torch_tem`, `hrm_sn`.
- Do not invent new data schemas, type aliases, or configuration contracts
  without first checking whether an equivalent exists in `ehc_sn/types.py` or
  in the owning component.
- Extend existing types rather than creating parallel hierarchies.
- New type aliases in `ehc_sn/types.py` must include docstrings with shape
  conventions.
- New Pydantic configs must use `extra="forbid"` unless there is an explicit,
  documented reason for leniency.

### 2.1 Policy Contracts

- Reusable policy interfaces and policy configs belong in `ehc_sn/policies/`.
- Policies own action selection only. They must not own loss construction,
  optimizer or update procedure, or learner-style RL algorithm logic.
- Policy APIs must consume explicit typed inputs rather than arbitrary
  controller internals.
- Hard action constraints must be represented explicitly in the policy input.
- Policy logits used for action selection must remain conceptually distinct
  from reward or value outputs unless a canonical spec documents a combined
  action-and-value surface.
- Policy objects own policy-local randomness and sampling semantics.

### 2.2 Figure Contracts

- The canonical public API of `ehc_sn.figures` is limited to registry/context
  contracts, built-in registration, sinks, and `ehc_sn`-native figure modules.
- A public figure module must define one primary figure class derived from
  `BaseFigureTemplate` and may expose a thin `plot(...)` wrapper.
- Modules under `src/ehc_sn/figures/` that still import legacy namespaces are
  migration inventory, not public API.

---

## 3 Architectural Enforcement Rules

- `spec/spec-architecture.md` owns canonical boundary vocabulary, component
  taxonomy, and dependency-layer rules.
- Changes must satisfy the owning architecture rules rather than reinterpreting
  them locally in this file.
- Model configs define architecture only. Task and adapter configs define task
  semantics and binding semantics.
- Lightning training surfaces instantiate `task -> model -> adapter` and must
  execute through adapter interfaces.
- Reusable objective scoring belongs in `objectives/`; Lightning remains the
  executable training orchestration surface.
- Architecture-affecting changes must update `spec/spec-architecture.md` in the
  same change.
- Detailed model and adapter interface patterns live in
  `spec/spec-model-interfaces.md`.
- Detailed controller-to-learner and family-specific runtime contracts live in
  `spec/spec-controller-runtime-contracts.md`.

---

## 4 Dependency Constraints

### 4.1 Runtime Dependencies

| Dependency           | Role                    | Notes                                     |
| -------------------- | ----------------------- | ----------------------------------------- |
| `torch`              | Core tensor computation | Pin-free user-managed CUDA compatibility. |
| `torchrl`            | TorchRL environments    | Runtime environment stepping.             |
| `lightning`          | Training orchestration  | `LightningModule`, `Trainer`, callbacks.  |
| `gymnasium`          | Environment interface   | Maze and navigation environments.         |
| `pydantic`           | Configuration schema    | `BaseModel(extra="forbid")` pattern.      |
| `pydantic_settings`  | CLI settings            | `BaseSettings(cli_parse_args=True)`.      |
| `scipy`              | Scientific computing    | Combinatorics and special functions.      |
| `networkx`           | Graph algorithms        | Connectivity and graph utilities.         |
| `matplotlib`         | Visualization           | Base figure rendering.                    |
| `SciencePlots`       | Visualization           | Publication plotting styles.              |
| `pub-ready-plots`    | Visualization           | Publication layout helpers.               |
| `tensorboard`        | Logging                 | Scalars, metrics, and images.             |
| `rich`               | Terminal output         | Progress and formatted console output.    |
| `huggingface_hub`    | Dataset download        | Raw/source dataset retrieval.             |
| `maze-nd`            | Maze generation         | Generator dependency used by scripts.     |
| `dungeongen`         | Dungeon generation      | Generator dependency used by scripts.     |
| `typer`              | Script CLIs             | Data-generation command-line interface.   |
| `adam-atan2-pytorch` | Optimizer               | AdamAtan2 for HRM training.               |
| `setuptools`         | Build backend           | Package build and version management.     |

`pyproject.toml` is the executable source of truth for dependency declarations.
This table is the human-readable inventory and must stay synchronized with it.

### 4.2 Dev and Script Dependencies

- `pytest` for testing.
- `black` for formatting.
- `flake8` for linting.
- `mypy` for type checking.

### 4.3 New Dependency Policy

- New runtime dependencies require justification.
- Dependency declaration changes must update `pyproject.toml` and the runtime
  dependency inventory in this file in the same change.
- Prefer well-maintained packages with compatible licenses and supported Python
  versions.

---

## 5 Runtime and Execution Constraints

### 5.1 Python and Package Baseline

- Python version: `>= 3.12`.
- Build backend: `setuptools`.
- Package layout: `src/` layout.
- Active first-party package: `ehc_sn` only.
- Single-source version: `src/ehc_sn/VERSION`.

### 5.2 Executable Surface Matrix

- `scripts/training/` contains thin training entry points only. Shared training
  logic belongs in `src/ehc_sn/`.
- `scripts/haicore/` contains cluster-launch wrappers only. They may provide
  scheduler glue but must not duplicate training or benchmark logic.
- `scripts/benchmarks/` contains thin wrappers only. Shared benchmark logic
  belongs in `ehc_sn.benchmarks`.
- `lightning/` is the executable training orchestration surface. It must
  execute through adapters rather than bypassing them with task-shaped model
  payloads directly.

---

## 6 Data and Path Constraints

- Canonical data pipeline: `data/raw/` → `data/interim/` → `data/processed/`.
- DataModules consume only `data/processed/` data.
- Raw data is not committed to version control.
- Processing scripts live in `scripts/data-gen/`.
- `data/interim/` is optional scratch space; `data/external/` is for
  third-party datasets.
- No hard-coded absolute paths. Use config or CLI inputs.
- All roots under `data/processed/` are **immutable versioned leaves**. The
  two canonical dataset classes are:
  - **Shared substrate**: `data/processed/<shared-family>/v<integer>/` —
    owned by the upstream source, task-neutral channels only.
  - **Task corpus**: `data/processed/<task-name>/<corpus-name>/v<integer>/` —
    owned by the task package, includes task-protocol channels.
- A shared-family name must not collide with a task namespace.
- The current shared-family and task-namespace registry is owned by
  `spec/spec-data-contracts.md`.
- `data/` owns provenance, normalization, shared schema, shared manifests,
  shared validation, and shared substrate materialization. `tasks/` own task
  schema, task corpus materialization, replay rows, episode protocol,
  reward/supervision semantics, and runtime reconstruction. `adapters/` own
  nothing persistent.
- Build reports and benchmark manifests are not canonical dataset contents and
  must not live under `data/processed/`. Use `reports/benchmarks/` or
  `outputs/`.
- Detailed versioned-root format, path grammar, dataset class rules, root
  manifest contract, and staged CLI semantics live in
  `spec/spec-data-contracts.md`.

---

## 7 Benchmark and Reporting Constraints

- Canonical benchmark definitions, corpus contracts, frozen-weight rules, claim
  semantics, and benchmark-specific reporting rules live in
  `spec/spec-benchmark-suite.md` and are normative.
- Canonical benchmark entry points belong under `scripts/benchmarks/` as thin
  wrappers around `ehc_sn.benchmarks`.
- Benchmark-like code outside the canonical script roots is non-canonical and
  must not duplicate shared benchmark orchestration or artifact writing.
- Repository-level reporting and documentation standards remain governed by
  `spec/spec-standards.md`.

---

## 8 Cross-Component Change Policy

Cross-component changes require a tracked plan under the manifest-owned plans
root (`spec/spec-manifest.toml [canonical_paths.plans_root]`) before
implementation begins. This includes:

- adding a new top-level package under `ehc_sn/`;
- moving code between components;
- changing a public API used across components;
- adding or removing a runtime dependency.

For new top-level packages, the same change must update
`spec/spec-architecture.md` so the component taxonomy remains complete and the
dependency boundaries remain explicit.

Single-component changes do not require a tracked plan unless they alter the
component's public contract.
