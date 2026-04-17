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

The only exception is the **Bootstrap Mode** override defined in the
`bootstrap-specs.prompt.md` prompt, which operates before specs exist.

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

- `models/` must remain task-agnostic.
- `tasks/` own observation/action/workspace/episode/reward semantics.
- `adapters/` are the only canonical model-task seam.
- Model configs define architecture only. Task and adapter configs define task
  semantics and binding semantics.
- Lightning training surfaces instantiate `task -> model -> adapter` and must
  execute through adapter interfaces.
- Reusable objective scoring belongs in `heads/`; Lightning remains the
  executable training orchestration surface.
- Adding a new task or puzzle must require changes only in `tasks/` and
  optional adapters.
- Adding a new architecture must require changes only in `models/`, optional
  adapters, and model-aware training surfaces.
- Detailed boundary vocabulary lives in `spec/spec-architecture.md`.
- Detailed model and adapter interface patterns live in
  `spec/spec-model-interfaces.md`.

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

### 4.2 Dev and Script Dependencies

- `pytest` for testing.
- `black` for formatting.
- `flake` for linting.
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

### 5.2 Ownership Matrix

| Component                       | Constraint                                                                                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `training/`                     | Must remain model-agnostic and task-agnostic. No imports from `models/`, `modules/`, `tasks/`, `adapters/`, or `lightning/`.                                                                |
| `tasks/`                        | Must remain model-agnostic. No imports from `models/`, `modules/`, `adapters/`, or `lightning/`.                                                                                            |
| `adapters/`                     | Canonical model-task seam. May import `models/`, `tasks/`, and lower reusable layers, but must not own benchmark semantics, CLI orchestration, or generic training primitives.              |
| `rollouts/`, `traces/`          | Must remain model-agnostic. No imports from `models/` or `lightning/`.                                                                                                                      |
| `controllers/`, `heads/`        | Controllers own rollout-state transitions and heads own reusable objective scoring. Both must remain model-agnostic and task-agnostic. No imports from `models/`, `tasks/`, or `adapters/`. |
| `policies/`                     | Own action selection only. Must remain model-agnostic and controller-agnostic. No imports from `models/`, `tasks/`, `adapters/`, `controllers/`, `heads/`, `training/`, or `modules/`.      |
| `benchmarks/` semantic packages | Must remain model-agnostic. No imports from `models/` or `lightning/`.                                                                                                                      |
| `benchmarks/_bindings/`         | Only model-aware benchmark subarea. May import `models/`, `tasks/`, `adapters/`, and lower reusable layers, but must not own benchmark semantics.                                           |
| `lightning/`                    | Owns executable training orchestration. Reusable objective scoring remains in `heads/`. Must execute through adapters rather than task-shaped model payloads.                               |
| `scripts/benchmarks/`           | Thin wrappers only. Shared benchmark logic belongs in `ehc_sn.benchmarks`.                                                                                                                  |
| `experiments/`                  | Exploratory or paper-specific entry points only. Must not duplicate shared benchmark or training infrastructure.                                                                            |

---

## 6 Data and Path Constraints

- Canonical data pipeline: `data/raw/` → `data/interim/` → `data/processed/`.
- DataModules consume only `processed` data.
- Raw data is not committed to version control.
- Processing scripts live in `scripts/data-gen/`.
- `data/interim/` is optional scratch space; `data/external/` is for
  third-party datasets.
- No hard-coded absolute paths. Use config or CLI inputs.
- Detailed processed-data format and dataset output contracts live in
  `spec/spec-data-contracts.md`.

---

## 7 Benchmark and Reporting Constraints

- Canonical benchmark definitions, corpus contracts, frozen-weight rules, claim
  semantics, and benchmark-specific reporting rules live in
  `spec/spec-benchmark-suite.md` and are normative.
- Canonical benchmark entry points belong under `scripts/benchmarks/` as thin
  wrappers around `ehc_sn.benchmarks`.
- Benchmark-like code under `experiments/` is non-canonical and must not
  duplicate shared benchmark orchestration or artifact writing.
- Repository-level reporting and documentation standards remain governed by
  `spec/spec-standards.md`.

---

## 8 Cross-Component Change Policy

Cross-component changes require a tracked plan in `.copilot-tracking/plans/`
before implementation begins. This includes:

- adding a new top-level package under `ehc_sn/`;
- moving code between components;
- changing a public API used across components;
- adding or removing a runtime dependency.

For new top-level packages, the same change must update
`spec/spec-architecture.md` so the component taxonomy remains complete and the
dependency boundaries remain explicit.

Single-component changes do not require a tracked plan unless they alter the
component's public contract.
