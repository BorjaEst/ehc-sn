# EHC-SN Requirements Specification

> Canonical source of truth for repo-level constraints, gating rules, and
> dependency contracts. See `spec/spec-manifest.toml` for precedence rules.

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

## 2 Namespace Constraint

All new code under `src/ehc_sn/` **MUST** use `ehc_sn` as the sole import
namespace.

- **Forbidden**: `from torch_tem ...`, `from hrm_sn ...`, `import torch_tem`,
  `import hrm_sn`.
- Legacy remnants under `temp/` are archived and not on the Python path.
- This constraint is enforced at review time; no automated linter exists yet.

---

## 3 Schema and Contract Constraints

- **Do not invent** new data schemas, type aliases, or configuration contracts
  without first checking whether an equivalent exists in:
  - `ehc_sn/types.py` (type aliases and dataclasses)
  - Existing Pydantic configs in the owning component
- **Extend** existing types rather than creating parallel hierarchies.
- New type aliases in `types.py` must include a docstring with shape conventions
  (following the existing pattern in that file).
- New Pydantic configs must use `extra="forbid"` unless there is an explicit,
  documented reason for leniency.

---

## 4 Dependency Constraints

### 4.1 Runtime Dependencies (declared in `pyproject.toml`)

| Dependency           | Role                    | Notes                                   |
| -------------------- | ----------------------- | --------------------------------------- |
| `torch`              | Core tensor computation | Pin-free (user-managed CUDA compat)     |
| `lightning`          | Training orchestration  | LightningModule, Trainer, Callbacks     |
| `gymnasium`          | Environment interface   | Maze environments                       |
| `pydantic`           | Configuration schema    | `BaseModel(extra="forbid")` pattern     |
| `pydantic_settings`  | CLI settings            | `BaseSettings(cli_parse_args=True)`     |
| `scipy`              | Scientific computing    | Combinatorics, special functions        |
| `matplotlib`         | Visualization (base)    | Figure rendering                        |
| `SciencePlots`       | Visualization (styles)  | Publication-ready plot styles           |
| `pub-ready-plots`    | Visualization (layout)  | Publication-ready plot utilities        |
| `tensorboard`        | Logging                 | Metric/scalar/image logging             |
| `rich`               | Terminal output         | Progress bars, formatted console output |
| `adam-atan2-pytorch` | Optimizer               | AdamAtan2 for HRM training              |
| `setuptools`         | Build backend           | Package build and version management    |

### 4.2 Audit Candidates

| Dependency | Issue                                                        | Action                                                                                         |
| ---------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `networkx` | Declared in `pyproject.toml` but **zero imports** in `src/`. | Audit before next release. Do not add new usage until confirmed needed. Candidate for removal. |

### 4.3 Known Dependency Bugs

| Dependency | Issue                                                                                                                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `typer`    | Imported in `scripts/data-gen/build_maze.py` but **not declared** in `pyproject.toml`. Must be added to `[project.optional-dependencies]` (e.g., under a `scripts` extra) or to the main dependencies. |

### 4.4 Dev / Script-Only Dependencies

| Dependency | Role                          | Where Used                          |
| ---------- | ----------------------------- | ----------------------------------- |
| `maze-nd`  | Raw maze structure generation | `scripts/data-gen/` pipeline input. |
| `typer`    | CLI framework                 | `scripts/data-gen/build_maze.py`    |
| `pytest`   | Testing                       | `tests/`                            |
| `black`    | Formatting                    | Dev tooling                         |
| `mypy`     | Type checking                 | Dev tooling                         |

### 4.5 New Dependency Policy

- New runtime dependencies require justification (why existing deps cannot
  serve the need).
- New dependencies must be added to `pyproject.toml` in the appropriate section.
- Prefer pure-Python or well-maintained packages with compatible licenses.

---

## 5 Data Handling Constraints

- **Raw data** is generated externally by maze-nd and placed in `data/raw/`.
  Raw data is **not** committed to version control.
- **Processing scripts** live in `scripts/data-gen/`. They transform raw data
  into labeled, processed datasets in `data/processed/`.
- **Interim data** (`data/interim/`) is optional scratch space for intermediate
  pipeline stages.
- **External data** (`data/external/`) is for third-party reference datasets.
- The `data/` project-root directory structure follows the convention:
  `raw → interim → processed`. DataModules in `ehc_sn.data` consume only
  `processed` data.
- Do not hard-code absolute paths to data directories. Use configuration
  (Pydantic settings or CLI args) to resolve data paths.

---

## 6 Python and Runtime

- **Python**: ≥ 3.12 (as declared in `pyproject.toml`).
- **Build backend**: `setuptools` with `pyproject.toml`-based configuration.
- **Package layout**: `src/` layout (`tool.setuptools.package-dir = {"" = "src"}`).
- **Packages**: `ehc_sn` (main). `mazes` is a legacy auxiliary package
- **Version**: Single-source in `src/ehc_sn/VERSION`.

---

## 7 Training Infrastructure Constraints

- All code in `training/` is **model-agnostic**. It contains algorithmic
  building blocks (step-loop, optimizer configs, LR schedules, paradigm
  primitives) with **no model-specific imports** (see `spec-architecture.md`
  §3.5).
- Model-specific training orchestration (loss aggregation, step assembly,
  partial-reset policies) lives in the `LightningModule` trainer co-located
  in `models/*.py`.
- If a training primitive currently serves only one model, it must be
  generalized or relocated to the model file before the next release.
- Shared protocols (`StepLoop`, `StepModule`), optimizer configs, and LR
  schedulers remain at the `training/` root.

---

## 8 Cross-Component Change Policy

Code changes that cross component boundaries (as defined in
`spec-architecture.md` §3) require a tracked plan in
`.copilot-tracking/plans/` **before** implementation begins. This includes:

- Adding a new top-level package under `ehc_sn/`.
- Moving code between components.
- Changing the public API of a component that other components depend on.
- Adding or removing a runtime dependency.

Single-component changes (bug fixes, internal refactors, new private helpers)
do not require a plan unless they alter the component's public contract.
