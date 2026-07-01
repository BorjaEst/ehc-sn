# Train Models

<!--
  canonical_package: ehp_sn
  implementation_package: ehc_sn  (temporary, during migration)
  status: draft
-->

> CLI surface for the training service — resolve, validate, and execute
> model-training runs through a registered recipe catalogue.

---

## Overview

The `train` command group replaces eight family-specific entry-point scripts
(`scripts/training/tem_v1_arena.py`, `hrm_v1_mazehard.py`, …) with a single
dispatch surface. It owns **selection, resolution, validation, and execution**
of training runs; it does not own model construction, checkpoint I/O, or
artifact publishing — those are delegated to the training application service
(`ehp_sn.training.service`) and downstream packages.

```
recipe/config ──► resolve_training_run()
                       │
                       ▼
                  ResolvedTrainingRun
                       │
                       ▼
                  validate_training_run()
                       │
                       ▼
                  build_training_experiment()
                       │
                       ▼
                  run_training()
                       │
                       ▼
                  TrainingRunResult
```

---

## Commands

### `train run`

Resolve a configuration, validate it, and execute one training run.

```
ehp-sn train run [RECIPE] [OPTIONS]
```

| Argument | Description                                                                                     |
| -------- | ----------------------------------------------------------------------------------------------- |
| `RECIPE` | Registered training recipe (e.g. `tem-v1-arena`). Optional when `--config` declares the recipe. |

| Option              | Description                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| `-c, --config PATH` | Additional or standalone training configuration TOML.                    |
| `--profile NAME`    | Execution profile layered over the recipe (e.g. `gpu-8gb`).              |
| `--set PATH=VALUE`  | Repeatable typed configuration override.                                 |
| `--resume PATH`     | Resume complete training state from a checkpoint.                        |
| `--init-from PATH`  | Initialize model weights for a new run from a checkpoint.                |
| `--init-group NAME` | Repeatable semantic parameter group to load (e.g. `pfc_core`, `all`).    |
| `--output-dir PATH` | Explicit run output root.                                                |
| `--run-id TEXT`     | Explicit run identifier.                                                 |
| `--tag KEY=VALUE`   | Repeatable run metadata tag.                                             |
| `--note TEXT`       | Human-readable run annotation.                                           |
| `--seed INTEGER`    | Explicit global seed (overrides config).                                 |
| `--device TEXT`     | Runtime device: `auto`, `cpu`, `cuda`, `cuda:0`, `mps`.                  |
| `--precision TEXT`  | Precision policy (e.g. `16-mixed`, `bf16-mixed`, `32-true`).             |
| `--dry-run`         | Resolve and validate without constructing models or training.            |
| `--print-config`    | Print the effective resolved configuration to stdout.                    |
| `--explain`         | Print provenance for every resolved configuration field.                 |
| `--yes`             | Accept explicitly defined destructive actions (e.g. overwriting output). |
| `--debug`           | Enable repository-defined debug behaviour (full tracebacks).             |

#### Examples

```
# Run a registered recipe with default configuration
ehp-sn train run tem-v1-arena

# Use a different complete configuration file
ehp-sn train run --config config/training/custom.toml

# Select a recipe and layer a partial configuration over defaults
ehp-sn train run tem-v1-arena \
    --config config/training/local.toml

# Apply leaf overrides without a config file
ehp-sn train run tem-v1-arena \
    --set optimizer.lr=1e-4 \
    --set trainer.max_steps=100000

# Use a hardware profile
ehp-sn train run tem-v2-arena \
    --profile gpu-16gb

# Resume an interrupted run
ehp-sn train run tem-v1-arena \
    --resume checkpoints/run-abc123/last.ckpt

# Initialize weights from a pretrained model but start a fresh run
ehp-sn train run tem-v1-arena \
    --init-from artifacts/tem-pretrained/weights.pt \
    --init-group all

# Inspect the resolved configuration without executing
ehp-sn train run tem-v1-arena \
    --set optimizer.lr=5e-5 \
    --dry-run --print-config

# Self-identifying config (recipe declared in TOML)
ehp-sn train run --config config/training/custom-arena.toml
```

#### Behaviour

| Condition                                 | Behaviour                                                                                                                            |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `RECIPE` and `--config` both present      | Recipe selects schema and builder; config TOML overlays defaults. The recipe must match the config's declared recipe field (if any). |
| Only `--config` present                   | Config TOML must declare `recipe = "..."` to identify the schema.                                                                    |
| Only `RECIPE` present                     | Default recipe configuration is loaded.                                                                                              |
| Neither present                           | Error — at least one is required.                                                                                                    |
| `--resume` and `--init-from` both present | Error — mutually exclusive.                                                                                                          |
| `--dry-run`                               | Resolution and validation run to completion; training is skipped.                                                                    |

---

### `train list`

List registered training recipes.

```
ehp-sn train list [OPTIONS]
```

| Option                 | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `--task TEXT`          | Filter by task name (e.g. `arena`, `mazehard`).   |
| `--model-family TEXT`  | Filter by model family (e.g. `tem-v1`, `hrm-v2`). |
| `--format table\|json` | Output format. Default: `table`.                  |

#### Example output (table)

```
NAME               TASK       MODEL      OBJECTIVE           DEFAULT PROFILE
tem-v1-arena       arena      tem-v1     variational-replay  gpu-8gb
tem-v2-arena       arena      tem-v2     variational-replay  gpu-8gb
hrm-v1-mazehard    mazehard   hrm-v1     supervised-act      gpu-8gb
hrm-v1-routebind   routebind  hrm-v1     supervised-act      gpu-8gb
hrm-v1-seqmaze     seqmaze    hrm-v1     supervised-act      gpu-8gb
hrm-v1-goaltrace   goaltrace  hrm-v1     supervised-act      gpu-8gb
hrm-v2-mazehard    mazehard   hrm-v2     actor-critic        gpu-8gb
hrm-v2-seqmaze     seqmaze    hrm-v2     actor-critic        gpu-8gb
```

---

### `train show`

Resolve the effective configuration for a recipe and emit it without
constructing datasets, models, or a Trainer.

```
ehp-sn train show [RECIPE] [OPTIONS]
```

| Option                | Description                                  |
| --------------------- | -------------------------------------------- |
| `-c, --config PATH`   | Additional or standalone configuration TOML. |
| `--profile NAME`      | Execution profile.                           |
| `--set PATH=VALUE`    | Repeatable typed configuration override.     |
| `--format toml\|json` | Output format. Default: `toml`.              |

This is useful for:

- Inspecting the full effective configuration before a run.
- Debugging override resolution.
- Sharing resolved configs with collaborators.

---

### `train validate`

Validate a training configuration and runtime prerequisites. Three
validation levels with escalating cost.

```
ehp-sn train validate [RECIPE] [OPTIONS]
```

| Option                             | Description                                  |
| ---------------------------------- | -------------------------------------------- |
| `-c, --config PATH`                | Additional or standalone configuration TOML. |
| `--profile NAME`                   | Execution profile.                           |
| `--set PATH=VALUE`                 | Repeatable typed configuration override.     |
| `--level config\|resources\|build` | Validation depth. Default: `resources`.      |

#### Validation levels

| Level       | Checks                                                                                                                                                                                        |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config`    | Parse, compose, and schema-validate the effective configuration.                                                                                                                              |
| `resources` | All of `config`, plus: dataset paths exist, model checkpoint is accessible, accelerator is available, checkpoint compatibility, output directory is writable, `--resume` checkpoint is valid. |
| `build`     | All of `resources`, plus: construct the Lightning experiment (module + datamodule) and verify forward pass runs on a single batch.                                                            |

---

## Recipe catalogue

A recipe entry is a stable identity with dispatch metadata and a builder.
It is not a configuration file — the full parameterisation of a training run
lives in the existing `config/training/*.toml` files.

### Registered recipes

| Name               | Task      | Model  | Objective          |
| ------------------ | --------- | ------ | ------------------ |
| `tem-v1-arena`     | arena     | tem-v1 | variational-replay |
| `tem-v2-arena`     | arena     | tem-v2 | variational-replay |
| `hrm-v1-mazehard`  | mazehard  | hrm-v1 | supervised-act     |
| `hrm-v1-routebind` | routebind | hrm-v1 | supervised-act     |
| `hrm-v1-seqmaze`   | seqmaze   | hrm-v1 | supervised-act     |
| `hrm-v1-goaltrace` | goaltrace | hrm-v1 | supervised-act     |
| `hrm-v2-mazehard`  | mazehard  | hrm-v2 | actor-critic       |
| `hrm-v2-seqmaze`   | seqmaze   | hrm-v2 | actor-critic       |

### Recipe entry structure

```python
@dataclass(frozen=True, slots=True)
class TrainingRecipe:
    name: TrainingRecipeName
    task: str
    model_family: str
    objective_family: str
    default_config: Traversable
    config_type: type[TrainingConfig]
    builder: TrainingExperimentBuilder
    description: str
```

Recipes are registered in a lazy-populated catalogue (`TrainingRecipeCatalogue`)
inside `ehp_sn.training.recipes`. The catalogue is intentionally closed;
adding a new recipe requires a code change.

---

## Configuration precedence

The final effective configuration is resolved by layering sources in strict
order. Later sources override earlier ones.

```
 1. Schema defaults              ConfigType field defaults
 2. Recipe default config        config/training/recipes/{name}.toml
 3. Execution profile            config/training/profiles/{name}.toml
 4. User --config file           Explicit additional or standalone config
 5. --set overrides              Repeatable typed leaf overrides
 6. Dedicated operational flags  --seed, --device, --precision, etc.
```

Dedicated flags (`--seed`, `--device`, `--precision`, `--output-dir`,
`--run-id`, `--resume`, `--init-from`) override equivalent configuration
fields. The resolved configuration records that the value originated from
the command line.

### `--set` override syntax

Overrides use a repeatable `--set PATH=VALUE` syntax. The path is a
dot-separated key into the configuration hierarchy. Values are parsed using
TOML scalar syntax:

```bash
--set optimizer.lr=1e-4
--set trainer.max_steps=100000
--set data.num_slots=64
--set data.augment=true
--set execution.validation.max_rollout_steps=500
```

String values must be quoted in TOML style:

```bash
--set checkpointing.selection.monitor="\"val/accuracy\""
```

The parser must reject:

- Unknown configuration keys.
- Type-invalid values.
- Overrides that would introduce structural ambiguity.

---

## Resume and initialisation

These are separate contracts with different semantics.

### Resume (`--resume`)

```bash
ehp-sn train run tem-v1-arena --resume checkpoints/run-abc/last.ckpt
```

| Property         | Value                                                                                                                                                                  |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| What is restored | Model parameters, optimizer state, scheduler state, step counters, gradient scaler, callback state, RNG state (where supported), data/rollout state (where supported). |
| New run identity | No — the same logical training lineage continues.                                                                                                                      |
| Configuration    | Loaded from the original run manifest; `--config` and `--set` are validated for compatibility.                                                                         |

### Initialisation (`--init-from`)

```bash
ehp-sn train run tem-v1-arena \
    --init-from artifacts/pretrained/model.ckpt \
    --init-group all
```

| Property         | Value                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------- |
| What is restored | Selected model parameter groups only.                                                   |
| New run identity | Yes — a new training lineage starts.                                                    |
| Configuration    | Resolved normally from recipe and overrides.                                            |
| Semantic groups  | `all`, `pfc_core`, `striatum` (HRM); family-specific groups defined by model contracts. |

`--resume` and `--init-from` are mutually exclusive.

---

## Distributed execution

The CLI supports two deployment patterns and does not reimplement `torchrun`.

### Pattern A: Trainer-managed

Lightning spawns worker processes internally.

```bash
ehp-sn train run tem-v2-arena \
    --set trainer.accelerator=cuda \
    --set trainer.devices=4 \
    --set trainer.strategy=ddp
```

### Pattern B: External launcher

The user (or a job scheduler) launches workers directly.

```bash
torchrun --standalone --nproc-per-node=4 \
    -m ehp_sn.cli train run tem-v2-arena \
    --set trainer.strategy=ddp
```

The application detects that it is running inside a distributed launch
(via environment variables set by `torchrun`) and must not recursively
spawn another process group.

Scheduler submission (SLURM, Kubernetes) is the responsibility of a
separate `jobs` or `launch` surface.

---

## Exit codes

| Code  | Meaning                                         |
| ----- | ----------------------------------------------- |
| `0`   | Success.                                        |
| `2`   | CLI usage or parsing error.                     |
| `3`   | Configuration resolution or validation failure. |
| `4`   | Missing resource or incompatible artifact.      |
| `5`   | Experiment construction failure.                |
| `6`   | Training execution failure.                     |
| `7`   | Artifact publication failure.                   |
| `130` | Interrupted by user (SIGINT).                   |

Errors identify: what failed, which source introduced the value, the
invalid value, the expected form, and the probable correction.

---

## Run manifest

Every successful resolution produces a run manifest before training starts.
The manifest is written to the run output directory.

```
runs/{run-id}/
├── manifest.toml
├── request.toml
├── resolved-config.toml
├── environment.json
├── metrics/
├── checkpoints/
├── artifacts/
└── logs/
```

### `manifest.toml`

```toml
schema_version = 1
run_id = "20260701-213552-tem-v1-arena-a73c"
recipe = "tem-v1-arena"
config_digest = "sha256:a1b2c3d4e5f6g7h8i9j0"
created_at = "2026-07-01T21:35:52+02:00"
seed = 42
resume_mode = "fresh"

[source]
git_commit = "abc123def456"
git_dirty = false

[inputs]
config = "config/training/recipes/tem-v1-arena.toml"
profile = "gpu-8gb"

[artifacts]
run_dir = "runs/20260701-213552-tem-v1-arena-a73c"
```

---

## Configuration self-identification

Standalone configuration TOMLs may declare the recipe they belong to:

```toml
schema_version = 1
recipe = "tem-v1-arena"

[name]
value = "tem-v1-arena-baseline"
```

When `--config` is supplied without a positional recipe, the `recipe` field
identifies the schema and builder. When both are supplied, the CLI verifies
they match.

---

## Hardware profiles

Hardware profiles separate environment-dependent settings from experiment
semantics. They live in `config/training/profiles/`.

```
config/training/profiles/
├── local.toml
├── gpu-8gb.toml
├── gpu-16gb.toml
└── cluster.toml
```

A profile typically overrides `trainer` settings only:

```toml
[trainer]
accelerator = "gpu"
devices = 1
precision = "16-mixed"
enable_progress_bar = true
```

Selected via `--profile gpu-8gb`. When no profile is given, the recipe's
default profile is used.

---

## Underlying Python API

The CLI is a thin adapter around the training application service. The
same API can be called from tests, notebooks, SLURM entry points, and
automated experiment agents.

```python
from ehc_sn.training.requests import TrainingRunRequest
from ehc_sn.training.service import run_training_request

request = TrainingRunRequest(
    recipe="tem-v1-arena",
    config_path=None,
    profile=None,
    overrides=("optimizer.lr=1e-4",),
    resume_from=None,
    initialize_from=Path("artifacts/pretrained/model.ckpt"),
    initialize_groups=("all",),
    output_dir=None,
    run_id=None,
    seed=42,
    tags={"author": "bot"},
    dry_run=False,
)

result = run_training_request(request)
print(f"run_id={result.run_id}")
print(f"output_dir={result.output_dir}")
```

### Public types

```python
@dataclass(frozen=True, slots=True)
class TrainingRunRequest:
    recipe: str | None
    config_path: Path | None
    profile: str | None
    overrides: tuple[str, ...]
    resume_from: Path | None
    initialize_from: Path | None
    initialize_groups: tuple[str, ...]
    output_dir: Path | None
    run_id: str | None
    seed: int | None
    tags: Mapping[str, str]
    dry_run: bool = False

@dataclass(frozen=True, slots=True)
class ResolvedTrainingRun:
    recipe: TrainingRecipe
    config: TrainingConfig
    provenance: ConfigProvenance
    run_identity: TrainingRunIdentity
    config_digest: str

@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    run_id: str
    output_dir: Path
    final_checkpoint: Path | None
    model_artifact: Path | None
    status: str
```

### Public functions

```python
def resolve_training_run(
    request: TrainingRunRequest,
) -> ResolvedTrainingRun:
    """Resolve recipes, configs, profiles, and overrides into an
    immutable effective configuration and run identity."""

def validate_training_run(
    resolved: ResolvedTrainingRun,
    *,
    level: ValidationLevel,
) -> ValidationReport:
    """Validate the resolved configuration at the specified depth."""

def build_training_experiment(
    resolved: ResolvedTrainingRun,
) -> TrainingExperiment:
    """Construct the Lightning module and datamodule from a resolved run."""

def execute_training_run(
    resolved: ResolvedTrainingRun,
) -> TrainingRunResult:
    """Execute training from a resolved run (no re-resolution)."""

def run_training_request(
    request: TrainingRunRequest,
) -> TrainingRunResult:
    """Full pipeline: resolve → validate → build → execute."""
```

---

## Migration path

Existing scripts in `scripts/training/` are deprecated wrappers that
call the public service API directly:

```python
# scripts/training/tem_v1_arena.py
from ehc_sn.training.requests import TrainingRunRequest
from ehc_sn.training.service import run_training_request

def main() -> None:
    run_training_request(TrainingRunRequest.for_recipe("tem-v1-arena"))

if __name__ == "__main__":
    main()
```

These wrappers are preserved for backward compatibility and scheduled
for removal in a future release. New tooling and documentation should
use the CLI surface.

---

## Module layout

```
src/ehp_sn/training/
├── __init__.py
├── cli.py              Typer declarations, shell types, exit codes
├── recipes.py          TrainingRecipe, TrainingRecipeName, catalogue
├── configuration.py    Config schemas, loading, composition, typed overrides
├── requests.py         TrainingRunRequest, ResolvedTrainingRun, TrainingRunResult
├── resolution.py       Recipe/config/profile/override resolution
├── validation.py       Config, resource, compatibility, and build validation
├── service.py          Application-level orchestration
├── runner.py           Framework execution (existing run_training)
├── artifacts.py        Run manifest and artifact publication
└── ...                 (existing tem.py, hrm.py, optim.py, etc.)
```

---

## Related documentation

| Document                                      | Description                                                                 |
| --------------------------------------------- | --------------------------------------------------------------------------- |
| [`design/training.md`](../design/training.md) | Internal design — training execution policy, contracts, and layer ownership |
| [`cli/index.md`](index.md)                    | CLI overview and conventions                                                |
| [`cli/evaluate.md`](evaluate.md)              | Evaluation CLI — companion to training                                      |
