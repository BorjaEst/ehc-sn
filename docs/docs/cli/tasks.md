# Tasks CLI — `ehp tasks`

> Lifecycle-oriented CLI for generating, validating, and inspecting
> processed task corpora.  Operation-first design: the verb is the
> command, the task family is the argument.

---

## Overview

The `tasks` command group owns the step from **interim substrates**
(spatial layouts, DAG graphs, maze grids) to **versioned processed task
corpora** ready for training and evaluation.

```
data/interim/ or data/external/
        │
        ▼  substrate builders (data pipelines, external datasets)
data/processed/<substrate-family>/v<N>/
        │
        ▼  ehp tasks build <task> --config <config.toml>
data/processed/<task>/<corpus>/v<N>/
        │
        ▼  training and evaluation entrypoints
```

---

## Quick reference

```
ehp tasks list
ehp tasks describe TASK [--show-schema] [--format text|json]
ehp tasks plan   TASK --config PATH [--set KEY=VALUE]...
ehp tasks build  TASK --config PATH [--output PATH] [--set KEY=VALUE]...
                  [--dry-run] [--non-interactive]
ehp tasks validate CORPUS [--level manifest|sample|full]
                  [--split SPLIT] [--format text|json]
ehp tasks inspect CORPUS [--split SPLIT] [--sample INDEX]
                  [--summary] [--render-dir PATH]
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Success |
| `1`  | Generation or validation failure |
| `2`  | Invalid CLI usage or configuration |
| `3`  | Existing immutable output (use `--force` to override) |
| `4`  | Missing or incompatible input artifact |

---

## Commands

### `ehp tasks list`

List registered task families and their status.

```
ehp tasks list
ehp tasks list --format json
```

Example output:

```
TASK         INPUT SUBSTRATE     DEFAULT CORPUS    STATUS
arena        spatial-layout      default           stable
mazehard     maze substrate      default           stable
seqmaze      DAG layout          default           experimental
goaltrace    DAG layout          default           experimental
routebind    topology + DAG      default           experimental
```

The task registry is defined programmatically in
[`src/ehc_sn/tasks/registry.py`][src].  Every task plugin exports
a `registered_name`, a `config_type` (Pydantic model), and a
`description` string.

[src]: https://github.com/borja-eb/ehc-sn/tree/main/src/ehc_sn/tasks

---

### `ehp tasks describe TASK`

Describe a task family: its contract, required inputs, configuration
schema, produced channels, and output format.

```
ehp tasks describe arena
ehp tasks describe goaltrace --format json
ehp tasks describe routebind --show-schema
```

When `--show-schema` is passed the command prints the full Pydantic
configuration schema for the task, including field types, defaults,
and documentation strings.  This is the canonical reference for
authoring task generation TOML files.

The plain-text output summarises:

- **Task** — family name and protocol version.
- **Contract** — what the model must predict (observation, field, path…).
- **Inputs** — what interim substrates or parent artifacts are required.
- **Channels** — list of all channels in the produced corpus.
- **Presets** — named configuration presets (if any).
- **Stability** — stable / experimental / deprecated.

---

### `ehp tasks plan TASK`

Resolve generation configuration and display intended output without
writing any artifacts.  A dry run for task corpus generation.

```
ehp tasks plan arena \
    --config config/tasks/arena/default.toml

ehp tasks plan goaltrace \
    --config config/tasks/goaltrace/default.toml \
    --set generation.seed=7
```

The command prints:

```
Task:        arena
Protocol:    v1
Config:      config/tasks/arena/default.toml

Inputs:
  layout_root:  data/interim/tem-square/v1  (45 layouts)

Output:
  root:         data/processed/arena/default/v2
  status:       NEW (does not exist)

Splits:
  train:  400  (40 layouts × 10 episodes)
  val:     40  ( 4 layouts × 10 episodes)
  test:    40  ( 4 layouts × 10 episodes)

Generation:
  walk_policy:         angle_bias
  n_episodes_per_layout: 10
  max_steps:          250
  seed:               42
```

If the output root already exists the plan reports a conflict:

```
Output:
  root:         data/processed/arena/default/v1
  status:       EXISTS (immutable, will refuse to overwrite)
```

---

### `ehp tasks build TASK`

Build a versioned processed task corpus.

```
ehp tasks build arena \
    --config config/tasks/arena/default.toml

ehp tasks build goaltrace \
    --config config/tasks/goaltrace/default.toml \
    --set generation.seed=7 \
    --set generation.distance_tau=6.0

ehp tasks build mazehard \
    --config config/tasks/mazehard/default.toml \
    --output data/processed/mazehard/experiment-v2/v1 \
    --non-interactive
```

#### Parameters

| Option | Description |
|--------|-------------|
| `TASK` | Registered task-family name (positional argument). |
| `--config PATH` | Path to a TOML generation configuration (required). |
| `--output PATH` | Override the configured output directory. |
| `--set KEY=VALUE` | Override a single config value (repeatable). |
| `--dry-run` | Resolve configuration and inputs only; do not write. |
| `--non-interactive` | Fail instead of prompting on conflicts. |

#### Configuration file

Task-specific parameters live in typed TOML configuration files, not
on the command line.  Example for Arena:

```toml
# config/tasks/arena/default.toml

schema_version = 1
task = "arena"
corpus = "default"
version = 1

[inputs]
layout_root = "data/interim/tem-square/v1"

[generation]
seed = 42
walk_policy = "angle_bias"
n_episodes_per_layout = 10
max_steps = 250

[splits]
train = 400
validation = 40
test = 40
```

For Goaltrace:

```toml
# config/tasks/goaltrace/default.toml

schema_version = 1
task = "goaltrace"
corpus = "default"
version = 1

[inputs]
layout_root = "data/interim/dagflow/sparse/v1"

[generation]
seed = 42
n_observations = 45
oracle_semantics = "reliability"
field_decay = 0.8
static_weights = true
distance_tau = 8.0

[splits]
train = 500
validation = 250
test = 240
```

#### Behavioural guarantees

| Property | Implementation |
|----------|---------------|
| **Determinism** | Same config + seed + parent fingerprints → identical output. |
| **Atomic publication** | Write to staging directory; rename only after successful validation. |
| **Immutability** | Refuse to overwrite an existing version root by default. |
| **Provenance** | Record parent artifacts, config fingerprint, schema version, seed in manifest. |
| **Machine-readable output** | `--format json` supported for CI consumption. |

On success the command prints:

```
Task:        arena
Config:      config/tasks/arena/default.toml
Output:      data/processed/arena/default/v2
Fingerprint: a1b2c3d4e5f67890
Samples:     train=400, validation=40, test=40
Status:      completed
```

---

### `ehp tasks validate CORPUS`

Validate a processed task corpus against its declared schema and
semantic invariants.

```
ehp tasks validate data/processed/arena/default/v2
ehp tasks validate data/processed/arena/default/v2 --level full
ehp tasks validate data/processed/arena/default/v2 --split val
ehp tasks validate data/processed/arena/default/v2 --format json
```

The task family is read from the corpus `manifest.json` — users never
repeat the task name.  This prevents contradictory invocations such
as validating an Arena corpus through MazeHard validation logic.

#### Validation levels

| Level | Checks |
|-------|--------|
| `manifest` | Manifest fields, path grammar, channel declarations |
| `sample` | Manifest-level + per-sample structural invariants (shapes, dtypes, sentinels) |
| `full` | Sample-level + task-specific semantic invariants (oracle consistency, revisit validity, support channel correctness) |

On success:

```
Corpus:      data/processed/arena/default/v2
Task:        arena
Schema:      v1
Level:       full
Result:      PASS
Samples:     train=400, validation=40, test=40 (all clean)
```

On failure detailed per-sample issues are reported:

```
Corpus:      data/processed/arena/default/v2
Level:       full
Result:      FAIL  (2 errors, 3 warnings)

Errors:
  [train/012] OBSERVATION_OUT_OF_RANGE: observation_id=52 exceeds vocab size 45
  [val/003]  EPISODE_START_MISMATCH: step 0 is not marked episode_start

Warnings:
  [train/045] HIGH_REVISIT_RATE: revisit_rate=0.92 exceeds 0.85 threshold
```

---

### `ehp tasks inspect CORPUS`

Inspect corpus metadata, aggregate statistics, or individual samples.

```
ehp tasks inspect data/processed/arena/default/v2
ehp tasks inspect data/processed/arena/default/v2 --summary
ehp tasks inspect data/processed/arena/default/v2 --split val --sample 12
ehp tasks inspect data/processed/arena/default/v2 --render-dir artifacts/inspection
```

Without flags the command prints a corpus overview:

```
Corpus:      data/processed/arena/default/v2
Task:        arena
Version:     2
Channels:    12 (trajectory_row, trajectory_col, …)
Fingerprint: a1b2c3d4e5f67890

Splits:
  train:  400 episodes
  val:     40 episodes
  test:    40 episodes

Episode length distribution (all splits):
  min:     10
  median:  87
  mean:    92.3
  max:    250

Revisit rate distribution:
  min:     0.03
  median:  0.42
  mean:    0.44
  max:     0.91
```

With `--sample INDEX` the command prints a decoded human-readable view
of a single episode, including the observation sequence, coordinate
trajectory, revisit pattern, and wall density of the associated layout.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI layer                               │
│  ehc_sn/tasks/cli.py (Typer adapter only)                      │
│  Commands: list, describe, plan, build, validate, inspect       │
└──────────────────────────────┬──────────────────────────────────┘
                               │ dispatches to
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Service / Orchestration                      │
│  ehc_sn/tasks/service.py — build / validate / inspect           │
│  ehc_sn/tasks/registry.py — TaskPlugin protocol + registry      │
│  ehc_sn/tasks/requests.py — TaskBuildRequest / result models   │
│  ehc_sn/tasks/config.py    — shared config loading + overrides │
└──────────────────────────────┬──────────────────────────────────┘
                               │ delegates to task plugin
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Task plugins (one per family)                  │
│  ehc_sn/tasks/arena/plugin.py      ArenaTaskPlugin              │
│  ehc_sn/tasks/mazehard/plugin.py   MazeHardTaskPlugin           │
│  ehc_sn/tasks/seqmaze/plugin.py    SeqMazeTaskPlugin            │
│  ehc_sn/tasks/goaltrace/plugin.py  GoalTraceTaskPlugin          │
│  ehc_sn/tasks/routebind/plugin.py  RouteBindTaskPlugin          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ calls existing builder functions
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Existing builder / validation / inspection          │
│  ehc_sn/tasks/arena/builder.py         build_arena_task_corpus  │
│  ehc_sn/tasks/arena/validation.py      validate_arena_sample    │
│  ehc_sn/tasks/arena/inspection.py      prepare_arena_inspection │
│  … (same pattern for all 5 families)                            │
└─────────────────────────────────────────────────────────────────┘
```

### Plugin protocol

Every task family exposes a `TaskPlugin` that satisfies this protocol:

```python
@dataclass(frozen=True, slots=True)
class TaskBuildRequest:
    task: str
    config_path: Path
    output_root: Path | None
    overrides: tuple[str, ...]
    dry_run: bool

@dataclass(frozen=True, slots=True)
class TaskBuildResult:
    task: str
    corpus_root: Path
    manifest_path: Path
    fingerprint: str
    sample_counts: Mapping[str, int]

class TaskPlugin(Protocol):
    """Interface satisfied by every task-family plugin."""

    name: str
    config_type: type[TaskBuildConfig]

    def build(
        self,
        config: TaskBuildConfig,
        *,
        output_root: Path,
    ) -> TaskBuildResult: ...

    def validate(
        self,
        corpus: TaskCorpus,
        *,
        level: ValidationLevel,
    ) -> ValidationReport: ...

    def inspect(
        self,
        corpus: TaskCorpus,
        request: InspectionRequest,
    ) -> InspectionReport: ...
```

### Registry

```python
TASK_PLUGINS: Final[Mapping[str, TaskPlugin]] = {
    "arena":     ArenaTaskPlugin(),
    "mazehard":  MazeHardTaskPlugin(),
    "seqmaze":   SeqMazeTaskPlugin(),
    "goaltrace": GoalTraceTaskPlugin(),
    "routebind": RouteBindTaskPlugin(),
}
```

The CLI never contains `if task == "arena"` branching.  Every command
resolves `registry.require(task_name)` and dispatches through the
plugin interface.

---

## Corpus identity and paths

### Canonical path

```
data/processed/<task>/<corpus>/v<version>/
```

The logical identity (`task`, `corpus`, `version`) is declared in the
configuration file.  A resolver computes the physical path.  Users may
override the output root with `--output PATH`.

### Immutability

Corpus roots are immutable after publication.  A second build with the
same identity refuses with exit code 3 unless `--force` is passed
(which is exceptional and should never be used for released versions).

### Manifest

Every corpus root contains a `manifest.json` that drives validation
and inspection dispatch:

```json
{
  "artifact_type": "task_corpus",
  "task": "arena",
  "corpus": "default",
  "version": 2,
  "schema_version": 1,
  "builder_version": "1.0.0",
  "channels": [
    "trajectory_row", "trajectory_col",
    "trajectory_observation_id", "trajectory_previous_action",
    "trajectory_landmark_id", "trajectory_is_revisit",
    "trajectory_episode_start", "trajectory_valid_step",
    "trajectory_length", "topology",
    "observations", "mask_valid"
  ],
  "inputs": [
    {
      "uri": "data/interim/tem-square/v1",
      "fingerprint": "sha256:abc123..."
    }
  ],
  "config_fingerprint": "sha256:def456...",
  "splits": {
    "train": 400,
    "validation": 40,
    "test": 40
  },
  "parents": {
    "shared_substrate": {
      "family": "dungeongen",
      "root": "data/processed/dungeongen/v1",
      "version": 1
    }
  }
}
```

---

## Task families

| Task | Family | Status | Input substrate | Primary metric |
|------|--------|--------|-----------------|----------------|
| `arena` | structural navigation | `stable` | spatial-layout (`dungeongen` / `openfield`) | `accuracy_revisit` |
| `mazehard` | batch token prediction | `stable` | maze shared substrate (`maze-nd`) | `token_accuracy` |
| `seqmaze` | graph path prediction | `experimental` | DAG layout (`dagflow`) | `sequence_exact` |
| `goaltrace` | prospective field (HRM) | `experimental` | DAG layout (`dagflow`) | `field_mse` |
| `routebind` | spatial route binding | `experimental` | topology layout + DAG (`dungeongen` / `openfield` + `dagflow`) | `balanced_trajectory_field_error` |

---

## Comparison with standalone scripts

The earlier `scripts/data-gen/build-<family>.py` scripts served the
same purpose but violated three principles this CLI enforces:

1. **Single entry point.**  Users no longer need to remember which
   script lives where.  `ehp tasks build arena` is discoverable through
   `ehp tasks --help`.
2. **Operation-first hierarchy.**  `build`, `validate`, and `inspect`
   apply uniformly across all families.  New task families add one
   plugin, not three CLI branches.
3. **Configuration-driven generation.**  Builder parameters live in
   typed TOML files, not as CLI flags with inconsistent defaults.

The standalone scripts are retained during a deprecation period and
then removed.  All builder, validator, and inspector functions remain
importable Python APIs — the CLI is an orchestration adapter, not a
replacement.
