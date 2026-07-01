---
title: Data Generation CLI
description: ehp-sn data — lifecycle commands for interim-data substrates
---

# `ehp-sn data` — Interim-Data Generation

The `data` CLI manages **interim-data substrates** — versioned, immutable
artefacts that describe world topologies without task-specific protocol
(trajectories, episodes, supervision).  Task-ready corpora belong under
[`ehp-sn tasks`](tasks.md).

## Ownership boundary

```
external/raw / synthetic sources
        │
        ▼
ehp-sn data                          ◄── YOU ARE HERE
        │
        ▼
data/interim/<family>/<variant>/v<N>/   manifest-bearing substrate
        │
        ▼
ehp-sn tasks
        │
        ▼
data/processed/<task>/<corpus>/v<N>/    task corpus
```

`data` owns:

- Acquisition or synthesis of source material.
- Topology and environment generation.
- Normalisation into canonical substrate schemas.
- Deterministic split assignment when splits are intrinsic to the substrate.
- Sensory enrichment that belongs to the environment (observation ids, landmarks).
- Manifest, provenance, checksums, schema version and validation.
- Materialisation under `data/interim/`.

`data` must not own:

- Task examples or episodes.
- Labels and learning targets.
- Task-specific sequence construction.
- Model tokenisation.
- Training batches.
- Task-scoring semantics.

## Stable commands

| Command | Operates on | Purpose |
|---|---|---|
| `list` | — | List registered substrate generators |
| `show` | target | Describe a generator and its configuration contract |
| `plan` | target | Resolve configuration and display a build plan without writing data |
| `build` | target | Materialise one immutable interim-data artefact |
| `validate` | artefact | Validate an already-materialised artefact |
| `inspect` | artefact | Display metadata and bounded samples from an artefact |
| `clean` | target | Remove temporary or failed build state (optional) |
| `migrate` | artefact | Schema migration — creates a new artefact (future) |

## Registered substrate generators

```
ehp-sn data list
```

| Target | Kind | Default config | Description |
|---|---|---|---|
| `dagflow` | synthetic | `config/data/dagflow/default.toml` | Directed acyclic graph substrates |
| `dungeongen` | synthetic | `config/data/dungeongen/default.toml` | Dungeon topology and sensory layouts |
| `maze-nd` | imported | `config/data/maze-nd/default.toml` | Shared N-dimensional maze substrate |
| `openfield` | synthetic | `config/data/openfield/default.toml` | Open-field spatial layouts |

```
$ ehp-sn data list

TARGET         KIND       DEFAULT CONFIG                             DESCRIPTION
dagflow        synthetic  config/data/dagflow/default.toml           Directed acyclic graph substrates
dungeongen     synthetic  config/data/dungeongen/default.toml        Dungeon topology and sensory layouts
maze-nd        imported   config/data/maze-nd/default.toml           Shared N-dimensional maze substrate
openfield      synthetic  config/data/openfield/default.toml         Open-field spatial layouts
```

Pass `--json` for machine-readable output:

```
$ ehp-sn data list --json
```

```json
[
  {
    "target": "dagflow",
    "kind": "synthetic",
    "default_config": "config/data/dagflow/default.toml",
    "description": "Directed acyclic graph substrates"
  },
  ...
]
```

## `data show TARGET`

Describes a generator's contract without materialising anything:

```
ehp-sn data show dungeongen
```

Reports:

- Generator identifier.
- Description.
- Config schema.
- Default config path.
- Output artefact type.
- Output path convention.
- Required external sources.
- Validator.
- Current schema version.

```
$ ehp-sn data show dungeongen

target:            dungeongen
kind:              synthetic
description:       Procedural dungeon topology and sensory layout generator
default_config:    config/data/dungeongen/default.toml
config_schema:     1
output_kind:       interim-substrate
output_pattern:    data/interim/dungeongen/{variant}/v{version}/
external_sources:  dungeongen (PyPI library)
validator:         validate_dungeongen_layout_root
```

## `data plan TARGET`

Resolves the configuration and prints what would happen without writing
anything:

```
ehp-sn data plan dungeongen \
    --config config/data/dungeongen/routebind-30.toml

ehp-sn data plan dungeongen \
    --config config/data/dungeongen/routebind-30.toml \
    --json
```

`plan` is strictly more useful than a weak `--dry-run` flag because it:

- Has a stable machine-readable output schema (`--json`).
- Is independently discoverable (it is its own command).
- Can be checked into CI review pipelines.

It resolves and reports:

- Effective configuration (config + CLi overrides merged).
- Destination version root.
- Input dependencies (raw sources, staging files).
- Expected output structure.
- Existing-output conflicts (version root already exists).
- Deterministic identity / fingerprint.
- Stages to execute or reuse.

## `data build TARGET`

Materialises one immutable interim-data artefact:

```
ehp-sn data build dungeongen \
    --config config/data/dungeongen/routebind-30.toml

ehp-sn data build dungeongen \
    --config config/data/dungeongen/default.toml \
    --output-root /fast-scratch/datasets \
    --force
```

### Semantics

1. Load and validate configuration.
2. Resolve the registered generator from the target name.
3. Resolve input sources and destination path.
4. Check for existing artefact at the destination — refuse unsafe overwrites
   unless `--force` is passed.
5. Generate into a **temporary staging directory** (`.building-v<N>` sibling).
6. Validate the staged result structurally and semantically.
7. **Atomically** rename the staging directory to the final version root.
8. Write manifest, provenance, and checksums.
9. Print the artefact reference to stdout.

A successful build must never leave a partially published version root.

### Configuration-first interface

Generator parameters are **not** first-class CLI flags.  They live in
version-controlled TOML configuration files under `config/data/`.

```
ehp-sn data build dungeongen \
    --config config/data/dungeongen/routebind-30.toml
```

Example configuration (`config/data/dungeongen/routebind-30.toml`):

```toml
schema_version = 1
target = "dungeongen"
variant = "routebind-30"
version = 1
seed = 1729

[output]
root = "data/interim"

[splits]
train = 4000
validation = 500
test = 500

[topology]
preset = "routebind-30"
pad_height = 30
pad_width = 30

[sensory]
vocabulary_size = 45
instances = 1
policy = "bounded-occurrence"
max_occurrences = 4
```

### CLI overrides

The CLI exposes a deliberately small override surface:

| Option | Purpose |
|---|---|
| `--config`, `-c <PATH>` | Generator configuration file |
| `--set <KEY>=<VALUE>` | Override a single configuration value |
| `--output-root <PATH>` | Override the configured interim-data root |
| `--seed <INT>` | Override the base seed |
| `--force` | Replace an existing equivalent artefact |
| `--dry-run` | Print what would happen without building |
| `--json` | Emit the build result as machine-readable JSON |
| `--quiet` | Suppress informational output |
| `--verbose` | Show detailed progress |

The guiding principle:

> A flag controls **invocation behaviour**; a configuration field **defines
> the artefact**.

Therefore `--dry-run`, `--force`, `--json`, `--quiet` are CLI flags, while
`pad_height`, `edge_density`, `sensory.policy` are configuration fields.

### Artefact contract

Every successful build produces a standard version root:

```
data/interim/<family>/<variant>/v<N>/
├── manifest.json
├── config.resolved.toml
├── provenance.json
└── splits/
    ├── train/        (or train.*)
    ├── validation/   (or validation.*)
    └── test/         (or test.*)
```

`manifest.json` structure:

```json
{
  "artifact_kind": "interim-substrate",
  "schema_version": 1,
  "target": "dungeongen",
  "variant": "routebind-30",
  "version": 1,
  "fingerprint": "a1b2c3d4e5f6...",
  "generator_version": "1.0.0",
  "created_at": "2026-07-01T12:00:00Z",
  "config": "config.resolved.toml",
  "inputs": [],
  "splits": {
    "train": 4000,
    "validation": 500,
    "test": 500
  },
  "files": [
    "splits/train/topology.npy",
    "splits/train/observations.npy",
    "splits/validation/topology.npy",
    ...
  ]
}
```

## `data validate ARTEFACT`

Validates an already-materialised artefact without modifying it:

```
ehp-sn data validate data/interim/dungeongen/routebind-30/v1

ehp-sn data validate data/interim/dungeongen/routebind-30/v1 \
    --level quick

ehp-sn data validate data/interim/dungeongen/routebind-30/v1 \
    --level full --json
```

Validation covers:

- Manifest schema compliance.
- Declared family, variant, and version.
- Required files present and readable.
- Channel array shapes, dtypes, and value ranges.
- Split cardinality matches manifest declarations.
- Cross-file referential integrity (index entries ↔ array lengths).
- Content checksums where applicable.
- Domain-specific invariants (largest-component connectivity,
  observation-id bounds, adjacency matrix consistency).
- Compatibility with the current reader contract.

### Validation levels

| Level | Scope |
|---|---|
| `quick` | Manifest schema, file existence, top-level invariants |
| `full` (default) | All checks including data-intensive traversals |

## `data inspect ARTEFACT`

Displays artefact metadata and bounded samples.  Human-oriented:

```
ehp-sn data inspect data/interim/dagflow/default/v1

ehp-sn data inspect data/interim/dagflow/default/v1 \
    --sample 3

ehp-sn data inspect data/interim/dungeongen/routebind-30/v1 \
    --json
```

Reports:

- Artefact identity (target, variant, version).
- Generator and config fingerprint.
- Schema version.
- Creation metadata.
- Split counts.
- Channel dimensions and value ranges.
- Files and sizes.
- Representative records (when `--sample N`).
- Validation summary (pass/fail state of last known checks).

## `data clean TARGET` (optional)

Cleans temporary or failed build state:

```
ehp-sn data clean dungeongen             # clean all .building-* dirs
ehp-sn data clean dungeongen --staging   # clean only staging temp files
ehp-sn data clean dungeongen --failed    # clean failed partial build dirs
```

Deleting a published artefact requires an explicit path and confirmation:

```
ehp-sn data clean data/interim/dungeongen/routebind-30/v1 --yes
```

## `data migrate ARTEFACT` (future)

Schema migration creates a new artefact; it never mutates an existing one:

```
ehp-sn data migrate data/interim/dungeongen/v1 \
    --to-schema 2 \
    --output data/interim/dungeongen/v2
```

## Internal Python API

The CLI is a thin adapter.  It delegates to a registry-driven internal API
in `ehc_sn.data.api`:

| Function | Purpose |
|---|---|
| `list_generators() -> tuple[GeneratorInfo, ...]` | Registered generator catalogue |
| `describe_generator(target) -> GeneratorInfo` | Single generator metadata |
| `plan_build(request: DataBuildRequest) -> DataBuildPlan` | Resolve and preview a build |
| `build_data(request: DataBuildRequest) -> DataBuildResult` | Execute a build |
| `validate_data(request: DataValidationRequest) -> ValidationReport` | Validate an artefact |
| `inspect_data(request: DataInspectionRequest) -> DataInspection` | Inspect an artefact |

Core request types:

```python
@dataclass(frozen=True)
class DataBuildRequest:
    target: str                 # generator name
    config_path: Path | None    # TOML config
    overrides: Mapping[str, object]  # --set KEY=VALUE
    output_root: Path | None    # override for output.root
    force: bool                 # overwrite flag
```

```python
@dataclass(frozen=True)
class DataBuildPlan:
    target: str
    artifact: ArtifactRef
    resolved_config: Mapping[str, object]
    inputs: tuple[InputRef, ...]
    outputs: tuple[Path, ...]
    fingerprint: str
    actions: tuple[PlannedAction, ...]
```

```python
@dataclass(frozen=True)
class DataBuildResult:
    artifact: ArtifactRef
    manifest_path: Path
    fingerprint: str
    validation: ValidationReport
    reused: bool                 # True if no new work was needed
```

Generator extension protocol:

```python
class DataGenerator(Protocol):
    name: ClassVar[str]
    config_type: type[DataConfig]

    def plan(self, config: DataConfig, context: DataBuildContext) -> DataBuildPlan: ...
    def build(self, plan: DataBuildPlan, workspace: Path) -> None: ...
    def validate(self, artifact: ArtifactRef, *, level: ValidationLevel) -> ValidationReport: ...
    def inspect(self, artifact: ArtifactRef, *, sample_size: int) -> DataInspection: ...
```

Registration:

```python
DATA_GENERATORS = GeneratorRegistry()
DATA_GENERATORS.register(DungeonGenGenerator())
DATA_GENERATORS.register(OpenFieldGenerator())
DATA_GENERATORS.register(MazeNDGenerator())
DATA_GENERATORS.register(DagFlowGenerator())
```

## Behavioural guarantees

| Property | Requirement |
|---|---|
| **Determinism** | Same code version + same resolved config + same input fingerprints → same logical artefact |
| **Immutability** | A published version root is never silently modified |
| **Idempotence** | Re-running an equivalent build reuses the artefact or reports it already exists |
| **Atomic publication** | Generate and validate in staging, then atomic rename into place |
| **Explicit replacement** | `--force` is never implicit; operates only after destination identity is resolved |
| **Machine-readable output** | Every informational command supports `--json` |
| **Stable exit codes** | 0=success, 1=build failure, 2=bad invocation, 3=validation failure, 4=not found, 5=conflict |
| **No tracebacks by default** | Concise domain errors; `--verbose` or `EHP_DEBUG=1` for Python tracebacks |
| **No CI prompts** | All commands support deterministic noninteractive execution |
| **Bounded inspection** | `inspect` never dumps entire datasets |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success / valid |
| `1` | Operational build failure |
| `2` | Invalid CLI invocation or configuration error |
| `3` | Validation failure |
| `4` | Target or artefact not found |
| `5` | Artefact conflict (version root already exists without `--force`) |

## Comparison: registry-driven vs. sub-Typer-per-generator

| Concern | Registry-driven (adopted) | Sub-Typer per generator (alternative) |
|---|---|---|
| **Command grammar** | Stable: `data build TARGET` | Changes when generators are added or renamed |
| **Lifecycle consistency** | One set of verbs (`build`, `validate`, `inspect`) shared across all generators | Each generator can drift to different verb sets |
| **Parameter surface** | Config-first; CLI stays small | Generator flags proliferate on CLI |
| **Extensibility** | Register a new class; no CLI changes | Add a new Typer sub-app |
| **Machine-readable automation** | Uniform `--json` on all commands | JSON must be reimplemented per sub-app |
| **Discoverability** | `data list` shows all generators | Must inspect repo to find sub-apps |

## See also

- [Substrates overview](../substrates/_index.md) — substrate families and their properties
- [Design: Data subsystem](../design/data.md) — two-plane architecture
- [Tasks CLI](tasks.md) — downstream task corpus generation
