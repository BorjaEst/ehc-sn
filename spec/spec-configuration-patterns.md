# EHC-SN Configuration Patterns Specification

> Non-required companion spec for configuration taxonomy, default loading, and
> composition patterns.

## 1 Scope

This spec documents the preferred configuration shapes used by the repository.
Core ownership and validation rules remain in `spec/spec-architecture.md`,
`spec/spec-requirements.md`, and `spec/spec-standards.md`.

---

## 2 Config Types

- **Component config**
  - Base class: `pydantic.BaseModel(extra="forbid")`
  - Scope: single-component settings.
  - Lives in: same file as the owning implementation.
  - Example: `AttractorSettings`.
- **Model config**
  - Base class: `pydantic.BaseModel(extra="forbid")`
  - Scope: architecture-only settings.
  - Lives in: `models/*.py`.
  - Example: `ModelSettingsV1`.
- **Task config**
  - Base class: `pydantic.BaseModel(extra="forbid")`
  - Scope: observation/action protocol, geometry, episode semantics, reward or
    supervision semantics.
  - Lives in: `tasks/**/*.py`.
  - Example: `DungeonNavigationTaskConfig`.
- **Adapter config**
  - Base class: `pydantic.BaseModel(extra="forbid")`
  - Scope: model-task binding, packing, decoding, objective wiring, rollout
    binding.
  - Lives in: `adapters/**/*.py`.
  - Example: `TEMDungeonAdapterConfig`.
- **Lightning training config**
  - Base class: `pydantic.BaseModel(extra="forbid")`
  - Scope: optimizer, scheduler, buffers, and trainer-local execution
    settings.
  - Lives in: `lightning/**/*.py`.
  - Example: `HRMV1ModelConfig`.
- **Data config**
  - Base class: `pydantic.BaseModel(extra="forbid")`
  - Scope: dataset paths, batch size, workers.
  - Lives in: `data/*.py`.
  - Example: `DatamoduleConfig`.
- **Entry-point settings**
  - Base class: `pydantic_settings.BaseSettings(cli_parse_args=True)`
  - Scope: composition of the above plus launcher- or benchmark-local knobs.
  - Lives in: `scripts/training/*.py` and `scripts/benchmarks/*.py`.
  - Example: `RunArguments`.

Rules:

- Architectural dimensions use `frozen=True` when they must not change after
  construction.
- Model configs describe architecture only.
- Task configs may describe navigation-grounded predictive or exploratory
  tasks as well as goal-directed navigation tasks.
- Task and adapter configs hold task semantics and binding semantics.

---

## 3 Static Defaults

Canonical defaults are entry-point-owned TOML files under `config/`. Entry
points load file defaults first, then allow CLI values to override them.

```python
defaults = tomllib.load(Path(CONFIGURATION_PATH).open("rb"))
settings = RunArguments(**defaults)
```

Rules:

- Do not assume a single repo-wide defaults file.
- Prefer one canonical TOML per executable entry point or benchmark launcher.
- If an entry point supports an environment-selected config path, document the
  environment variable next to that entry point rather than in a shared global
  registry.

---

## 4 Leaf Composition Pattern

Entry-point settings are structured as a flat tree of leaf configs. Downstream
composed configs are assembled by properties rather than duplicated manually.

```text
RunArguments(BaseSettings)
  ├── architecture: PFCSettings
  ├── task: MazeHardTaskConfig
  ├── adapter: HRMMazeHardAdapterConfig
  ├── optimizer: AdamATan2Config
  ├── scheduler: SchedulerConfig
  ├── dataset: DatasetSettings
  ├── global_batch_size: int
  └── composed via @property:
      ├── .task
      ├── .adapter
      ├── .training
      └── .datamodule
```

Benefits:

- CLI overrides stay ergonomic.
- Shared values are defined once.
- Full experiment settings remain serializable.
- The TOML file remains close to the executable surface that owns it.

---

## 5 Named Evaluation Regimes

Named evaluation regimes decouple diagnostic evaluation from the fit-path
`validation_step`. They are driven by `EvaluationRegimesCallback` and
configured in entry-point settings.

### Regime config shape

```python
class EvaluationRegimeSettings(BaseModel, extra="forbid"):
    regime_id: str                       # unique ID, used as metric namespace component
    phase_kind: Literal["diag", "bench"] # namespace tier
    provider_ref: str                    # dotted import path to an EvaluationSourceProvider class
    provider_settings: dict              # forwarded verbatim to the provider's constructor
    schedule: EvaluationScheduleSettings
    trace_request: EvaluationTraceRequest
```

The `metric_namespace` property returns `f"{phase_kind}/{regime_id}/"`.

### Provider reference pattern

`provider_ref` is a dotted Python import path to a class that implements
`EvaluationSourceProvider`. The runner imports it at runtime via
`importlib.import_module`. Providers belong to the task package (or benchmark
package for future bench-tier regimes); they must not live in callbacks or
Lightning shared utilities.

```toml
[[eval_regimes.regimes]]
regime_id = "arena_diag"
phase_kind = "diag"
provider_ref = "ehc_sn.tasks.arena.providers.ArenaReplayDiagnosticProvider"

[eval_regimes.regimes.provider_settings]
dataset_path = "data/processed/arena/default/v1"
split        = "val"
batch_size   = 4
n_cases      = 0

[eval_regimes.regimes.schedule]
every_n_epochs = 5
every_n_steps  = 0
max_batches    = 10

[eval_regimes.regimes.trace_request]
enabled    = false
trace_keys = []
```

Current task-owned provider modules:

| Task      | Module                             |
| --------- | ---------------------------------- |
| Arena     | `ehc_sn.tasks.arena.providers`     |
| MazeHard  | `ehc_sn.tasks.mazehard.providers`  |
| Dungeon   | `ehc_sn.tasks.dungeon.providers`   |
| Countwalk | `ehc_sn.tasks.countwalk.providers` |

### Entry-point wiring

Each training entry point exposes an optional `eval_regimes` field on its
`RunArguments`:

```python
eval_regimes: Optional[EvaluationRegimesCallbackSettings] = Field(
    default=None,
    description="Named evaluation regime settings.",
)
```

The callback is appended **before** `FiguresCallback` in the `callbacks_list`
so that regime artifacts are available when figure callbacks run.

### Namespace rules

`phase_kind` selects the leading namespace tier for regime-owned metrics.
Canonical namespace ownership and write-boundary rules live in
`spec/spec-model-interfaces.md`.
