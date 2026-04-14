# EHC-SN Configuration Patterns Specification

> Non-required companion spec for configuration taxonomy, default loading, and
> composition patterns.

## 1 Scope

This spec documents the preferred configuration shapes used by the repository.
Core ownership and validation rules remain in `spec/spec-architecture.md`,
`spec/spec-requirements.md`, and `spec/spec-standards.md`.

---

## 2 Config Types

| Type                          | Base class                                            | Scope                                                                                     | Lives in                               | Example                       |
| ----------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------- |
| **Component config**          | `pydantic.BaseModel(extra="forbid")`                  | Single-component settings                                                                 | Same file as the owning implementation | `AttractorSettings`           |
| **Model config**              | `pydantic.BaseModel(extra="forbid")`                  | Architecture-only settings                                                                | `models/*.py`                          | `ModelSettings_V1`            |
| **Task config**               | `pydantic.BaseModel(extra="forbid")`                  | Observation/action protocol, geometry, episode semantics, reward or supervision semantics | `tasks/**/*.py`                        | `DungeonNavigationTaskConfig` |
| **Adapter config**            | `pydantic.BaseModel(extra="forbid")`                  | Model-task binding, packing, decoding, objective wiring, rollout binding                  | `adapters/**/*.py`                     | `TEMDungeonAdapterConfig`     |
| **Lightning training config** | `pydantic.BaseModel(extra="forbid")`                  | Optimizer, scheduler, buffers, and trainer-local execution settings                       | `lightning/**/*.py`                    | `ModelConfig_HRM_V1`          |
| **Data config**               | `pydantic.BaseModel(extra="forbid")`                  | Dataset paths, batch size, workers                                                        | `data/*.py`                            | `DatamoduleConfig`            |
| **Experiment settings**       | `pydantic_settings.BaseSettings(cli_parse_args=True)` | Composes the above plus trainer knobs                                                     | `experiments/*.py`                     | `RunArguments`                |

Rules:

- Architectural dimensions use `frozen=True` when they must not change after
  construction.
- Model configs describe architecture only.
- Task and adapter configs hold task semantics and binding semantics.

---

## 3 Static Defaults

TOML files under `config/` provide default values. Entry points load defaults
first, then allow CLI values to override them.

```python
defaults = tomllib.load(Path("config/defaults_ehc.toml").open("rb"))
settings = RunArguments(**defaults)
```

---

## 4 Leaf Composition Pattern

Experiment settings are structured as a flat tree of leaf configs. Downstream
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
