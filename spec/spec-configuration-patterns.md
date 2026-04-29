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
  - Example: `ModelConfig_HRM_V1`.
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
config_path = Path(os.environ.get("EHC_V1_CONFIGURATION_PATH", "config/training.ehc-v1.toml"))
defaults = tomllib.load(config_path.open("rb"))
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
