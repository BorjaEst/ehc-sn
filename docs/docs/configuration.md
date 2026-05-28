## Configuration

This page summarizes repository configuration patterns from
spec/spec-configuration-patterns.md.

## Configuration Types

- Component config: local owning module settings.
- Model config: architecture-only settings.
- Task config: observation/action/reward protocol settings.
- Adapter config: model-task binding settings.
- Lightning training config: optimizer, scheduler, runtime settings.
- Data config: paths and dataloader settings.
- Entry-point settings: composition layer under scripts/training and
  scripts/benchmarks.

## Validation Rules

- Prefer Pydantic models with extra="forbid".
- Keep architecture settings separated from task semantics.
- Keep adapter semantics in adapters, not in models.

## Defaults Loading

Canonical defaults are TOML files under config/ and are loaded by each
entrypoint.

CLI arguments are parsed with pydantic-settings and override defaults.

## Training Config Overrides

Training scripts support environment variable configuration path overrides:

- TEM_V1_CONFIGURATION_PATH
- TEM_V2_CONFIGURATION_PATH
- HRM_V1_CONFIGURATION_PATH
- HRM_V2_CONFIGURATION_PATH
- EHC_V1_CONFIGURATION_PATH

## Benchmark Config

Benchmark wrappers use defaults under config/benchmarks and support CLI
overrides for track, manifest, recipe, and output parameters.

## Related Specs

- ../../spec/spec-configuration-patterns.md
- ../../spec/spec-requirements.md
- ../../spec/spec-benchmark-configuration-contracts.md
