## Specs And Governance

The `spec/` directory holds the canonical project contracts for architecture,
requirements, standards, data, benchmarks, and runtime interfaces.

The docs site links to those specs instead of duplicating them in prose.

## Canonical Source of Truth

- `spec/spec-manifest.toml` is the primary contract for required specs, topic
  scopes, and canonical docs paths.
- The required spec set is intentionally limited to:
  - `spec/spec-manifest.toml`
  - `spec/spec-architecture.md`
  - `spec/spec-requirements.md`
  - `spec/spec-standards.md`
- Topic-specific companion specs are included by the manifest when relevant.

## Companion Specs

- `spec/spec-data-contracts.md`
- `spec/spec-benchmark-suite.md`
- `spec/spec-benchmark-configuration-contracts.md`
- `spec/spec-configuration-patterns.md`
- `spec/spec-model-interfaces.md`
- `spec/spec-controller-runtime-contracts.md`
- `spec/spec-process-spec-maintenance.md`

## Precedence

When multiple specs overlap, precedence is declared in
`spec/spec-manifest.toml` under `precedence.order`.

## Docs Checklist for New Work

For any new feature, public interface, or user-visible behavior change:

1. Review `spec/spec-manifest.toml` to identify the relevant required and topic
   specs.
2. Update the applicable spec files in `spec/` before or alongside code changes.
3. Add or update README guidance if installation, usage, or configuration changed.
4. Add or update docs site pages under `docs/docs/` for workflows, benchmarks,
   model families, or evaluation changes.
5. Add or update API docs if new public modules, classes, or entrypoints appear.
6. Add a `CHANGELOG.md` entry for user-facing changes when appropriate.

## Change Discipline

- Architecture-affecting changes should update `spec/spec-architecture.md`.
- Dependency, compatibility, or runtime contract changes should update
  `spec/spec-requirements.md`.
- Style, docs, and process standards changes should update
  `spec/spec-standards.md`.

## Maintenance and Review

Documentation quality is part of repo hygiene. Review docs with every major
feature or refactor and at least quarterly to keep guidance current.

- Treat `docs/docs/` pages as living references, not one-time artifacts.
- Update docs and `CHANGELOG.md` together for user-facing changes.
- Add architecture visuals or workflows when a concept benefits from a diagram.
- Use `mkdocs build --strict` in CI to guard against broken docs.
