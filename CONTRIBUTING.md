# Contributing to ehp-sn

## Scope

This guide covers local setup, coding conventions, testing, and documentation
expectations for contributions to ehp-sn.

## Development Setup

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2. Install project with dev extras

```bash
pip install -e ".[dev]"
```

### 3. Verify install

```bash
python -c "import ehp_sn; print('ehp_sn import OK')"
pytest -q
```

## Coding Standards

- Python style and formatting are defined by Black and isort settings in
  pyproject.toml.
- Use the canonical namespace ehp_sn.
- Keep architecture boundaries aligned with spec/spec-architecture.md.
- Keep runtime and dependency constraints aligned with
  spec/spec-requirements.md.

## Tests

- Test framework: pytest
- Test location: tests/
- Add focused tests for behavior changes and regressions.

Run full tests:

```bash
pytest
```

Run specific tests:

```bash
pytest tests/test_data.py
```

## Documentation Requirements

Documentation updates are required when user-visible behavior changes.

- Review `spec/spec-manifest.toml` for the required spec set and topic file
  collection.
- Use `docs/docs/specs-and-governance.md` for docs governance and the new work
  checklist.
- Use `docs/docs/development.md` for local onboarding, tooling, and runtime
  workflow guidance.
- Update README.md for installation, usage, or command changes.
- Update docs/docs pages for workflow, benchmark, or config changes.
- Add or update API docs for new public entrypoints or module behavior.
- Add CHANGELOG.md entries for user-visible changes.

Relevant standards:

- spec/spec-manifest.toml
- spec/spec-standards.md
- .github/instructions/markdown.instructions.md
- .github/instructions/update-docs.instructions.md

## Pull Request Checklist

Before opening a PR, ensure:

1. Code follows repository style and architecture constraints.
2. Tests pass locally.
3. Documentation is updated for user-visible changes.
4. Changelog entry is added if behavior changed.
5. No unrelated files are modified.

## Commit Guidance

- Keep commits focused and atomic.
- Use clear, imperative commit messages.
- Reference issue IDs when applicable.

## Reporting Issues

Use the issue tracker listed in pyproject project URLs.
Include reproduction steps, expected behavior, actual behavior, and environment
details.
