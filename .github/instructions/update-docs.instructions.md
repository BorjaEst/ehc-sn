---
description: "Automatically update README.md and documentation files when application code changes require documentation updates"
applyTo: "{README.md,CHANGELOG.md,CONTRIBUTING.md,docs/**/*.md}"
---

# Update Documentation on Code Change

## Overview

Ensure documentation stays synchronized with code changes by automatically detecting when README.md,
API documentation, configuration guides, and other documentation files need updates based on code
modifications.

## Policy

- When code changes affect user workflows, configuration, or expected outputs, update the relevant documentation in the same change.
- Keep documentation scannable: short sections, bullets, and concrete commands.
- Do not invent dataset folder schemas or API contracts unless a spec defines them.

## When to Update Documentation

### Trigger Conditions

Update docs when any of these change:

- Training/evaluation entry points (`pretrain.py`, `evaluate.py`) behavior or flags
- Default configuration and YAML config semantics under `config/`
- Dataset build scripts under `dataset/` or dataset-loading behavior (`puzzle_dataset.py`)
- Dependencies (`requirements.txt`, Python version requirements)
- Expected outputs/metrics/logging that users rely on

## Documentation Update Rules

### README.md Updates

Update README.md when:

- You add/modify installation steps or required dependencies.
- You change how to run training/evaluation (commands, flags, `torchrun` usage).
- You change configuration examples or defaults.
- You add new datasets or modify dataset build steps.

### Code Examples

- If docs contain code snippets or commands, verify they still run with the updated code.
- Prefer small, copy-pastable examples that match current CLI/config names.

### Config Documentation

- When config keys change, update any referenced YAML snippets and explain new defaults.
- If behavior is subtle, add a short note to README or a dedicated docs page.

### CHANGELOG.md Updates

- Add an entry when changes are user-visible (new features, fixes, behavior changes).
- Call out breaking changes clearly.

### Breaking Changes & Migrations

- If a change breaks existing commands, configs, or expected outputs, document:
  - what changed
  - who is affected
  - the minimal migration steps

## Verification Checklist

- Commands in README/docs match the current CLI/config names.
- Any referenced config keys exist and defaults are up to date.
- Code snippets are consistent with current imports and entry points.
- Links are valid.
