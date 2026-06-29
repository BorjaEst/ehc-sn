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
- When documentation contains conflicting or outdated information, delete the
  stale content rather than adding corrective content alongside it.

## When to Update Documentation

### Trigger Conditions

Update docs when any of these change:

- Training/evaluation entry points (behavior or flags)
- Default configuration and YAML config semantics under `config/`
- Dataset build scripts or dataset-loading behavior
- Dependencies or Python version requirements
- Expected outputs, metrics, or logging that users rely on

## Documentation Update Rules

### README.md

Update when installation steps, dependencies, run commands, config examples,
defaults, datasets, or build steps change.

### Code Examples

Verify snippets and commands still run with current code. Prefer small,
copy-pastable examples that match current CLI/config names.

### Config Documentation

When config keys change, update YAML snippets and explain new defaults.

### CHANGELOG.md

Add an entry for user-visible changes (features, fixes, behavior). Call out
breaking changes.

### Breaking Changes

Document what changed, who is affected, and minimal migration steps.

## Verification Checklist

- Commands in README/docs match the current CLI/config names.
- Any referenced config keys exist and defaults are up to date.
- Code snippets are consistent with current imports and entry points.
- Links are valid.
