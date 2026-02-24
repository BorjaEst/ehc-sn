---
description: "Python coding conventions and guidelines"
applyTo: "**/*.py"
---

# Python Coding Conventions

This file contains Python-specific guidance and is intended to complement the
language-agnostic rules in `.github/instructions/core.coding.instructions.md`.

## Style & Structure

- Follow PEP 8 naming and layout conventions unless the surrounding codebase
  establishes a different local pattern.
- Keep functions small and single-purpose; extract helpers when logic becomes
  hard to read.
- Prefer explicit, readable code over clever one-liners.

## Docstrings & Comments

- Public functions, classes, and modules should have docstrings that follow
  PEP 257.
- Use comments to explain non-obvious intent, invariants, or trade-offs; avoid
  commenting every line or narrating the code.

## Type Hints

- Add type hints for public APIs and for non-trivial internal functions.
- Use `typing` constructs when needed (e.g., `Optional`, `Sequence`, `Mapping`),
  and match the style already used in the repo.

## Errors & Resource Handling

- Validate inputs at boundaries; raise specific exceptions with actionable
  messages.
- Prefer context managers (`with`) for files, locks, and other resources.
- Avoid bare `except:`; catch the narrowest exception type possible.

## Testing

- If the repository already has a test harness, add or update unit tests for
  changed behavior.
- Do not introduce a new test framework or large test scaffolding unless the
  task explicitly requires it.

1. **Ask User**: "Would you like me to generate test scripts for this implementation?"
2. **Provide Summary**: Brief overview of implementation and any caveats
3. **Validate Solution**: Ensure code actually runs and produces expected results
