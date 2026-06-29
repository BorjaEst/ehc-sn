---
description: "Generic coding conventions and engineering guidelines (language-agnostic)"
applyTo: "{**/*.py,**/*.pyi,**/*.ipynb,**/*.js,**/*.jsx,**/*.ts,**/*.tsx,**/*.java,**/*.go,**/*.rs,**/*.cs,**/*.c,**/*.cpp,**/*.h,**/*.hpp,**/*.sh,**/*.ps1}"
---

# Core Coding Instructions (Language-Agnostic)

These rules apply to all programming languages in this repository.
Language-specific rules belong in the relevant language instruction file.

## Goals

- Optimize for correctness, clarity, and maintainability.
- Keep behavior changes intentional, documented, and testable.
- Prefer focused, minimal changes within the owning abstraction.
  When removal is the right fix, delete across all affected call sites rather
  than leaving dead code or compatibility shims.

## Implementation Guidelines

- Follow existing project patterns and naming conventions. Study neighboring
  modules before introducing new abstractions, types, or wiring.
- Keep functions and modules single-responsibility; split complex logic into helpers.
- Write concise, idiomatic, and easily understandable code. Avoid unnecessary
  abstraction; choose the simplest thing that works.
- Avoid hidden side effects; prefer explicit inputs/outputs and dependency injection.
- For algorithm-related code, include a brief explanation of the approach.
- Use comments to explain non-obvious intent and trade-offs (avoid narrating the obvious).
- Handle edge cases and failures explicitly.
- Document any non-trivial external dependency or integration point.

## Validation & Error Handling

- Validate inputs at boundaries (CLI args, config, file IO, network).
- Fail fast with actionable error messages.
- Handle errors explicitly; do not silently swallow exceptions or failures.
- Prefer safe defaults; avoid potentially destructive actions unless explicitly requested.

## Testing & Verification

- When adding or changing behavior, add or update tests close to the change.
- Keep tests deterministic, minimal, and focused on the new behavior.
- Do not “fix” unrelated failing tests or formatting unless required by the task.

## Dependencies & Compatibility

- Minimize new dependencies; justify non-trivial additions.
- Keep dependency changes scoped and documented.
- Prefer solutions that work in the repo’s supported runtime/tooling.

## Logging & Observability

- Use the repository’s preferred logging/telemetry mechanism for non-trivial flows.
- Avoid logging secrets, credentials, tokens, or sensitive data.
