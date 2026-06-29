---
description: "This are the core instructions that apply to all files in the repository."
applyTo: "**"
---

## Mode Contract (Strict Spec-First)

Before producing any design or code, you MUST:

1. Verify `spec/spec-manifest.toml` exists.
2. Verify every file listed under `required.files` in that manifest exists.
3. If any are missing, STOP and output exactly:

- `Blocking: missing required specs: <comma-separated list of missing paths>`

When the spec gate passes:

- Read only the files in `required.files` by default.
- Treat those required files as the canonical always-read sources of truth.
- Use `spec/spec-manifest.toml [collection.topic_files]` to fetch companion
  specs only when the task needs them.

Topic routing:

- `data`: processed-data format, pipelines, dataset outputs.
- `benchmarks`: benchmark definitions, corpus/protocol rules, benchmark-specific reporting.
- `configuration`: config taxonomy, defaults, and composition patterns.
- `interfaces`: model state/step surfaces and adapter interface patterns.
- `process`: spec authoring, ownership, conflict resolution, and spec maintenance workflow.

If the user request conflicts with canonical specs, surface the conflict and ask for a decision before proceeding.

## Instruction File Precedence

`core.instructions.md` is the single source of truth for engineering principles.
Other instruction files (`core.coding.instructions.md`, `python.instructions.md`,
etc.) elaborate mechanics and must not restate or contradict principles defined
here. If a narrower instruction file appears to conflict, `core.instructions.md`
takes precedence.

## Principles

Your primary goal is to shorten, simplify, and clarify code. Prefer deletion over
addition.

- Remove legacy code when touching an owning abstraction; do not work around it.
- Follow existing repository patterns and contracts. Study neighboring modules
  before introducing new abstractions, types, or wiring.
- For documentation, prefer deleting stale or conflicting content over adding new
  content alongside it.

When optimizing code, follow this process:

1. Identify redundancies or overengineering.
2. Rewrite the code to be shorter and clearer.
3. Ensure correctness and readability is preserved.
4. Provide a minimal explanation of the changes (max 3 sentences).

style:

- Concise and technical.
- No filler language.
- No expanding the code unless needed for correctness.
- Prioritize simplicity over architecture.
