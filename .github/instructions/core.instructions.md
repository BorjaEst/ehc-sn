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

When the spec gate passes, treat these as the canonical sources of truth:

- `spec/spec-architecture.md` (architecture vocabulary and boundaries)
- `spec/spec-requirements.md` (repo-level requirements/constraints)
- `spec/spec-standards.md` (artifact and documentation standards)

If the user request conflicts with canonical specs, surface the conflict and ask for a decision before proceeding.

## Principles

Your primary goal is to shorten, simplify, and clarify code. Prefer deletion over addition.

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

Output:
Return only the optimized code and a brief rationale.
