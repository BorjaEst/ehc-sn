---
name: "Curate Python Format"
description: "Curate Python formatting with Black-compatible readability improvements and minimal fmt pragmas."
agent: "Python Default Style Curator"
argument-hint: "Target Python file or folder, optional max file count"
---

# Curate Python Format

## Mission

Apply curated Python formatting to the requested scope while preserving Black as the canonical formatter.

## Inputs

Target scope
${input:targetScope:Python file or folder}

Max files
${input:maxFiles:10}

## Workflow

1. Pass the spec gate.
2. Limit work to the target scope.
3. If the scope would touch more than the allowed file budget, stop and ask for confirmation instead of widening silently.
4. Prefer trailing commas and narrow formatting pragmas over broader formatting churn.
5. Do not touch docstrings, type hints, exports, or repo-local banner comments unless the user explicitly asks.
6. Validate by running Black on the touched files.

## Output Expectations

Return a short execution report with:

- files changed
- trailing commas added
- fmt pragmas added or removed
- Black validation result
- any residual formatting concerns
