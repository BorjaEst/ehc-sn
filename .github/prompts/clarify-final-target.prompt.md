---
name: "Clarify Final Target"
agent: "Mentor mode"
description: "Turn a rough implementation or refactor brief into a concrete final target spec with API shape, naming changes, module interactions, and acceptance criteria."
argument-hint: "Paste the rough request or desired change"
---

# Clarify Final Target

## Mission

Transform one rough implementation, refactor, or API-change request into one concrete final target specification with no avoidable ambiguity.
Do not implement code.
Do not produce an implementation plan unless the user explicitly asks for one.

## Input

Rough Request
${input:RoughRequest:paste the rough task, goal, or change request}

## Scope and Preconditions

1. Pass the spec gate and read the required files from [spec/spec-manifest.toml](../../spec/spec-manifest.toml).
2. Treat the Rough Request as the primary source of truth.
3. Read the smallest amount of repo context needed to ground named files, symbols, modules, helpers, tests, or commands.
4. If a mandatory fact cannot be confirmed from the request or nearby sources, ask the minimum blocking questions and stop. Do not guess.
5. Keep the task single-purpose. Do not merge design, implementation, migration, and rollout work into one target unless the request explicitly requires that.

## Workflow

1. Parse the Rough Request into desired outcome, changed surfaces, constraints, and missing facts.
2. Identify the smallest concrete anchors needed to remove ambiguity: file, symbol, API surface, neighboring helper, test, or command.
3. Read only enough local context to confirm ownership, boundaries, and terminology.
4. Rewrite vague goals into explicit end-state language. Replace words like "clean up", "improve", "support", or "refactor" with observable target behavior.
5. Define the exact API target when relevant: public names, inputs and outputs,
   types or shape expectations, validation and error behavior, compatibility.
6. Define the naming-refactor surface when relevant: current names, target names,
   affected files or call sites, explicit non-goals.
7. Define module and helper interactions when relevant: owning module, reused
   helpers, helper additions or removals, boundaries that must remain unchanged,
   wiring changes versus logic changes.
8. State concrete behavioral rules and invariants that must hold after the work is done.
9. Write acceptance criteria that are specific and falsifiable.
10. If ambiguity remains, surface it explicitly as a blocking question or clearly labeled assumption. Never hide ambiguity behind generic prose.

## Output Expectations

Return one markdown document using exactly this section order.
Do not omit sections. If a section is not applicable, say that explicitly.

```markdown
## Final Target

<One short paragraph that states the end state directly and concretely.>

## In Scope

- <Concrete change 1>
- <Concrete change 2>

## Out of Scope

- <Explicit non-goal 1>
- <Explicit non-goal 2>

## Primary Anchors

- <File, symbol, command, or test that most directly owns the change>

## API Target

- Public surface: <new API, changed API, or "No public API change.">
- Inputs: <exact inputs and types/shapes, or "Not applicable.">
- Outputs: <exact outputs and types/shapes, or "Not applicable.">
- Validation and errors: <expected failures, guards, or "Not applicable.">
- Compatibility: <backward-compatibility expectation or "Not applicable.">

## Naming Map

| Current        | Target        | Reason | Affected Surface              |
| -------------- | ------------- | ------ | ----------------------------- |
| <current name> | <target name> | <why>  | <files, imports, tests, docs> |

## Module Interactions

- Owning module: <where the behavior belongs>
- Reused helpers: <existing helpers to keep using, or "None.">
- Helper changes: <new helper, deleted helper, or "None.">
- Cross-module effects: <imports, adapters, callers, tests, docs>
- Boundary rules: <what must not move or be re-owned>

## Behavioral Rules

- <Invariant 1>
- <Invariant 2>

## Acceptance Criteria

- <Criterion 1 that can be checked>
- <Criterion 2 that can be checked>

## Open Questions

- None.
```

## Quality Checks

- Keep the result bounded to one target.
- Prefer concrete nouns, file names, symbol names, and observable behavior over abstract advice.
- Do not invent file paths, APIs, or helpers that were not provided or locally confirmed.
- Use repository terminology that matches the required specs.
- If the request is underspecified, ask concise blocking questions instead of filling gaps with assumptions.
