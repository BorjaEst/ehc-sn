---
agent: "Coordinator"
description: "Route a small local code change through the coordinator and delegate only if it qualifies for local-implementer."
argument-hint: "Objective, files or symbols, expected behavior, validation, docs impact"
---

# Coordinate Local Change

## Mission

Treat this request as a candidate for local-implementer.
Do not delegate if any implementation-contract field is missing.
Only add a separate docs-sync slice if docs impact is explicit and independent.

## Inputs

Objective
${input:Objective}

Allowed Files Or Symbols
${input:AllowedFilesOrSymbols}

Expected Behavior
${input:ExpectedBehavior}

Validation Command
${input:ValidationCommand}

Docs Impact
${input:DocsImpact:none or files to sync}

Hard Constraints
${input:HardConstraints:none if none}

Execution Mode
${input:ExecutionMode:emit-packets-only or dispatch-if-safe}

## Workflow

1. Validate the local-implementer contract exactly.
2. If the contract is complete, produce one local-implementer packet.
3. If docs impact is explicit and independent, optionally produce one docs-sync packet.
4. If the contract is incomplete, keep the task in the smart lane or block for clarification.

## Output Expectations

Return the Coordinator output format exactly.
