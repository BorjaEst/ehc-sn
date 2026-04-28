---
agent: "Coordinator"
description: "Route factual documentation synchronization through the coordinator and delegate only if it qualifies for docs-sync."
argument-hint: "Behavior sources, docs files, scope, constraints"
---

# Coordinate Docs Sync

## Mission

Treat this request as a candidate for docs-sync.
Use it only for factual synchronization to already-decided behavior.
Do not delegate speculative docs or unresolved product changes.

## Inputs

Underlying Change
${input:UnderlyingChange}

Behavior Sources
${input:BehaviorSources}

Docs To Update
${input:DocsToUpdate}

Audience Or Doc Type
${input:AudienceOrDocType:reference, how-to, tutorial, or explanation}

Hard Constraints
${input:HardConstraints:none if none}

Execution Mode
${input:ExecutionMode:emit-packets-only or dispatch-if-safe}

## Workflow

1. Validate the docs-sync contract exactly.
2. If the contract is complete, produce one docs-sync packet.
3. If the underlying behavior is unresolved, keep the task in the smart lane or block for clarification.

## Output Expectations

Return the Coordinator output format exactly.
