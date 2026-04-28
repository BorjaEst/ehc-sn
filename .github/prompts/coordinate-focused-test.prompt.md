---
agent: "Coordinator"
description: "Route narrow deterministic test work through the coordinator and delegate only if it qualifies for focused-test-author."
argument-hint: "Behavior, implementation anchors, test scope, validation"
---

# Coordinate Focused Test

## Mission

Treat this request as a candidate for focused-test-author.
Do not delegate if expected behavior is not already fixed.
Do not use this prompt for exploratory debugging or broad test-suite redesign.

## Inputs

Behavior To Lock
${input:BehaviorToLock}

Implementation Anchors
${input:ImplementationAnchors}

Existing Test File
${input:ExistingTestFile:none if new test}

Allowed Production Edits
${input:AllowedProductionEdits:none or tiny test-enabling edit}

Validation Command
${input:ValidationCommand}

Hard Constraints
${input:HardConstraints:none if none}

Execution Mode
${input:ExecutionMode:emit-packets-only or dispatch-if-safe}

## Workflow

1. Validate the focused-test-author contract exactly.
2. If the contract is complete, produce one focused-test-author packet.
3. If behavior is still unsettled, keep the task in the smart lane or block for clarification.

## Output Expectations

Return the Coordinator output format exactly.
