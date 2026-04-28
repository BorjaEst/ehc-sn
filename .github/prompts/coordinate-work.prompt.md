---
agent: "Coordinator"
description: "Route a task through the coordinator and delegate only when a worker contract is fully satisfied."
argument-hint: "Task, anchors, decided behavior, validation, docs impact, hard constraints"
---

# Coordinate Work

## Mission

Route this request through Coordinator using the strict worker contracts.
Prefer the smart lane over unsafe delegation.
If a worker contract is incomplete, block instead of guessing.

## Inputs

Task
${input:Task}

Anchors
${input:Anchors}

Behavior Already Decided
${input:BehaviorAlreadyDecided}

Validation Target
${input:ValidationTarget}

Docs Impact
${input:DocsImpact:none or files to sync}

Hard Constraints
${input:HardConstraints:none if none}

Execution Mode
${input:ExecutionMode:emit-packets-only or dispatch-if-safe}

## Workflow

1. Reduce the request to one sentence.
2. Check whether the task stays in the smart lane or qualifies for a worker contract.
3. Delegate only if all required contract fields are present.
4. If Execution Mode is emit-packets-only, return worker packets without dispatching them.
5. If Execution Mode is dispatch-if-safe, delegate only the qualifying slices and keep the rest in the smart lane.

## Output Expectations

Return the Coordinator output format exactly.
