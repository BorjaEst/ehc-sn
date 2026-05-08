---
agent: "Mentor mode"
description: "Generate a self-contained copy-paste prompt for another agent to implement the current task safely and without drift."
argument-hint: "Objective or target override, optional"
---

# Create Safe Handoff Prompt

## Mission

Generate one ready-to-paste implementation prompt for another coding agent.
Do not implement the task yourself.
Use the current conversation as the primary source of truth, then do the smallest local status check needed to avoid stale or misleading instructions.

## Inputs

Objective Override
${input:ObjectiveOverride:optional if already clear from chat}

Primary Anchor Override
${input:PrimaryAnchorOverride:file, symbol, failing test, or command if needed}

Expected Behavior Override
${input:ExpectedBehaviorOverride:optional if already clear from chat}

Validation Override
${input:ValidationOverride:optional if already clear from chat}

Allowed Scope Override
${input:AllowedScopeOverride:local slice only}

Target Agent
${input:TargetAgent:optional}

Selected Context
Use ${selection} if it is present and relevant. Otherwise rely on the current conversation and any override inputs.

## Workflow

1. Review the current conversation and latest user request. Extract the actual task, active target, constraints, validation path, and definition of done.
2. Pass the spec gate and read the required files from [spec/spec-manifest.toml](../../spec/spec-manifest.toml).
3. Re-check the current implementation status using the smallest local read around the active anchor. Determine whether the task appears done, partially done, or not done.
4. If any key fact is still unclear, do not invent it. Keep an explicit placeholder in the generated prompt.
5. Generate one self-contained prompt that another agent can execute safely.
6. The generated prompt must instruct the receiving agent to:
   - start from the named anchor,
   - re-check current status before editing,
   - state a short status summary and whether the task may already be done,
   - prefer removal, simplification, and reuse over adding code,
   - avoid unrelated cleanup, broad refactors, or scope expansion,
   - run the narrowest available validation immediately after the first substantive edit,
   - stop with a concise done-or-not-done report.
7. Keep the generated prompt concrete and bounded to the current task.

## Output Expectations

Return only one fenced text block containing the handoff prompt.
Make it ready to copy and paste.
Use this exact section structure inside the generated prompt:

```text
You are taking over one bounded implementation task in /home/borja/ehc-sn.

Task
<one short paragraph>

Current status
<2 to 4 lines on what already exists and what is still missing>

Primary anchor
<file, symbol, failing test, or command>

Expected behavior
<concrete expected outcome>

Constraints
<scope, architecture, repo rules, or placeholders>

Workflow
1. Verify spec/spec-manifest.toml exists and that every required spec file exists before doing any design or code work.
2. Start from the Primary anchor and read only enough nearby code to confirm the current control path and status.
3. Before the first edit, state whether the task appears already done, partially done, or not done.
4. If the task is already done, do not edit code. Run the cheapest relevant validation and report the result.
5. If changes are needed, make the smallest local edit that moves the task toward done.
6. Prefer removal, simplification, and reuse over adding code.
7. After the first substantive edit, run the narrowest available validation before doing more reading or patching.
8. If validation fails, repair the same slice and rerun the same validation before expanding scope.
9. Stop when the definition of done is met or when a concrete blocker remains.

Validation
<test, command, or explicit placeholder>

Definition of done
<what must be true to stop>

Reporting format
- Status: done or not done
- Changes: short summary or none
- Validation: what ran or why it could not run
- Result: whether the definition of done was met
- Blockers: only if not done
```

## Quality Checks

- Do not perform implementation work.
- Do not return analysis outside the fenced handoff prompt.
- Do not widen scope beyond the current target.
- Prefer explicit placeholders over guessed details.
