---
agent: "Mentor mode"
description: "Generate a self-contained copy-paste prompt for another agent to implement the current task safely and without drift."
argument-hint: "Objective or target override, optional"
---

# Create Handoff Prompt

## Mission

Generate one ready-to-paste implementation prompt for another coding agent.
Do not implement the task yourself.
Use the current conversation as the primary source of truth for intent, then use the smallest local repo status check needed to avoid stale or misleading instructions.
Before writing the handoff, compare the verified repo state against the target and decide whether the target is already reached or what concrete gap still remains.

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
3. Re-check the current implementation status using the smallest discriminating local read around the active anchor. Base the status on what you verified in the repo during this run, not on the conversation alone. Determine whether the task appears done, partially done, or not done.
4. Compare the verified current repo state against the target and determine the remaining gap, if any. Choose exactly one handoff mode:
   - validation-only handoff if the target is already reached
   - delta-only implementation handoff if something is still missing
5. If any key fact is still unclear, do not invent it. Keep an explicit placeholder in the generated prompt.
6. Generate one self-contained prompt that another agent can execute safely.
7. The generated prompt must instruct the receiving agent to:
   - start from the named anchor,
   - do one cheap freshness check against the provided Current status and Gap to target before editing,
   - distinguish requested intent from sender-verified repo status,
   - identify the owning abstraction before editing and explain why the change belongs there,
   - treat the provided Current status as the sender-owned baseline rather than generating a new initial status summary,
   - if the freshness check matches and the target is already reached, run the provided validation and stop,
   - if the freshness check differs, report the drift briefly and continue from the verified delta instead of restarting broad triage,
   - prefer removal, simplification, and reuse over adding code,
   - treat a new wrapper, helper, file, or public API as disallowed by default unless it has a real justification,
   - if a new wrapper, helper, file, or public API is proposed, justify it as boundary translation, a second real consumer, compatibility surface, or explicit isolation need,
   - if no such justification exists, change the existing owner instead of adding another layer,
   - avoid unrelated cleanup, broad refactors, or scope expansion,
   - run the narrowest available validation immediately after the first substantive edit,
   - stop with a concise done-or-not-done report.
8. Keep the generated prompt concrete and bounded to the current task.

## Output Expectations

Return only one fenced text block containing the handoff prompt.
Make it ready to copy and paste.
Use this exact section structure inside the generated prompt:

```text
You are taking over one bounded implementation task in /home/borja/ehc-sn.

Task
<one short paragraph describing only the remaining delta to target, or "validation only" when Gap to target is none.>

Current status
<2 to 4 lines covering: what was verified to already exist, what is still missing or uncertain, and the verdict: done, partially done, or not done. Name the observed file, symbol, test, or command used for the status check.>

Gap to target
<one short paragraph describing the remaining delta from the verified current state to the target; use "none - validation only" when the target is already reached.>

Primary anchor
<file, symbol, failing test, or command>

Expected behavior
<concrete expected outcome>

Constraints
<scope, architecture, repo rules, or placeholders>

Owning abstraction
<module, class, function, or explicit placeholder>

Abstraction gate
<state whether a new wrapper, helper, file, or public API is needed; if yes, justify it>

Workflow
1. Verify spec/spec-manifest.toml exists and that every required spec file exists before doing any design or code work.
2. Start from the Primary anchor and do one cheap freshness check against the provided Current status and Gap to target. Do not redo broad status discovery unless that check disagrees.
3. Before the first edit, identify the Owning abstraction and state whether the provided status appears to match current repo state.
4. If the freshness check matches and Gap to target is none - validation only, do not edit code. Run the cheapest relevant validation and stop.
5. If the freshness check differs, report the drift briefly, update your local understanding from the verified delta, and continue without widening scope.
6. If the cheapest local patch would place logic in the wrong layer, move one hop to the owning abstraction instead of adding another wrapper.
7. Treat new wrappers, helpers, files, and public APIs as disallowed by default. Only add one if the Abstraction gate has a concrete justification: boundary translation, second real consumer, compatibility surface, or explicit isolation need.
8. If changes are needed, make the smallest local edit in the owning abstraction that closes the provided Gap to target.
9. Prefer removal, simplification, and reuse over adding code.
10. After the first substantive edit, run the narrowest available validation before doing more reading or patching.
11. If validation fails, repair the same slice and rerun the same validation before expanding scope.
12. Stop when the definition of done is met or when a concrete blocker remains.

Validation
<test, command, or explicit placeholder>

Definition of done
<what must be true to stop>

Reporting format
- Status check: matched or drifted
- Status: done or not done
- Changes: short summary or none
- Status evidence: observed file, symbol, test, or command from this run
- Drift: brief mismatch summary or none
- Validation: what ran or why it could not run
- Result: whether the definition of done was met
- Blockers: only if not done
```

## Quality Checks

- Do not perform implementation work.
- Do not return analysis outside the fenced handoff prompt.
- Do not widen scope beyond the current target.
- Sender owns current-status evaluation and gap-to-target computation; the receiving agent owns only a cheap freshness check.
- Do not ask the receiving agent to generate a fresh initial status summary; provide the sender-verified status yourself.
- Make the ownership check and abstraction gate explicit in the generated prompt.
- Prefer explicit placeholders over guessed details.

---

agent: "Mentor mode"
description: "Generate a self-contained copy-paste prompt for another agent to implement the current task safely and without drift."
argument-hint: "Objective or target override, optional"

---

# Create Handoff Prompt

## Mission

Generate one ready-to-paste implementation prompt for another coding agent.
Do not implement the task yourself.
Use the current conversation as the primary source of truth for intent, then use the smallest local repo status check needed to avoid stale or misleading instructions.

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
3. Re-check the current implementation status using the smallest discriminating local read around the active anchor. Base the status on what you verified in the repo during this run, not on the conversation alone. Determine whether the task appears done, partially done, or not done.
4. If any key fact is still unclear, do not invent it. Keep an explicit placeholder in the generated prompt.
5. Generate one self-contained prompt that another agent can execute safely.
6. The generated prompt must instruct the receiving agent to:
   - start from the named anchor,
   - re-check current status before editing,
   - distinguish requested intent from verified current repo status,
   - identify the owning abstraction before editing and explain why the change belongs there,
   - state a short status summary, name the observed file, symbol, test, or command that informed it, and say whether the task may already be done,
   - prefer removal, simplification, and reuse over adding code,
   - treat a new wrapper, helper, file, or public API as disallowed by default unless it has a real justification,
   - if a new wrapper, helper, file, or public API is proposed, justify it as boundary translation, a second real consumer, compatibility surface, or explicit isolation need,
   - if no such justification exists, change the existing owner instead of adding another layer,
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
<2 to 4 lines covering: what was verified to already exist, what is still missing or uncertain, and the verdict: done, partially done, or not done. Name the observed file, symbol, test, or command used for the status check.>

Primary anchor
<file, symbol, failing test, or command>

Expected behavior
<concrete expected outcome>

Constraints
<scope, architecture, repo rules, or placeholders>

Owning abstraction
<module, class, function, or explicit placeholder>

Abstraction gate
<state whether a new wrapper, helper, file, or public API is needed; if yes, justify it>

Workflow
1. Verify spec/spec-manifest.toml exists and that every required spec file exists before doing any design or code work.
2. Start from the Primary anchor and read only enough nearby code to confirm the current control path and status.
3. Before the first edit, identify the Owning abstraction and state whether the task appears already done, partially done, or not done based on what you verified in the repo during this run.
4. If the cheapest local patch would place logic in the wrong layer, move one hop to the owning abstraction instead of adding another wrapper.
5. Treat new wrappers, helpers, files, and public APIs as disallowed by default. Only add one if the Abstraction gate has a concrete justification: boundary translation, second real consumer, compatibility surface, or explicit isolation need.
6. If the task is already done, do not edit code. Generate a validation-only handoff, run the cheapest relevant validation, and report the result.
7. If changes are needed, make the smallest local edit in the owning abstraction that moves the task toward done.
8. Prefer removal, simplification, and reuse over adding code.
9. After the first substantive edit, run the narrowest available validation before doing more reading or patching.
10. If validation fails, repair the same slice and rerun the same validation before expanding scope.
11. Stop when the definition of done is met or when a concrete blocker remains.

Validation
<test, command, or explicit placeholder>

Definition of done
<what must be true to stop>

Reporting format
- Status: done or not done
- Changes: short summary or none
- Status evidence: observed file, symbol, test, or command from this run
- Validation: what ran or why it could not run
- Result: whether the definition of done was met
- Blockers: only if not done
```

## Quality Checks

- Do not perform implementation work.
- Do not return analysis outside the fenced handoff prompt.
- Do not widen scope beyond the current target.
- Make the ownership check and abstraction gate explicit in the generated prompt.
- Prefer explicit placeholders over guessed details.
