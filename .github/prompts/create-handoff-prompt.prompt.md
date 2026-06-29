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
You are taking over one bounded implementation task in /home/borja/ehp-sn.

Task
<one short paragraph; or "validation only" when Gap to target is none.>

Current status
<2-4 lines: what was verified, what is missing, verdict. Name file, symbol, test, or command used.>

Gap to target
<remaining delta; "none - validation only" when target already reached.>

Primary anchor
<file, symbol, failing test, or command>

Expected behavior
<concrete expected outcome>

Constraints
<scope, architecture, repo rules, or placeholders>

Owning abstraction
<module, class, function, or explicit placeholder>

Abstraction gate
<whether a new wrapper/helper/file/public API is needed; justify if yes>

Workflow
1. Verify spec/spec-manifest.toml and all required spec files exist.
2. Start from Primary anchor; do one cheap freshness check against Current status and Gap to target.
3. Before first edit, identify Owning abstraction; state whether status matches.
4. If freshness matches and Gap is "none - validation only": do not edit, run validation, stop.
5. If freshness differs: report drift, update from verified delta, continue without widening scope.
6. If the cheapest patch would put logic in the wrong layer, move one hop to the owning abstraction.
7. New wrappers, helpers, files, public APIs disallowed by default. Add only with concrete justification.
8. If changes needed: smallest local edit in owning abstraction that closes the Gap.
9. Prefer removal, simplification, and reuse over adding code.
10. After first edit, run narrowest validation before more reading or patching.
11. If validation fails, repair same slice and rerun same validation before expanding scope.
12. Stop when definition of done is met or a concrete blocker remains.

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
- Result: definition of done met or not
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
