---
name: "Implement Final Target"
agent: "agent"
description: "Implement one clarified target in the current repo with explicit anti-drift, ownership, and validation gates."
argument-hint: "Paste the Final Target markdown from Clarify Final Target, or select it in the editor"
---

# Implement Final Target

## Mission

Implement exactly one clarified target from [clarify-final-target.prompt.md](./clarify-final-target.prompt.md) in the current repository.
Act on the codebase. Do not generate another prompt, plan, or design document unless the workflow below requires a blocking stop.

## Inputs

Final Target
${input:FinalTarget:paste the exact markdown output from Clarify Final Target}

Selected Context
Use ${selection} if it contains the final target document. If both are present, prefer the more complete version and say which source you used.

Validation Override
${input:ValidationOverride:optional}

## Scope and Preconditions

1. Pass the spec gate and read the required files from [spec/spec-manifest.toml](../../spec/spec-manifest.toml).
2. Treat the Final Target document as the canonical task definition. Use earlier chat only for non-conflicting context.
3. If the Final Target conflicts with the required specs, surface the conflict and stop for a decision. Do not silently choose one.
4. If the Final Target is missing a concrete Primary Anchor, Behavioral Rules, or Acceptance Criteria, stop and ask for a corrected target. Do not invent missing requirements.
5. If the requested work would trigger the tracked-plan policy in [spec/spec-requirements.md](../../spec/spec-requirements.md), stop and report that implementation is blocked until the required plan exists.
6. Keep the task single-purpose. Do not add adjacent cleanup, opportunistic refactors, or unrelated documentation work unless the Final Target explicitly includes them.

## Workflow

1. Parse the Final Target into end state, in-scope work, out-of-scope work, primary anchors, behavioral rules, and acceptance criteria.
2. Start from the Primary Anchor and re-check current repository status using the smallest discriminating local read, failing test, or command. Distinguish clarified intent from verified current status.
3. Identify the owning abstraction that directly controls the behavior. If the anchor is only wiring, forwarding, registration, or a failing test, move one hop to the nearest code that computes, mutates, or decides the behavior.
4. Before the first edit, state one falsifiable local hypothesis, one cheap check that could disconfirm it, and whether the task appears already done, partially done, or not done.
5. If the task is already done, do not edit code. Run the cheapest relevant validation and report whether the acceptance criteria are already satisfied.
6. If changes are needed, make the smallest grounded edit in the owning abstraction.
7. After the first substantive edit, run the narrowest available validation immediately. Use this order:
   - cheapest behavior-scoped or previously failing check
   - narrow test for the touched slice
   - narrow compile, lint, or typecheck
   - diff-only review only if no executable check exists
8. If validation fails but still points to the same slice, repair that same slice and rerun the same validation before expanding scope.
9. If validation falsifies the current hypothesis, move one nearby hop to the code that more directly controls the behavior. Do not reopen broad exploration.
10. Treat new wrappers, helpers, files, and public APIs as disallowed by default. Add one only with a concrete justification: boundary translation, second real consumer, compatibility surface, or explicit isolation need.
11. Apply the deletion and legacy-removal principles from `core.instructions.md`.
12. Implement the general solution. Do not hard-code to the visible tests or add one-off workarounds.
13. Stop as soon as the acceptance criteria are satisfied or a concrete blocker remains.

## Execution Rules

- Do not reinterpret the Final Target into a broader redesign or reopen settled decisions.
- Do not continue searching once you know the owning abstraction and the cheapest check.
- When two approaches both look plausible, choose one; revisit only if validation falsifies it.
- Do not convert the task into a handoff prompt, plan, or speculative todo list.
- Do not create scratch scripts or temp files unless needed for validation; remove them after.
- Stop only at done or blocked.

## Output Expectations

Before the first edit, provide a short Preflight block.
At the end, provide one concise completion report using exactly this section order:

### Preflight

- Target source: <selection or Final Target input>
- Status: done | partially done | not done
- Status evidence: <file, symbol, test, or command from this run>
- Owning abstraction: <module, class, function, or explicit blocker>
- Local hypothesis: <one sentence>
- First validation: <test or command, or none>

### Completion

- Result: done | blocked
- Changes: <brief summary or none>
- Validation: <what ran and the result, or why it could not run>
- Acceptance criteria: <met or not met, criterion by criterion>
- Blockers: <none or concrete blocker>

## Quality Checks

- Never claim a file, symbol, behavior, or test result you did not verify in this run.
- Prefer explicit placeholders or blocking questions over guesses.
- Keep edits minimal and local.
- If no narrow executable validation exists, say so explicitly.
- Stop immediately once the definition of done is met.
