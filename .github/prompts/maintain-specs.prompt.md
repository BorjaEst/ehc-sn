---
agent: "agent"
description: "Audit and minimally maintain canonical specs under spec/ using repo ownership rules and recognized requirements-writing conventions."
argument-hint: "Optional: spec file, section, or maintenance concern"
---

# Maintain Specs

## Mission

Maintain the canonical specs under [spec/](../../spec/) so they stay short, unambiguous, non-redundant, and easy for both humans and LLM agents to retrieve.
Treat the task as remediation work, not a one-pass review.
Do not stop after listing findings.
End clean only when a final rescan finds no actionable remaining issues in scope.

Default behavior:

- make the smallest correct edit set per iteration;
- update the owning spec instead of restating rules in multiple places;
- continue iterating until every actionable finding in scope is fixed or explicitly blocked;
- stop only when the run is clean or a blocker prevents further correct edits.

## Input

Focus
${input:Focus:optional spec file, section, or maintenance concern}

Scope Override
${input:ScopeOverride:optional, one of implicated or all-specs}

## Always-Read Sources

1. Pass the spec gate from [spec/spec-manifest.toml](../../spec/spec-manifest.toml).
2. Read every file listed in `required.files`.
3. Because this prompt is for spec maintenance, also read [spec/spec-process-spec-maintenance.md](../../spec/spec-process-spec-maintenance.md).
4. Read companion specs from `spec/spec-manifest.toml [collection.topic_files]` only when the Focus or the touched text requires them.

## Authoritative Maintenance Baselines

Apply these baselines explicitly and cite only the ones that materially affected the edit:

1. BCP 14: RFC 2119 (Bradner, 1997) and RFC 8174 (Leiba, 2017), normative requirement language.
   Why: `MUST`, `SHOULD`, and `MAY` need stable meanings, and RFC 8174 clarifies that only uppercase forms carry the normative interpretation.

2. ISO/IEC/IEEE 29148:2018, requirements quality.
   Why: good requirements are necessary, unambiguous, singular, consistent, feasible, and verifiable; those properties make specs easier to review, test, and retrieve.

3. EARS, Easy Approach to Requirements Syntax (Mavin et al., 2009), when a rule is event-driven, state-dependent, or conditional.
   Why: controlled requirement patterns reduce ambiguous conditionals and hidden assumptions.

4. Single Source of Truth and DRY conventions.
   Why: one owner per rule prevents drift, conflicting instructions, and retrieval noise.

5. Semantic line-break convention for prose maintenance.
   Why: avoid arbitrary hard wrapping; keep one sentence or semantic clause per line when editing touched prose so diffs stay small and sentence boundaries stay clear.

## Maintenance Workflow

1. Determine the working scope.
   - Default to `implicated`: the focused spec files plus any owning specs they reference.
   - Use `all-specs` only when the Focus explicitly requests a repository-wide maintenance pass.
2. Inspect the current git diff for files under `spec/` before editing.
   - Use the diff to identify recent churn, moved ownership, and likely contradiction zones.
   - Do not treat the diff as the only source of truth; the current full file contents remain authoritative.
3. Determine the owning spec for every rule or taxonomy you touch.
4. If the Focus is empty, start with the required specs plus the process spec and expand only when a concrete issue requires a companion spec.
5. Build a remediation ledger with one row per finding and one status: `fixed`, `remaining`, `blocked`, or `not-applicable`.
6. For each candidate change, prefer deletion, reduction, or cross-reference over expansion.
7. Keep required specs short and hot. Move low-frequency detail to the owning companion spec only when the detail is truly required and the move improves clarity.
8. Rewrite policy text as falsifiable statements.
9. Use uppercase `MUST`, `SHOULD`, and `MAY` only when you intend normative force. Otherwise use ordinary prose.
10. Keep one rule, invariant, or claim per bullet or paragraph when practical.
11. Remove duplicate rules, repeated rationale, and repeated path or precedence tables. Keep the owner and reduce the rest to references.
12. Preserve canonical vocabulary from [spec/spec-architecture.md](../../spec/spec-architecture.md) and the ownership model from [spec/spec-process-spec-maintenance.md](../../spec/spec-process-spec-maintenance.md).
13. Replace vague modifiers such as "robust", "clean", "simple", "fast", or "flexible" unless the file defines measurable meaning.
14. Prefer active voice, explicit subjects, and concrete repository nouns over pronouns and paper-local aliases.
15. Do not mass-reflow unrelated text. If you touch prose, remove arbitrary hard wraps only in the touched block and preserve the surrounding file style unless the user explicitly asks for a broader normalization.
16. After each edit batch, rescan the current full contents of the touched files and update the remediation ledger.
17. Continue the loop `audit -> edit -> validate -> rescan` until no actionable `remaining` findings exist in scope or a concrete blocker remains.

## Completion Contract

- A run is `clean` only when the final rescan finds no actionable `remaining` findings in scope.
- A run is `partial` when at least one actionable finding remains but no blocking conflict prevents the run from continuing.
- A run is `blocked` when unresolved ownership, precedence, or source-of-truth conflicts prevent further correct edits.
- Do not report success when unresolved findings still exist unless they are explicitly marked `blocked` with a reason.
- If a finding remains because it is outside the chosen scope, mark it `not-applicable` and say why.

## Conflict Rules

Stop and surface a blocker instead of editing through it when:

- a required spec is missing;
- two specs appear to own the same rule and the precedence order does not resolve the conflict cleanly;
- the requested change would force duplication instead of updating the owning spec;
- the request conflicts with [spec/spec-manifest.toml](../../spec/spec-manifest.toml), [spec/spec-architecture.md](../../spec/spec-architecture.md), [spec/spec-requirements.md](../../spec/spec-requirements.md), or [spec/spec-standards.md](../../spec/spec-standards.md).

## Edit Rules

- Stay inside `spec/` unless the maintenance task explicitly requires a synchronized update elsewhere.
- Make minimal edits with local reasoning; avoid style-only churn.
- If you introduce or move a rule, update the non-owning location to reference the owner.
- Do not invent new canonical taxonomies, precedence tables, or duplicate governance text when an owning spec already defines them.
- Do not claim compliance with a convention unless the edited text actually reflects it.
- Every changed hunk in `spec/` must map to one remediation-ledger finding or one validation repair.

## Validation

After editing:

1. Recheck that the spec gate still passes.
2. Verify that every touched rule has a single obvious owner.
3. Verify that no touched paragraph silently changed normative force.
4. Verify that companion-detail moves did not leave broken references or duplicate text behind.
5. Re-read the full current contents of each touched spec and any directly implicated owning spec, not just the edited hunk.
6. Run the narrowest available validation for the touched files. If no executable validation exists, inspect the diff for redundancy, ownership drift, wrapping noise, and unexplained hunks.
7. Use the post-edit diff to reconcile the run:
   - every changed hunk must correspond to `fixed` or `blocked` ledger items;
   - unexplained hunks are defects until explained or reverted in the same change.

## Output Expectations

Before the first edit, provide a short Preflight block.
At the end, return results in this exact order:

### Preflight

- Scope: <implicated or all-specs>
- Status: clean | partial | blocked
- Status evidence: <file, section, diff hunk, or command from this run>
- Primary owner(s): <owning spec files or explicit blocker>
- Local hypothesis: <one sentence>
- First validation: <check, command, or none>

### Completion

- Run Status: clean | partial | blocked
- Fixed Findings: <resolved items, ordered by severity>
- Remaining Findings: <unresolved actionable items, or none>
- Blocked Findings: <blocked items with concrete reason, or none>
- Files Reviewed But Unchanged: <files reviewed with no edit, or none>
- Changes: <brief summary mapped to the fixed findings>
- Standards Applied: <only the baselines that materially affected the run, with one-sentence why>
- Validation And Diff Reconciliation: <spec-gate result, ownership checks, full-file reread result, diff reconciliation, and remaining risks>

## Stop Condition

If the task becomes design work, code work, or broad documentation work outside `spec/`, stop after reporting the spec-maintenance findings and ask for a narrower follow-up.
