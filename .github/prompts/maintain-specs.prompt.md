---
agent: "agent"
description: "Audit and minimally maintain canonical specs under spec/ using repo ownership rules and recognized requirements-writing conventions."
argument-hint: "Optional: spec file, section, or maintenance concern"
---

# Maintain Specs

## Mission

Maintain the canonical specs under [spec/](../../spec/) so they stay short, unambiguous, non-redundant, and easy for both humans and LLM agents to retrieve.

Default behavior:

- make the smallest correct edit set;
- update the owning spec instead of restating rules in multiple places;
- stop when ownership, precedence, or source-of-truth is ambiguous.

## Input

Focus
${input:Focus:optional spec file, section, or maintenance concern}

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

1. Determine the owning spec for every rule or taxonomy you touch.
2. If the Focus is empty, start with the required specs plus the process spec and expand only when a concrete issue requires a companion spec.
3. For each candidate change, prefer deletion, reduction, or cross-reference over expansion.
4. Keep required specs short and hot. Move low-frequency detail to the owning companion spec only when the detail is truly required and the move improves clarity.
5. Rewrite policy text as falsifiable statements.
6. Use uppercase `MUST`, `SHOULD`, and `MAY` only when you intend normative force. Otherwise use ordinary prose.
7. Keep one rule, invariant, or claim per bullet or paragraph when practical.
8. Remove duplicate rules, repeated rationale, and repeated path or precedence tables. Keep the owner and reduce the rest to references.
9. Preserve canonical vocabulary from [spec/spec-architecture.md](../../spec/spec-architecture.md) and the ownership model from [spec/spec-process-spec-maintenance.md](../../spec/spec-process-spec-maintenance.md).
10. Replace vague modifiers such as "robust", "clean", "simple", "fast", or "flexible" unless the file defines measurable meaning.
11. Prefer active voice, explicit subjects, and concrete repository nouns over pronouns and paper-local aliases.
12. Do not mass-reflow unrelated text. If you touch prose, remove arbitrary hard wraps only in the touched block and preserve the surrounding file style unless the user explicitly asks for a broader normalization.

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

## Validation

After editing:

1. Recheck that the spec gate still passes.
2. Verify that every touched rule has a single obvious owner.
3. Verify that no touched paragraph silently changed normative force.
4. Verify that companion-detail moves did not leave broken references or duplicate text behind.
5. Run the narrowest available validation for the touched files. If no executable validation exists, inspect the diff for redundancy, ownership drift, and wrapping noise.

## Output Expectations

Return results in this order:

1. `Findings`
   - list the concrete issues found, ordered by severity;
   - include file and section anchors when possible.
2. `Edits`
   - summarize the minimal changes made;
   - if no edit was needed, say so explicitly.
3. `Standards Applied`
   - cite the baselines used from the list above;
   - explain in one sentence each why they mattered for this change.
4. `Validation`
   - report spec-gate status, ownership checks, and any remaining risks.

## Stop Condition

If the task becomes design work, code work, or broad documentation work outside `spec/`, stop after reporting the spec-maintenance findings and ask for a narrower follow-up.
