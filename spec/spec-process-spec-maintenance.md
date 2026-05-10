---
title: Specification Maintenance and Conflict Resolution
version: 1.0
date_created: 2026-04-24
owner: Repository Maintainers
tags:
  - process
  - spec
  - governance
---

# Specification Maintenance

This file owns spec authoring, conflict resolution, and maintenance rules for
`spec/`.

## 1. Scope

- **Owns**: spec governance, cross-spec conflict handling, and update workflow.
- **Does not own**: architecture, runtime policy, engineering standards, or
  domain-specific semantics.

## 2. Core Rules

- **REQ-001**: Each rule, taxonomy, or contract **MUST** have exactly one owning
  spec.
- **REQ-002**: Non-owning specs **MUST** reference the owning spec rather than
  restating its rule.
- **REQ-003**: Required specs **MUST** stay short and hot. Low-frequency detail
  belongs in companion specs.
- **REQ-004**: When code, config, or docs invalidate a rule, the same change
  **MUST** update the owning spec.
- **REQ-005**: Unresolved spec conflicts **MUST** be surfaced as blocking.
- **REQ-006**: Conflict precedence is owned only by
  `spec/spec-manifest.toml [precedence]`. Losing text **MUST** be corrected or
  reduced to a reference in the same change.
- **REQ-007**: New non-canonical specs **MUST** follow the
  `spec-<purpose>-<slug>.md` naming pattern.

## 3. Writing Rules

- **PAT-001**: Write normative rules as falsifiable statements.
- **PAT-002**: Use `MUST`, `SHOULD`, and `MAY` only for policy.
- **PAT-003**: Avoid ambiguous modifiers unless the file defines the allowed
  behavior.
- **PAT-004**: Prefer repository vocabulary over paper-local aliases; map
  aliases once if needed.
- **PAT-005**: Put rationale and examples after the rule, not inside it.

## 4. Update Triggers

- **REQ-008**: Top-level package taxonomy changes **MUST** update
  `spec/spec-architecture.md`.
- **REQ-009**: Runtime dependency or repo-level policy changes **MUST** update
  `spec/spec-requirements.md`.
- **REQ-010**: Coding, testing, documentation, or artifact convention changes
  **MUST** update `spec/spec-standards.md`.
- **REQ-011**: Data, benchmark, configuration, or interface contract changes
  **MUST** update the owning companion spec.
