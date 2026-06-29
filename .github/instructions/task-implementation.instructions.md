---
applyTo: "**/.copilot-tracking/changes/*.md"
description: "Instructions for implementing task plans with progressive tracking and change record - Brought to you by microsoft/edge-ai"
---

# Task Plan Implementation Instructions

Implement your task plan from `.copilot-tracking/plans/**` and `.copilot-tracking/details/**`.
Track progress in `.copilot-tracking/changes/**`.

## Process

### 1. Before starting

Read the complete plan file (scope, objectives, phases, checklist items), the
corresponding changes file, and all files referenced in the plan. Understand the
project structure and conventions.

### 2. For each task, in plan order

a. Read the full details section for the task from `.copilot-tracking/details/**`.
b. Implement with working code that follows workspace patterns, includes error
handling, and meets all task requirements.
c. Validate against the task requirements; fix issues before moving on.
d. Mark the task `[x]` in the plan file.
e. Append to the changes file (Added, Modified, or Removed sections) with
relative file paths and a one-sentence summary.
f. If any change diverges from the plan, call it out in the changes file with
the specific reason.
g. When all tasks in a phase are `[x]`, mark the phase header `[x]`.

### 3. Quality standards

- Follow workspace patterns and `.github/instructions/` conventions.
- Include appropriate error handling and validation.
- Add documentation for complex logic.
- Ensure compatibility with existing systems.

### 4. Completion

All plan tasks `[x]`, all specified files contain working code, all success
criteria verified, no errors remain. After ALL phases are `[x]`, add a Release
Summary section to the changes file with full file inventory and implementation
summary.

## Template Changes File

Create in `.copilot-tracking/changes/` as `YYYYMMDD-task-description-changes.md`.
Include `<!-- markdownlint-disable-file -->` at the top.
Update after EVERY task completion by appending to Added, Modified, or Removed sections.

Replace `{{ }}` with appropriate values.

<!-- <changes-template> -->

```markdown
<!-- markdownlint-disable-file -->

# Release Changes: {{task name}}

**Related Plan**: {{plan-file-name}}
**Implementation Date**: {{YYYY-MM-DD}}

## Summary

{{Brief description of the overall changes made for this release}}

## Changes

### Added

- {{relative-file-path}} - {{one sentence summary of what was implemented}}

### Modified

- {{relative-file-path}} - {{one sentence summary of what was changed}}

### Removed

- {{relative-file-path}} - {{one sentence summary of what was removed}}

## Release Summary

**Total Files Affected**: {{number}}

### Files Created ({{count}})

- {{file-path}} - {{purpose}}

### Files Modified ({{count}})

- {{file-path}} - {{changes-made}}

### Files Removed ({{count}})

- {{file-path}} - {{reason}}

### Dependencies & Infrastructure

- **New Dependencies**: {{list-of-new-dependencies}}
- **Updated Dependencies**: {{list-of-updated-dependencies}}
- **Infrastructure Changes**: {{infrastructure-updates}}
- **Configuration Updates**: {{configuration-changes}}

### Deployment Notes

{{Any specific deployment considerations or steps}}
```

<!-- </changes-template> -->
