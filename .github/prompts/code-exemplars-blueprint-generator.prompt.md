---
description: "Scan a repository and generate exemplars.md with high-quality code examples grouped by a chosen categorization."
agent: "SE: Tech Writer"
tools: ["codebase", "search", "edit/editFiles", "web/fetch"]
---

# Code Exemplars Blueprint Generator

use the architecture taxonomy from `spec/spec-architecture.md` when categorizing or describing exemplars.

## Inputs

- `${input:PROJECT_TYPE}`: `Auto-detect|.NET|Java|JavaScript|TypeScript|React|Angular|Python|Other` (default: `Auto-detect`)
- `${input:SCAN_DEPTH}`: `Basic|Standard|Comprehensive` (default: `Standard`)
- `${input:INCLUDE_CODE_SNIPPETS}`: `true|false` (default: `false`)
- `${input:CATEGORIZATION}`: `Pattern Type|Architecture Layer|File Type` (default: `Pattern Type`)
- `${input:MAX_EXAMPLES_PER_CATEGORY}`: Integer (default: `3`)
- `${input:INCLUDE_COMMENTS}`: `true|false` (default: `true`)
- `${input:SPEC_MANIFEST_PATH}`: Path (default: `spec/spec-manifest.toml`)

## Workflow

### 0) Strict Spec Preconditions (Block if missing)

- Verify `${input:SPEC_MANIFEST_PATH}` exists.
- Read `${input:SPEC_MANIFEST_PATH}` and verify every file listed under `required.files` exists.
- If any are missing, STOP and output exactly:
  - `Blocking: missing required specs: <comma-separated list of missing paths>`

When preconditions pass, treat these as canonical sources of truth:

- `spec/spec-architecture.md` (architecture vocabulary and boundaries)
- `spec/spec-requirements.md` (repo-level requirements/constraints)
- `spec/spec-standards.md` (artifact and documentation standards)

### 1) Codebase Analysis

- If `${input:PROJECT_TYPE}` is `Auto-detect`, detect primary languages/frameworks by scanning file extensions and configuration files.
- Otherwise, focus analysis primarily on `${input:PROJECT_TYPE}` files.
- Only reference files that exist in the repository. Do not invent files or paths.

### 2) Exemplar Selection Criteria

Select files/regions that demonstrate:

- Readable structure and clear naming
- Appropriate validation and error handling
- Separation of concerns and single responsibility
- Representative patterns used across the repo

### 3) Category Guidance

Primary categories MUST come from the architecture taxonomy defined in `spec/spec-architecture.md`.

- Use those taxonomy categories as the top-level structure in `exemplars.md`.
- Do not introduce an alternate architecture framework (e.g., controllers/services/repositories)
  unless the spec explicitly defines it.

Within each spec-defined category, you may add language/framework-specific subcategories that are
grounded in the actual codebase.

### 4) Cross-Cutting Exemplars

Within each spec-defined category, call out cross-cutting patterns when exemplary:

- Error handling and validation
- Logging/telemetry
- Reproducibility and determinism
- Performance-sensitive code paths

## Output Expectations

Create `exemplars.md` at the repository root with:

1. An introduction describing the purpose of the exemplar set
2. A table of contents
3. Sections organized by `${input:CATEGORIZATION}`
4. Up to `${input:MAX_EXAMPLES_PER_CATEGORY}` exemplars per category

For each exemplar, include:

- File path (relative)
- Pattern/component type
- Why it is exemplary

If `${input:INCLUDE_COMMENTS}` is `true`, also include 2-5 concise bullets describing key implementation details.

If `${input:INCLUDE_CODE_SNIPPETS}` is `true`, include a small snippet (prefer 10-30 lines) and ensure it is copied exactly from the repository.

If `${input:SCAN_DEPTH}` is `Comprehensive`, add a short final section:

- Consistency patterns observed
- Architecture observations
- Conventions to follow
- Anti-patterns to avoid (only if confirmed in the codebase)
