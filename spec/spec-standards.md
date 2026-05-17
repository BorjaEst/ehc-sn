# EHC-SN Standards Specification

> Canonical source of truth for coding, testing, documentation, and repository
> artifact standards.

This file owns repository-wide engineering conventions only. Architecture
boundaries, benchmark semantics, and spec-governance workflow are owned by
other specs and are referenced here rather than restated.

## 1 Code Style

### 1.1 Formatting

- **PEP 8** naming and layout conventions.
- **Black** formatter: `line-length = 80`, `target-version = ["py312"]`.
- **isort**: `profile = "black"`, `line_length = 80`,
  `known_first_party = ["ehc_sn"]`.
- `pyproject.toml` is the executable source of truth for tool settings. If a
  setting in this file conflicts with `pyproject.toml`, update this file in the
  same change or reduce it to a policy-only statement.
- Lines that Black cannot format cleanly (e.g., long function signatures) may
  use `# fmt: skip` or `# fmt: off` / `# fmt: on` sparingly.

### 1.2 Type Hints

- Public APIs (functions, methods, class attributes) **must** have type hints.
- Non-trivial internal functions should have type hints.
- Use `typing` constructs (`Optional`, `Sequence`, `Mapping`, `List`,
  `Literal`, `TypeAlias`) matching the style already established in
  `ehc_sn/types.py`.
- Tensor shape conventions must be documented in docstrings, not encoded in
  type hints (PyTorch tensors are not shape-typed).

### 1.3 Docstrings

- Follow **PEP 257**.
- Public functions, classes, and modules must have docstrings.
- Docstrings should document: purpose, arguments, return values, shape
  conventions (for tensor-bearing APIs), and any non-obvious side effects.
- Use comments for non-obvious intent, invariants, or trade-offs. Do not
  narrate the obvious.

### 1.4 Naming

- **Modules**: lowercase with underscores (`path_integration.py`).
- **Classes**: PascalCase (`AttractorNetwork`, `ModelSettingsV1`).
- **Functions/methods**: lowercase with underscores (`sample_diag_gaussian`).
- **Constants**: UPPER_SNAKE_CASE.
- **Type aliases**: PascalCase (following existing pattern: `MultiScaleCode`,
  `AbstractLocation`, `WalkBatch`).
- **Brain-region abbreviations**: Use established abbreviations (LEC, MEC, HPC,
  PFC) in class names and module names. Spell out in docstrings on first use.

---

## 2 Module Structure

- Each component directory has an `__init__.py` that exports its public API
  via `__all__` or explicit imports.
- Internal helpers are prefixed with `_` (single underscore).
- Private modules (not part of the component's public API) are prefixed `_`.
- One class per file is preferred for significant classes (>50 lines). Small
  related classes may share a file.

---

## 3 Configuration Standards

- Configuration taxonomy, entry-point setting shapes, static defaults, and
  composition rules are owned by `spec/spec-configuration-patterns.md`.
- **Validation**: Use Pydantic validators (`@field_validator`) for non-trivial
  constraints. Fail fast with actionable error messages.
- Executable-surface placement and thin-entry-point rules are owned by
  `spec/spec-requirements.md`.

---

## 4 Testing Standards

- **Framework**: `pytest` with `--import-mode=importlib`.
- **Location**: All tests under `tests/`. Mirror the `src/ehc_sn/` structure
  where practical (e.g., `tests/test_modules_hpc.py`).
- **Requirements**:
  - New behavior must be accompanied by tests.
  - Tests must be deterministic, minimal, and focused on the specific behavior.
  - Do not fix unrelated failing tests as part of a feature change.
- **Coverage**: No hard coverage target, but critical paths (loss computation,
  attractor dynamics, path integration) should have unit tests.
- **Fixtures**: Use `pytest` fixtures for shared setup (e.g., model configs,
  sample tensors). Avoid heavy fixtures that slow the test suite.

---

## 5 Documentation Standards

### 5.1 Repository Documentation

- If build metadata references a repo-root `README.md`, that file must exist
  and remain non-empty.
- **`docs/`** (outer): Documentation root (`docs_root` in manifest). Contains
  MkDocs configuration (`docs/mkdocs.yml`) and content pages
  (`docs/docs/*.md`).
- **`docs/docs/`**: MkDocs content directory. `index.md` is the landing page;
  `getting-started.md` is the onboarding guide.

### 5.2 Code Documentation

Docstrings are the primary code-level documentation.
Module-level docstrings describe purpose and key classes. Tensor shape
conventions must be documented wherever tensors are created, transformed,
or consumed.

### 5.3 Specs as Documentation

- Canonical specs (`spec/*.md`) are living documents. They must be updated
  when the architecture, requirements, or standards change.
- Spec ownership, conflict resolution, and required/companion routing are owned
  by `spec/spec-process-spec-maintenance.md` and `spec/spec-manifest.toml`.
- Specs are not duplicated into `docs/`. If the MkDocs site needs to reference
  specs, it should link to the `spec/` files.
- Changes to top-level packages under `src/ehc_sn/` must update the component
  taxonomy and dependency rules in `spec/spec-architecture.md` in the same change.
- Changes to dependency declarations in `pyproject.toml` must update the
  dependency inventory in `spec/spec-requirements.md` in the same change.

---

## 6 Artifact Standards

- Plans use `spec/spec-manifest.toml [canonical_paths.plans_root]`, Markdown,
  and the sections Goal, Scope, Steps, and Acceptance Criteria.
- Plan details use `spec/spec-manifest.toml [canonical_paths.plan_details_root]`
  and Markdown.
- Change records use
  `spec/spec-manifest.toml [canonical_paths.plan_changes_root]`, Markdown, and
  the sections What Changed, Why, Files Affected, and Testing.

Canonical artifact roots and spec precedence are defined in
`spec/spec-manifest.toml`. Do not duplicate those path or precedence tables in
other specs.

---

## 7 Logging Standards

- Use the repository's TensorBoard logger wrapper (`ehc_sn.logging`) for
  training metrics and scalar logging.
- Use Python's `logging` module for runtime/debug output. Obtain loggers via
  `logging.getLogger(__name__)`.
- **Never** log secrets, credentials, tokens, or sensitive data.
- Log levels: `DEBUG` for trace-level detail, `INFO` for routine milestones,
  `WARNING` for recoverable issues, `ERROR` for failures.

---

## 8 Visualization Standards

- Figures use the **registry pattern** defined in `figures/registry.py`.
  New figure types must be registered via `FigureSpec` and the `REGISTRY`.
- Publication-ready output via **SciencePlots** (style) and **pub-ready-plots**
  (layout).
- Public figure modules should keep low-level drawing in `figures/plots/` and
  reserve figure classes for data selection, layout, and panel orchestration.
- Panel methods on `BaseFigureTemplate` subclasses accept an `Axes` and mutate
  it in place; shared colorbars must use the component's declared panel/colorbar
  conventions rather than ad hoc figure-level state.

---

## 9 Experiment Reporting Standards

- Canonical benchmark reporting semantics are owned by
  `spec/spec-benchmark-suite.md`.
- Repository-level reports and docs must not restate benchmark outcomes in a way
  that drops benchmark id, corpus/split identity, seed treatment, or major
  evaluation caveats.
