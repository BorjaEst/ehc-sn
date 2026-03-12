# EHC-SN Standards Specification

> Canonical source of truth for coding, testing, documentation, and artifact
> standards.

## 1 Code Style

### 1.1 Formatting

- **PEP 8** naming and layout conventions.
- **Black** formatter: `line-length = 110`, `target-version = ["py312"]`.
- **isort**: `profile = "black"`, `line_length = 110`,
  `known_first_party = ["ehc_sn"]`.
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
- **Classes**: PascalCase (`AttractorNetwork`, `ModelSettings_V1`).
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

- **Component configs**: `pydantic.BaseModel(extra="forbid")` with `Field(...)`
  descriptors. Use `frozen=True` on fields that must not change after
  construction (e.g., architectural dimensions).
- **CLI entry points**: `pydantic_settings.BaseSettings(extra="forbid",
cli_parse_args=True)`. CLI source must have highest precedence.
- **Static defaults**: TOML files under `config/`. Currently
  `config/defaults_ehc.toml` (empty — to be populated).
- **Validation**: Use Pydantic validators (`@field_validator`) for non-trivial
  constraints. Fail fast with actionable error messages.

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

- **`README.md`** (repo root): Must be non-empty. Contains project overview,
  installation instructions, and quick-start guide.
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
- Specs are not duplicated into `docs/`. If the MkDocs site needs to reference
  specs, it should link to the `spec/` files.

---

## 6 Artifact Standards

| Artifact       | Location                     | Format   | Required sections                          |
| -------------- | ---------------------------- | -------- | ------------------------------------------ |
| Plans          | `.copilot-tracking/plans/`   | Markdown | Goal, Scope, Steps, Acceptance Criteria    |
| Plan details   | `.copilot-tracking/details/` | Markdown | Linked from parent plan                    |
| Change records | `.copilot-tracking/changes/` | Markdown | What Changed, Why, Files Affected, Testing |

Spec file precedence is defined in `spec/spec-manifest.toml [precedence]` and
is the single source of truth. Do not duplicate the precedence order elsewhere.

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
- Figure sinks: `save_pdf` for file output; interactive `plt.show()` for
  development.
- Axis utilities in `figures/utils/` for consistent subplot layout.
- Public figure modules should keep low-level drawing in `figures/plots/` and
  reserve figure classes for data selection, layout, and panel orchestration.
- Panel methods on `BaseFigureTemplate` subclasses accept an `Axes` and mutate
  it in place; shared colorbars must use the component's declared panel/colorbar
  conventions rather than ad hoc figure-level state.

---
