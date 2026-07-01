---
title: Report CLI
description: ehp-sn report — build, validate, inspect, render, and export report packages from evaluation artifacts
---

# `ehp-sn report` — Report Package Generation

The `report` CLI transforms **existing evaluation artifacts** into validated,
portable, presentation-ready **report packages**.  It does **not** evaluate
models, rerun inference, select checkpoints, or recompute primary scientific
results.

## Ownership boundary

```
artifacts/evaluation/<task>/<run>/
        │
        ▼
ehp-sn report                          ◄── YOU ARE HERE
        │
        ▼
artifacts/reports/<task>/<run>/
        │
        ▼
notebooks / exports / publications
```

`report` owns:

- Resolution and validation of evaluation artifact sources.
- Extraction of metrics, cases, traces, predictions, and provenance.
- Computation of report-only derived views (e.g. pathway metrics).
- Case selection according to strategy or explicit IDs.
- Figure rendering from a canonical figure registry.
- Assembly of a self-describing Frictionless Data Package.
- Package validation (structural, referential, and semantic).
- Export to delivery formats (HTML, PDF, JSON, CSV).
- All output under `artifacts/reports/`.

`report` must **not** own:

- Model evaluation or inference.
- Checkpoint loading or selection.
- Training state or runtime configuration.
- Primary scientific metric computation (those belong to the evaluation step).
- Dataset generation or task corpus construction.

---

## Stable commands

| Command | Operates on | Purpose |
|---|---|---|
| `build` | artifact source | Build a canonical report package from evaluation artifacts |
| `validate` | report package | Validate structure, references, and semantic content |
| `inspect` | report package | Display metadata, metrics, resources, figures, or provenance |
| `render` | report package | Render or re-render presentation resources (figures) |
| `export` | report package | Produce a delivery artifact for humans or external systems |
| `schema` | — | Print the canonical report package schema |
| `list-profiles` | — | List registered report profiles |

---

## `report build SOURCE`

Build a canonical report package from one or more evaluation artifact sources.

### Synopsis

```bash
ehp-sn report build SOURCE \
    --output artifacts/reports/arena/run-001 \
    [--request configs/reports/arena-diagnostic.toml] \
    [--profile NAME] \
    [--figure ID ...] \
    [--all-figures] \
    [--render/--no-render] \
    [--format png svg pdf] \
    [--overwrite] \
    [--dry-run] \
    [--json]
```

### Arguments

| Argument | Description |
|---|---|
| `SOURCE` | Evaluation artifact path or supported artifact URI (local path, `runs:/<id>/<path>`). Positional. |

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--output`, `-o` | `PATH` | *required* | Destination directory for the report package. |
| `--request`, `-r` | `PATH` | `None` | TOML or YAML report request file (overrides `--profile` defaults). |
| `--profile` | `str` | `None` | Named report profile selecting resources, views, and default figures. |
| `--figure` | `list[str]` | `[]` | Figure ID(s) to render; repeatable. |
| `--all-figures` | flag | `False` | Render every compatible registered figure for the profile. |
| `--render` / `--no-render` | flag | `True` | Enable or disable figure rendering during build. |
| `--format` | `list[str]` | `[]` | Render format(s): `png`, `svg`, `pdf`; repeatable. |
| `--overwrite` | flag | `False` | Replace an existing destination package atomically. |
| `--dry-run` | flag | `False` | Resolve and validate the request without writing files. |
| `--json` | flag | `False` | Emit a machine-readable result to stdout. |

### Behaviour

```
report build SOURCE
│
├─ 1. Resolve artifact source (local or MLflow).
│
├─ 2. Validate source compatibility (task, regime, schema version).
│
├─ 3. Load request parameters:
│      └── request.toml  >  --profile defaults  >  built-in default profile
│
├─ 4. Extract canonical resources:
│      ├── metrics.records.json    structured metric records
│      ├── cases/index.parquet     case metadata
│      ├── cases/selected.json     selected case IDs
│      ├── traces/                 per-case traces (if available)
│      └── predictions/            per-case predictions (if available)
│
├─ 5. Compute derived views (profile-driven):
│      └── e.g. headline.parquet, pathways.parquet
│
├─ 6. Render figures (unless --no-render):
│      ├── figures/index.json
│      └── figures/<figure_id>.<format>
│
├─ 7. Validate completed package (structural + referential + content).
│
├─ 8. Atomically publish to --output.
│      └── datapackage.json  +  provenance.json  +  _SUCCESS
│
└─ 9. Print or emit result.
```

### Defaults

The defaults are conservative:

- **No overwrite**: `--overwrite` must be explicitly set.
- **No rendering without profile**: if no profile and no `--request` are given, the build extracts resources but skips figures unless `--all-figures` or `--figure` is used.
- **Deterministic case selection**: the same source + request always selects the same cases.
- **Validation before commit**: the package is validated before atomic publish.

### Error handling

| Condition | Exit code | Behaviour |
|---|---|---|
| Source not found or incompatible | 3 | Error message on stderr. |
| Output path already exists (no `--overwrite`) | 5 | Error message on stderr. |
| Unknown `--profile` or `--figure` | 6 | Error message on stderr. |
| Partial render failure | 7 | Warnings on stderr; package still published. |

### Examples

**Basic build with default profile:**

```bash
ehp-sn report build artifacts/evaluation/tem-v2-arena \
    --output artifacts/reports/tem-v2-arena
```

**Configuration-driven build with explicit figures:**

```bash
ehp-sn report build artifacts/evaluation/tem-v2-arena \
    --request config/reporting/arena-tem-diagnostic.toml \
    --output artifacts/reports/tem-v2-arena \
    --figure arena-task-layout \
    --figure tem-prediction-overlay \
    --format png pdf \
    --overwrite
```

**Dry run to inspect the plan:**

```bash
ehp-sn report build artifacts/evaluation/tem-v2-arena \
    --output artifacts/reports/tem-v2-arena \
    --profile arena-diagnostic \
    --dry-run \
    --json
```

---

## `report validate REPORT`

Validate an existing report package without rebuilding it.

### Synopsis

```bash
ehp-sn report validate REPORT \
    [--level structure|references|content] \
    [--strict] \
    [--json]
```

### Arguments

| Argument | Description |
|---|---|
| `REPORT` | Report package directory. Positional. |

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--level` | `str` | `content` | Validation depth. |
| `--strict` | flag | `False` | Treat warnings as validation failures. |
| `--json` | flag | `False` | Emit validation results as JSON. |

### Validation levels

| Level | Checks |
|---|---|
| `structure` | Package root exists, `_SUCCESS` sentinel present, `datapackage.json` decodes, schema version is supported, required top-level keys present. |
| `references` | Everything in `structure` + every declared resource and figure file exists on disk, no duplicate resource or figure IDs, no paths escaping the report root. |
| `content` | Everything in `references` + metric records parse correctly, schemas match resource formats, provenance fields are internally consistent, selected case IDs exist in the source artifact, figure-index entries point to real files with correct formats, primary metrics are registered for the declared task, required resources for the profile are present. |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | All checks passed (warnings may exist unless `--strict`). |
| 4 | Validation failure — at least one check failed. |

### Examples

```bash
ehp-sn report validate artifacts/reports/tem-v2-arena --level content
ehp-sn report validate artifacts/reports/tem-v2-arena --strict --json
```

---

## `report inspect REPORT`

Display report metadata, metrics, resources, figures, or provenance without modifying the package.

### Synopsis

```bash
ehp-sn report inspect REPORT \
    [--section summary|metrics|resources|figures|provenance] \
    [--metric NAME] \
    [--json]
```

### Arguments

| Argument | Description |
|---|---|
| `REPORT` | Report package directory. Positional. |

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--section` | `str` | `summary` | Section to display. |
| `--metric` | `str` | `None` | Filter to a specific metric name (only with `--section metrics`). |
| `--json` | flag | `False` | Emit the selected section as JSON. |

### Sections

| Section | Output |
|---|---|
| `summary` | Report root, schema version, task, model family, source URI, metric count, primary metric, case count, figure count, validation status. |
| `metrics` | All metric records, or a single metric with `--metric`. |
| `resources` | Declared resources with name, format, path, and size. |
| `figures` | Figure-index entries with ID, format, path, title, and description. |
| `provenance` | Full provenance record. |

### Example output (`--section summary`)

```
Report:        artifacts/reports/arena/run-001
Schema:        ehp.report.v1
Task:          arena
Model family:  tem-v2
Source:        artifacts/evaluations/arena/run-001
Metrics:       38
Primary:       accuracy_ancestral_revisit = 0.8421
Cases:         12
Figures:       7
Status:        valid
```

### Examples

```bash
ehp-sn report inspect artifacts/reports/tem-v2-arena
ehp-sn report inspect artifacts/reports/tem-v2-arena --section figures
ehp-sn report inspect artifacts/reports/tem-v2-arena --section metrics --metric accuracy_ancestral_revisit --json
```

---

## `report render REPORT`

Render or re-render presentation resources (figures) from an existing canonical report package.  Does **not** recompute metrics or re-extract data.

### Synopsis

```bash
ehp-sn report render REPORT \
    --output PATH \
    [--figure ID ...] \
    [--all] \
    [--format png svg pdf] \
    [--dpi INTEGER] \
    [--overwrite] \
    [--json]
```

### Arguments

| Argument | Description |
|---|---|
| `REPORT` | Source report package. Positional. |

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--output`, `-o` | `PATH` | *required* | Destination for the rendered report package (new or updated). |
| `--figure` | `list[str]` | `[]` | Figure ID(s) to render; repeatable. |
| `--all` | flag | `False` | Render every compatible figure registered in the package profile. |
| `--format` | `list[str]` | `[]` | Render format(s); repeatable. |
| `--dpi` | `int` | `150` | Output DPI for raster formats. |
| `--overwrite` | flag | `False` | Replace an existing destination. |
| `--json` | flag | `False` | Emit machine-readable result to stdout. |

### Behaviour

```
report render REPORT
│
├─ 1. Open and validate the source report package.
│
├─ 2. Resolve figures to render:
│      └── --figure  >  --all  >  profile defaults
│
├─ 3. For each figure:
│      ├── Check compatibility via figure registry.
│      ├── Render to requested formats.
│      └── Stage in temp directory.
│
├─ 4. Build figure index.
│
├─ 5. Atomically publish to --output (new or overwrite).
│      └── figures/index.json + figure files
│
└─ 6. Print or emit result.
```

### Notes

- `render` writes to a **new** output directory by default.  Use `--overwrite` to replace an existing destination.
- Renderer version and rendering parameters are recorded in the output package's provenance.
- Partial failures (some figures fail) produce exit code 7; successfully rendered figures are still written.

### Examples

```bash
ehp-sn report render artifacts/reports/tem-v2-arena \
    --output artifacts/reports/tem-v2-arena-rendered \
    --all

ehp-sn report render artifacts/reports/tem-v2-arena \
    --output artifacts/reports/tem-v2-arena \
    --figure arena-task-layout \
    --format png pdf \
    --overwrite
```

---

## `report export REPORT`

Produce a delivery artifact for humans or external systems.  An export is a **projection** of the canonical report package — never the canonical representation itself.

### Synopsis

```bash
ehp-sn report export REPORT \
    --format html|pdf|json|csv \
    --output PATH \
    [--template NAME] \
    [--title TEXT] \
    [--overwrite]
```

### Arguments

| Argument | Description |
|---|---|
| `REPORT` | Report package directory. Positional. |

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--format` | `str` | *required* | Export format: `html`, `pdf`, `json`, or `csv`. |
| `--output`, `-o` | `PATH` | *required* | Destination export file or directory. |
| `--template` | `str` | `None` | Named export template (format-specific). |
| `--title` | `str` | package name | Document title for HTML/PDF exports. |
| `--overwrite` | flag | `False` | Replace an existing export destination. |

### Format notes

| Format | Output | Description |
|---|---|---|
| `html` | single `.html` file | Self-contained HTML report with embedded figures and tables. |
| `pdf` | single `.pdf` file | Print-ready PDF (requires a LaTeX or headless-browser renderer). |
| `json` | single `.json` file | Machine-readable projection of all report resources. |
| `csv` | directory of `.csv` files | Tabular projections of metrics, cases, and derived views. |

### Examples

```bash
ehp-sn report export artifacts/reports/tem-v2-arena \
    --format html \
    --output exports/arena-run-001.html

ehp-sn report export artifacts/reports/tem-v2-arena \
    --format json \
    --output exports/arena-run-001.json
```

---

## `report schema`

Print the canonical report package schema to stdout.

```bash
ehp-sn report schema
```

Outputs the JSON Schema for the report `datapackage.json` descriptor, including the EHP-specific extension fields (`ehp` namespace, profile, resource kinds, figure index schema).

---

## `report list-profiles`

List registered report profiles.

```bash
ehp-sn report list-profiles
```

```
PROFILE                TASK           MODEL FAMILIES         DEFAULT FIGURES
arena-diagnostic       arena          tem-v1, tem-v2         arena-task-layout, tem-prediction-overlay
mazehard-diagnostic    mazehard       hrm-v1, hrm-v2         mazehard-task-layout, mazehard-prediction-evolution
routebind-diagnostic   routebind      hrm-v1, hrm-v2         routebind-task-layout
goaltrace-diagnostic   goaltrace      hrm-v1, hrm-v2         goaltrace-task-layout
seqmaze-diagnostic     seqmaze        ehc-v1                 seqmaze-task-layout
```

Pass `--json` for machine-readable output.

---

## Canonical report package

Every `report build` produces one canonical package layout:

```
<report-root>/
├── datapackage.json           Frictionless Data Package descriptor
├── provenance.json            Source identity, generation context, renderer version
├── validation.json            Results of post-build validation
├── _SUCCESS                   Empty sentinel (written last, validates atomic completion)
│
├── metrics/
│   ├── records.json           Structured metric records [{metric, value, unit, role, ...}]
│   ├── headline.parquet       Consumer-friendly headline metric DataFrame
│   └── pathways.parquet       Consumer-friendly pathway metric DataFrame
│
├── cases/
│   ├── index.parquet          Case metadata and selection flags
│   └── selected.json          Selected case IDs (deterministic ordering)
│
├── traces/                    Per-case trace artifacts (if requested)
│   └── <case-id>.pkl
│
├── predictions/               Per-case prediction artifacts (if requested)
│   └── <case-id>.pkl
│
├── tables/
│   └── index.json             Derived table catalog
│
├── figures/
│   ├── index.json             Figure catalog [{figure_id, path, format, title, description}]
│   ├── <figure-id>.<png|svg|pdf>
│   └── ...
│
└── derived/                   Profile-driven derived resources
    ├── <derived-name>.parquet
    └── ...
```

### `datapackage.json` descriptor

```json
{
  "profile": "data-package",
  "name": "arena-diagnostic-run-001",
  "resources": [
    {
      "name": "metrics",
      "path": "metrics/records.json",
      "format": "json",
      "mediatype": "application/json",
      "schema": {
        "fields": [
          {"name": "metric", "type": "string"},
          {"name": "value", "type": "number"},
          {"name": "unit", "type": "string"},
          {"name": "higher_is_better", "type": "boolean"},
          {"name": "is_primary", "type": "boolean"},
          {"name": "role", "type": "string"}
        ]
      }
    },
    {
      "name": "headline_metrics",
      "path": "metrics/headline.parquet",
      "format": "parquet",
      "mediatype": "application/vnd.apache.parquet",
      "ehp": {"kind": "derived", "task": "arena"}
    },
    {
      "name": "figures_index",
      "path": "figures/index.json",
      "format": "json",
      "mediatype": "application/json",
      "ehp": {"kind": "figure-index"}
    }
  ],
  "ehp": {
    "schema": "ehp.report.v1",
    "profile": "arena-diagnostic",
    "task": "arena",
    "model_family": "tem-v2"
  }
}
```

---

## Reusable Python API

The Typer commands are thin adapters over a stable Python API defined by `ReportService`.  Consumers (notebooks, CI scripts, library code) should use this API directly:

```python
from ehc_sn.reporting import ReportService, BuildReportRequest

service = ReportService()

result = service.build(
    BuildReportRequest(
        source="artifacts/evaluation/tem-v2-arena",
        output=Path("artifacts/reports/tem-v2-arena"),
        profile="arena-diagnostic",
        render_figures=True,
    )
)

print(result.report_root)     # Path
print(result.schema_version)  # "ehp.report.v1"
print(result.resource_count)  # 14
print(result.figure_count)    # 7
```

### Notebook-facing read model

For interactive analysis, `open_report_view()` returns a typed wrapper:

```python
from ehc_sn.reporting import open_report_view

report = open_report_view("artifacts/reports/tem-v2-arena")

report.summary()
# {'root': ..., 'metric_count': 38, 'figure_count': 7}

report.headline_metrics
# DataFrame with one row of headline metrics

report.pathway_metrics
# DataFrame indexed by pathway (ancestral, retrieved, inference)

report.figure_entry("arena-task-layout", preferred_format="png")
# FigureEntry(path=..., figure_id=..., format="png", title=...)
```

---

## Design rules

1. **`report` consumes artifacts.**  It never evaluates models, runs inference, or selects checkpoints.

2. **One canonical contract.**  Every CLI command operates on the same report package format (`datapackage.json` + resources).  Internal representations (legacy loaders, raw artifact readers) are never exposed through the CLI.

3. **Metric semantics belong to `TaskScoringSpec`, not to hardcoded metric-name conventions.**  Derived views (headline metrics, pathway metrics) are computed via the task scoring registry, not by prefix matching on metric names.

4. **The package is the interface.**  Notebooks and exports consume the package through `open_report_view()` or export tools.  They do not access evaluation artifacts directly.

5. **Separation of extraction and rendering.**  `build` extracts data and produces figures by default, but `--no-render` defers rendering to a separate `render` step.  This accommodates slower or GPU-dependent figure generators.

6. **Atomic writes.**  Every command that writes uses a temp directory + final rename.  Partial failures never leave a corrupted package.

7. **Conservative defaults.**  No overwrite, no implicit deletion, deterministic case selection, validation before commit.

---

## Exit code contract

| Code | Meaning | Commands |
|---|---|---|
| 0 | Success | all |
| 1 | Unexpected internal failure | all |
| 2 | CLI usage error (reserved by Click/Typer) | all |
| 3 | Invalid or incompatible source artifact | `build` |
| 4 | Invalid report package or validation failure | `validate`, `build` |
| 5 | Output conflict (destination exists, no `--overwrite`) | `build`, `render`, `export` |
| 6 | Unsupported schema, task, profile, figure, or export format | `build`, `validate`, `render`, `export` |
| 7 | Partial render/export failure | `render`, `export` |

Machine-readable JSON output uses these error codes:

```json
{
  "status": "error",
  "code": "report.source.incompatible",
  "message": "Source does not contain traces required by profile 'arena-diagnostic'.",
  "details": {
    "source": "artifacts/evaluation/tem-v2-arena",
    "missing_resource": "traces"
  }
}
```

---

## See also

- [Evaluation CLI](evaluate.md) — producing the artifacts that `report build` consumes.
- [Figures documentation](../figures.md) — the figure registry and rendering pipeline.
- [Checkpoints and Artifacts](../checkpoints-and-artifacts.md) — artifact layout and conventions.
- [Configuration](../configuration.md) — report request TOML schema and profile registration.
