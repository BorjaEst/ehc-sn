---
title: Figures Design Contract
description: Domain kernel for ehp-sn figures — views, builders, renderers, registry, export, and diagnostics
---

# Figures Design Contract (`ehp_sn.figures`)

> A deterministic visualization layer over already-computed data. The
> figures package converts typed analysis results into renderable
> Matplotlib figures without owning evaluation, aggregation, experiment
> tracking, or artifact discovery.

The core rule:

> **`evaluation`** determines _what happened_; **`analysis`** computes
> _derived representations_; **`figures`** decides _how those
> representations are visualized_.

---

## 1. Architectural position

```
models / tasks
     ↓
evaluation              traces, predictions, benchmark metrics
     ↓
analysis                scientific computation (grid scores, place fields, RSA, projections)
     ↓
figures ◄────────────── consumes typed analysis results, produces visualizations
     ↓
reports / notebooks / CLI / MLflow artifact logging
```

The figures package is **one-way downstream** of analysis. It never imports
from `evaluation`, `training`, `controllers`, `objectives`, or `tasks`.

Allowed imports:

- `analysis.contracts` — shared analysis result types consumed by builders;
- `contracts` — shared domain types owned outside figures
  (`SpatialExtent`, `ValueRange` from `ehp_sn.contracts`);
- `numpy` — numerical manipulation for display-only transforms;
- `matplotlib` — primary rendering backend.

Conditional imports (lazy, inside optional adapters):

- `seaborn` — optional helper for statistical distribution plots;
- `plotly` + `kaleido` — optional interactive backend (see §18).

Must not import:

- `evaluation` — evaluation protocol types, runners, recipes;
- `training` — model checkpoint loaders, training loops, callbacks;
- `controllers` — rollouts, step functions, bridge outputs;
- `objectives` — loss functions, objective terms;
- `tasks` — task definitions, runtime execution;
- `models` — model implementations, modules, parameters;
- `traces.trace_tree` — raw `TraceTree` traversal (except in optional
  convenience paths deprecated in favour of analysis-prepared views);
- `mlflow` — experiment tracking client state (only imported lazily by
  the separate `integrations/mlflow/` package, never by `figures` core);
- `torch` — GPU tensors (all inputs must be CPU-resident).

---

## 2. Target data flow

```
TraceTree / evaluation artifacts
       │
       ▼
analysis extraction + computation
       │  - loads trace data
       │  - validates
       │  - computes grid scores, place fields, RSA, projections
       │  - aggregates across cases
       ▼
typed analysis result          (lives in analysis/contracts/)
       │  e.g. MECGridAnalysis, HPCPlaceAnalysis
       │
       ▼
figure view builder            (lives in figures domain module, e.g. spatial/mec.py)
       │  - selects units by score
       │  - orders for display
       │  - constructs labels
       │  - builds legend entries
       │  - derives extents
       ▼
FigureView                     (lives in figures domain module)
       │  e.g. MECGridMetricsView, HPCPlaceMetricsView
       │
       ▼
figures renderer               (lives in figures domain module)
       │  - creates axes
       │  - draws heatmaps, trajectories, timelines
       │  - applies colormaps
       │  - renders legends and colorbars
       ▼
FigureResult
       │  { figure: Figure, axes: Mapping[str, Axes], provenance: FigureProvenance }
       │
   ┌───┼───────────┐
   ▼   ▼           ▼
 save  MLflow    notebook
```

### 2.1 Three distinct objects

```
raw trace              stored as TraceTree, artifact files
    ≠
analysis result        scientific product: computed metrics, provenances
    ≠
figure view            presentation contract: selected samples, labels, extents
```

The separation avoids trace-schema coupling, prevents scientific computation
from leaking into visualisation, makes figures testable in isolation, and
allows the same analysis results to feed notebooks, tables, reports, and
multiple visualisations.

---

## 3. Responsibility boundaries

### 3.1 `analysis` — owns scientific meaning

Examples:

- computing rate maps from location‑mean activations;
- computing gridness scores from autocorrelograms;
- computing place‑field statistics (spatial information, sparsity);
- computing content–structure RSA;
- computing observation tuning curves;
- computing PCA/UMAP projections of latent states;
- aggregating metrics across episodes;
- computing confidence intervals;
- computing unit‑quality metrics (grid scores, spatial information, etc.).

Analysis may optionally produce scientifically defined rankings when ranking
is itself an analysis output (e.g. a formal ranking statistic). Ordinary
display ordering — "top 12 by precomputed grid score" — belongs to
`figures.builders`.

The current code already places the pure kernels in `analysis/`
(e.g. `compute_gridness()`, `compute_rate_map_stats()`,
`compute_content_structure_rsa()`). The invocation of those
kernels must happen **before** the figures package receives data.

### 3.2 `figures.builders` — owns presentational preparation only

A builder transforms an analysis result into a figure view. It may:

- select already‑scored units (e.g. "top 12 by grid score");
- sort units for display order;
- choose visible episodes;
- decide panel ordering;
- convert numeric IDs to human‑readable labels;
- build categorical legend entries from observation‑ID mappings;
- derive plotting extents from wall masks or arena dimensions;
- clip values for display contrast (not for statistical correctness);
- construct `CategoricalLegend`, `ColorStrip`, or `ContinuousScale` specs.

It must **not**:

- compute gridness, place‑field quality, RSA, or any other scientific quantity;
- load raw traces or access artifact paths;
- call `compute_*` functions from `analysis/`.

### 3.3 `figures.views` — owns figure input contracts

A view is an immutable dataclass containing:

- already‑computed values;
- labels and colour annotations;
- masks and coordinates;
- selected sample indices;
- visual grouping hints;
- optional pre‑computed display ranges.

A view must **not** contain:

- lazy artifact loaders or file paths;
- `TraceTree` objects;
- checkpoint references;
- MLflow clients;
- analysis callbacks or function references;
- arbitrary trace keys;
- GPU tensors — all arrays are CPU‑resident `numpy.ndarray`;
- mutable configuration dictionaries.

### 3.4 `figures.renderers` — owns visual encoding

A renderer draws one view onto Matplotlib axes. It may:

- create a `Figure` and `Axes` when none are supplied;
- draw into a caller‑supplied axes bundle;
- apply colormaps and normalisation;
- place legends and colorbars;
- set titles, labels, and tick marks;
- arrange multi‑panel mosaics.

It must **not**:

- compute any scientific quantity;
- save files (delegated to `figures.export`);
- modify global Matplotlib state (`plt.rcParams`, `sns.set_theme`);
- accept raw traces, artifact paths, or MLflow run IDs.

### 3.5 `figures.registry` — maps stable identifiers to figure capabilities

The registry owns the mapping from `FigureId` → `FigureDefinition`. It
must **not** know trace keys, artifact formats, or analysis internals.

### 3.6 `figures.export` — owns explicit file output

Responsible for:

- saving to PNG, SVG, PDF via `Figure.savefig()`.

Tracker‑specific logging (MLflow, TensorBoard, W&B) lives in the separate
`integrations/` package so that `figures` core remains tracker‑independent
(see §12.2–§12.3 and §18). Those integration adapters accept a
`FigureResult` but are not part of `ehp_sn.figures`.

### 3.7 `figures.diagnostics` — owns post‑render checks

Runs on a completed `FigureResult`:

- text artist overlap;
- artist outside figure canvas;
- clipped data images;
- inconsistent panel dimensions;
- missing labels or titles.

### 3.8 `integrations/` (separate package) — owns tracker logging

MLflow, TensorBoard, and W&B adapters live outside `figures` so that
`figures` core remains tracker‑independent. Each adapter accepts a
`FigureResult` and performs lazy imports of its tracker dependency.

---

## 4. Core contracts

### 4.1 Figure identifier

```python
import re

_FIGURE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

@dataclass(frozen=True, order=True)
class FigureId:
    value: str

    def __post_init__(self) -> None:
        if not _FIGURE_ID_PATTERN.fullmatch(self.value):
            raise ValueError(
                f"Invalid figure id {self.value!r}; expected snake_case."
            )

    def __str__(self) -> str:
        return self.value
```

Stable IDs:

| Constant                    | Value                         |
| --------------------------- | ----------------------------- |
| `LEC_PIPELINE`              | `"lec_pipeline"`              |
| `MEC_GRID_METRICS`          | `"mec_grid_metrics"`          |
| `HPC_PLACE_METRICS`         | `"hpc_place_metrics"`         |
| `ARENA_TASK_LAYOUT`         | `"arena_task_layout"`         |
| `TEM_PREDICTION_OVERLAY`    | `"tem_prediction_overlay"`    |
| `MAZEHARD_SOLUTION_OVERLAY` | `"mazehard_solution_overlay"` |
| `PFC_LATENT_DYNAMICS`       | `"pfc_latent_dynamics"`       |
| `HALTING_TIMELINE`          | `"halting_timeline"`          |

Do not use a Python `Enum` if external experiments may register additional
figures.

### 4.2 Figure style

`FigureStyle` controls purely visual appearance. It must not contain
scientific thresholds, figure‑specific display parameters, output DPI,
or transparency settings (those belong to `FigureExportRequest`, §12).

```python
@dataclass(frozen=True)
class FigureStyle:
    profile: Literal["paper", "notebook", "presentation"] = "paper"
    font_scale: float = 1.0
    line_width: float = 1.25
    marker_size: float = 4.0
    constrained_layout: bool = True

    @property
    def rc_params(self) -> dict[str, object]:
        """Return Matplotlib rcParams dict for this style."""
        ...


# Named profiles
PAPER_STYLE = FigureStyle(profile="paper")
NOTEBOOK_STYLE = FigureStyle(profile="notebook")
PRESENTATION_STYLE = FigureStyle(
    profile="presentation",
    font_scale=1.4,
    line_width=1.8,
)
```

An optional `canvas_dpi` field may be added to `FigureStyle` if notebook
display quality genuinely requires it, but output DPI is owned by
`FigureExportRequest`.

Domain‑specific and figure‑specific options (e.g. `max_units`,
`selection`, `sort_order`) belong in a figure‑specific spec
dataclass, not in `FigureStyle`.

### 4.3 Style application

Renderers accept `style: FigureStyle` directly. There is no separate
`RenderContext` object — the canonical renderer signature is:

```python
Renderer = Callable[[ViewT, FigureStyle], FigureResult]
```

Style is applied locally inside each renderer using a context manager:

```python
def plot_halting_timeline(
    view: HaltingTimelineView,
    *,
    ax: Axes | None = None,
    style: FigureStyle = PAPER_STYLE,
) -> FigureResult:
    with matplotlib.rc_context(style.rc_params):
        ...
```

No global `plt.style.use(...)`, `sns.set_theme()`, or `mpl.rcParams.update(...)`
call may appear in any library module inside `figures`.

### 4.4 Figure definition (generic)

```python
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

ViewT = TypeVar("ViewT")

Renderer = Callable[[ViewT, FigureStyle], FigureResult]

@dataclass(frozen=True)
class FigureDefinition(Generic[ViewT]):
    id: FigureId
    title: str
    description: str
    view_type: type[ViewT]
    renderer: Renderer[ViewT]
    default_style: FigureStyle = PAPER_STYLE
    tags: frozenset[str] = frozenset()
```

At runtime the registry stores `FigureDefinition[object]` — the generic
parameter provides documentation and static‑checking intent but cannot
guarantee full type compatibility at lookup time (§6.3).

Usage:

```python
FigureDefinition(
    id=FigureId("mec_grid_metrics"),
    title="MEC grid metrics",
    description="Rate maps, autocorrelograms, and grid statistics.",
    view_type=MECGridMetricsView,
    renderer=render_mec_grid_metrics,
    default_style=PAPER_STYLE,
    tags=frozenset({"mec", "spatial", "diagnostics"}),
)
```

### 4.5 Figure request

```python
@dataclass(frozen=True)
class FigureRequest(Generic[ViewT]):
    figure: FigureId
    view: ViewT
    style: FigureStyle | None = None
```

Must **not** contain:

```
run_id
artifact_path
checkpoint
trace_keys
split
recipe
model
device
```

Those belong to application‑level orchestration (see §8).

### 4.6 Figure result

```python
from matplotlib.axes import Axes
from matplotlib.figure import Figure

@dataclass(frozen=True)
class FigureProvenance:
    """Reproducibility record for one rendered figure."""
    figure_id: "FigureId"
    view_type: str
    selected_ids: tuple[int | str, ...] = ()
    selection_rule: str | None = None
    random_seed: int | None = None
    style_profile: str | None = None


@dataclass(frozen=True)
class FigureResult:
    figure: Figure
    axes: Mapping[str, Axes]
    provenance: FigureProvenance
    metadata: Mapping[str, object] = field(default_factory=dict)

    def axis(self, name: str = "main") -> Axes:
        try:
            return self.axes[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown axis {name!r}; available axes: {tuple(self.axes)}"
            ) from exc
```

For repeated panels (e.g. per‑unit tiles in a mosaic), use stable names:

```python
{
    "rate_map/0": ax0,
    "autocorrelation/0": ax1,
    "rate_map/1": ax2,
    "autocorrelation/1": ax3,
}
```

### 4.7 Errors

```python
class FigureError(Exception):
    """Base exception for figure operations."""

class UnknownFigureError(FigureError):
    pass

class FigureViewError(FigureError):
    pass

class FigureViewTypeError(FigureViewError):
    pass

class FigureShapeError(FigureViewError):
    pass

class FigureLayoutError(FigureError):
    pass

class FigureExportError(FigureError):
    pass

class MissingFigureDependencyError(FigureError):
    """Raised when an optional dependency required by a specific figure
    or adapter is not installed (see §18)."""
```

### 4.8 Common value objects

Shared domain types live in `ehp_sn.contracts`, not inside `figures`,
so that `analysis` can import them without reversing the dependency
direction:

```python
# In ehp_sn.contracts.spatial
@dataclass(frozen=True)
class SpatialExtent:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

# In ehp_sn.contracts.values
@dataclass(frozen=True)
class ValueRange:
    minimum: float
    maximum: float
```

Figure‑specific annotation types remain in `figures`:

```python
# In ehp_sn.figures (was _contracts.py)
@dataclass(frozen=True)
class CategoricalLegend:
    entries: list[tuple[str, str]]
    ncol: int = 3
    title: str | None = None
    label_stride: int = 1
    compact: bool = False

@dataclass(frozen=True)
class ColorStrip:
    entries: list[tuple[str, str]]
    label_every: int | None = None
    title: str | None = None
    rows: int = 1

@dataclass(frozen=True)
@dataclass
class ContinuousScale:
    mappable: ScalarMappable | None = None
    label: str = ""
    vmin: float = 0.0
    vmax: float = 1.0
    ticks: list[float] | None = None
```

`PreparedRateMap` (from the current `_contracts.py`) straddles the boundary
between analysis and presentation. If it contains only display values
(masked arrays, colormap extents, smoothing applied for rendering), rename
it to `RateMapImageView` and keep it in `figures`. If it contains the raw
rate map, occupancy, and smoothing performed as a scientific step, move it
to `analysis.contracts.spatial` as `SpatialRateMap`.

---

## 5. Package structure

For the current repository size (≈8 canonical figures), a domain‑oriented
structure keeps each feature vertically cohesive without premature
horizontal layering:

```
src/ehp_sn/figures/
├── __init__.py                  # public API exports
├── api.py                      # render_figure(), get_figure(), list_figures()
├── contracts.py                # FigureId, FigureDefinition, FigureRequest,
│                               #   FigureResult, FigureProvenance
├── errors.py                   # FigureError hierarchy
├── registry.py                 # FigureRegistry
├── registration.py             # build_builtin_figure_registry()
├── styles.py                   # FigureStyle, PAPER_STYLE, etc.
├── export.py                   # save_figure(), FigureExportRequest
├── diagnostics.py              # DiagnosticEngine + checks (was core/diagnostics.py)
│
├── spatial/                    # arena, MEC, HPC figures
│   ├── __init__.py
│   ├── common.py                # annotation types, primitives shared by spatial figures
│   ├── arena.py
│   ├── mec.py
│   └── hpc.py
│
├── representations/            # sensory / latent representations
│   ├── __init__.py
│   ├── lec.py
│   └── pfc.py
│
├── predictions/                # prediction overlays
│   ├── __init__.py
│   ├── tem.py
│   └── mazehard.py
│
└── deliberation/               # reasoning-process figures
    ├── __init__.py
    ├── halting.py
    └── routebind.py
```

Each domain module may contain its view, builder, and renderer together:

```python
# spatial/mec.py

@dataclass(frozen=True)
class MECGridMetricsView:
    ...

@dataclass(frozen=True)
class MECGridMetricsViewSpec:
    ...

def build_mec_grid_metrics_view(
    analysis: MECGridAnalysis,
    *,
    spec: MECGridMetricsViewSpec = MECGridMetricsViewSpec(),
) -> MECGridMetricsView:
    ...

@dataclass(frozen=True)
class MECGridMetricsAxes:
    rate_maps: tuple[Axes, ...]
    autocorrelations: tuple[Axes, ...]
    score_distribution: Axes | None = None

def plot_mec_grid_metrics(
    view: MECGridMetricsView,
    *,
    axes: MECGridMetricsAxes | None = None,
    style: FigureStyle = PAPER_STYLE,
) -> FigureResult:
    ...
```

Separate `views/`, `builders/`, and `renderers/` directories should be
introduced only once there are enough implementations to justify horizontal
layering. "One file per figure" is a starting convention, not an invariant.

### 5.1 Mapping from current layout

The existing `plots/`, `renders/`, annotation contracts, `sinks.py`,
and `DiagnosticEngine` map directly:

| Current location                                 | Target location                                |
| ------------------------------------------------ | ---------------------------------------------- |
| `plots/*.py` (heatmaps, trajectories)            | `spatial/common.py` or inline                  |
| `renders/*.py` (task‑specific visual encoding)   | domain module or `spatial/common.py`           |
| `_contracts.py` (annotations, `PreparedRateMap`) | `contracts.py` or `spatial/common.py`          |
| `sinks.py`                                       | `export.py`                                    |
| `core/diagnostics.py`                            | `diagnostics.py`                               |
| `core/base.py`, `core/panels.py`                 | domain module or internal `_layout.py`         |
| `selectors/` (per‑task)                          | split into `analysis/runners/` + domain module |

---

## 6. Registry design

### 6.1 `FigureRegistry`

```python
class FigureRegistry:
    def register(self, definition: FigureDefinition[object]) -> None: ...

    def get(self, figure_id: FigureId | str) -> FigureDefinition[object]: ...

    def list(
        self,
        *,
        tags: frozenset[str] | None = None,
    ) -> tuple[FigureDefinition[object], ...]: ...

    def validate(self) -> None: ...
```

### 6.2 Registration

Use **explicit construction** in a composition root, not decorators with
import‑time side effects:

```python
def build_builtin_figure_registry() -> FigureRegistry:
    registry = FigureRegistry()

    _register_lec_figures(registry)
    _register_mec_figures(registry)
    _register_hpc_figures(registry)
    _register_task_figures(registry)
    _register_prediction_figures(registry)
    _register_deliberation_figures(registry)

    registry.validate()
    return registry
```

### 6.3 Registry validation

Enforces:

- unique figure IDs;
- valid `FigureId` values;
- renderer is callable;
- renderer has an introspectable positional signature (where Python
  introspection permits — decorators, overloads, and postponed annotations
  may prevent full verification at registration time);
- `view_type` is a class (not an instance);
- `default_style` is immutable (`frozen=True`);
- non‑empty `title` and `description`;
- deterministic iteration order.

Runtime rendering validates the actual view instance via `isinstance()`.
Static compatibility between renderer and view type is checked by type
checkers and tests, not guaranteed by runtime registry validation alone.

---

## 7. Renderer implementation

### 7.1 General pattern

All renderers accept `style: FigureStyle` (not a `RenderContext`) and
return `FigureResult`:

```python
def plot_mec_grid_metrics(
    view: MECGridMetricsView,
    *,
    axes: MECGridMetricsAxes | None = None,
    style: FigureStyle = PAPER_STYLE,
) -> FigureResult:
    with matplotlib.rc_context(style.rc_params):
        ...
```

### 7.2 Caller-supplied axes

#### Single‑panel renderer

```python
def plot_halting_timeline(
    view: HaltingTimelineView,
    *,
    ax: Axes | None = None,
    style: FigureStyle = PAPER_STYLE,
) -> FigureResult:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # ... draw on ax ...

    return FigureResult(
        figure=fig,
        axes={"main": ax},
    )
```

#### Multi‑panel renderer — optional axes bundle

```python
@dataclass(frozen=True)
class MECGridMetricsAxes:
    rate_maps: tuple[Axes, ...]
    autocorrelations: tuple[Axes, ...]
    score_distribution: Axes | None = None

def plot_mec_grid_metrics(
    view: MECGridMetricsView,
    *,
    axes: MECGridMetricsAxes | None = None,
    style: FigureStyle = PAPER_STYLE,
) -> FigureResult:
    if axes is None:
        # template creates its own layout
        ...
    else:
        # render into supplied axes
        ...
```

Do not force a single `ax` onto inherently multi‑panel figures.

### 7.3 Rules

- when `ax is None`, create a new `Figure` and `Axes`;
- when `ax` is supplied, draw into it without creating;
- always return a `FigureResult` with a `FigureProvenance`;
- never call `plt.show()`;
- never save implicitly;
- never modify global Matplotlib state (`plt.rcParams`, `sns.set_theme`);
- use `with matplotlib.rc_context(style.rc_params):` for style application;
- accept `Axes`, not the lower‑level `FigureCanvas`.

---

## 8. Two API levels

### 8.1 Direct typed API (for notebooks, tests, custom layouts)

```python
from ehp_sn.analysis import compute_mec_grid_analysis
from ehp_sn.figures.mec import (
    MECGridMetricsViewSpec,
    build_mec_grid_metrics_view,
    plot_mec_grid_metrics,
)

analysis = compute_mec_grid_analysis(trace)

view = build_mec_grid_metrics_view(
    analysis,
    spec=MECGridMetricsViewSpec(max_units=12),
)

result = plot_mec_grid_metrics(view, style=PAPER_STYLE)
result.figure
```

### 8.2 Registry API (for declarative recipes, automation)

```python
from ehp_sn.figures import FigureRequest, render_figure

result = render_figure(
    FigureRequest(
        figure=FigureId("mec_grid_metrics"),
        view=view,
        style=PAPER_STYLE,
    )
)
```

The direct API is the canonical implementation. The registry API dispatches
to it — do not implement separate rendering logic for each path.

```python
def render_figure(
    request: FigureRequest[object],
    *,
    registry: FigureRegistry | None = None,
) -> FigureResult:
    active_registry = registry or get_default_registry()
    definition = active_registry.get(request.figure)

    if not isinstance(request.view, definition.view_type):
        raise FigureViewTypeError(
            f"Figure {definition.id!s} expects "
            f"{definition.view_type.__name__}, received "
            f"{type(request.view).__name__}."
        )

    style = request.style or definition.default_style
    return definition.renderer(request.view, style)
```

No artifact loading, analysis dispatch, `RenderContext` construction, or
trace lookup occurs inside `render_figure`.

---

## 9. Public API

```python
from .api import get_figure, list_figures, render_figure
from .contracts import (
    FigureDefinition,
    FigureId,
    FigureProvenance,
    FigureRequest,
    FigureResult,
)
from .errors import (
    FigureError,
    FigureExportError,
    FigureLayoutError,
    FigureShapeError,
    FigureViewError,
    FigureViewTypeError,
    UnknownFigureError,
)
from .export import ExportedFigure, FigureExportRequest, save_figure
from .styles import (
    FigureStyle,
    NOTEBOOK_STYLE,
    PAPER_STYLE,
    PRESENTATION_STYLE,
)

__all__ = [
    "ExportedFigure",
    "FigureDefinition",
    "FigureError",
    "FigureExportError",
    "FigureExportRequest",
    "FigureId",
    "FigureLayoutError",
    "FigureProvenance",
    "FigureRequest",
    "FigureResult",
    "FigureShapeError",
    "FigureStyle",
    "FigureViewError",
    "FigureViewTypeError",
    "NOTEBOOK_STYLE",
    "PAPER_STYLE",
    "PRESENTATION_STYLE",
    "UnknownFigureError",
    "get_figure",
    "list_figures",
    "render_figure",
    "save_figure",
]
```

Domain view types, builders, and renderers are exposed from subpackage
namespaces:

```python
from ehp_sn.figures.spatial.mec import (
    MECGridMetricsAxes,
    MECGridMetricsView,
    MECGridMetricsViewSpec,
    build_mec_grid_metrics_view,
    plot_mec_grid_metrics,
)
```

Do not export all domain types at package root.

---

## 10. Application-level convenience workflows

A single operation that starts from traces may be convenient but must **not**
live in the figures package:

```
ehp_sn.workflows.figures      # recommended
ehp_sn.evaluation.figure_generation
```

Example:

```python
def generate_mec_grid_metrics(
    trace: TraceTree,
    *,
    analysis_spec: MECGridAnalysisSpec,
    view_spec: MECGridMetricsViewSpec,
    style: FigureStyle = PAPER_STYLE,
) -> FigureResult:
    analysis = compute_mec_grid_analysis(trace, spec=analysis_spec)
    view = build_mec_grid_metrics_view(analysis, spec=view_spec)
    return plot_mec_grid_metrics(view, style=style)
```

This preserves convenience without contaminating the figure package.

---

## 11. Validation strategy

Validation occurs at three boundaries.

### 11.1 Analysis contract validation (in `analysis/`)

Checks scientific consistency:

- number of units matches number of scores;
- autocorrelation shapes match rate‑map shapes;
- occupancy is non‑negative;
- projections have 2 or 3 dimensions;
- valid masks match temporal length.

### 11.2 View-builder validation (in domain modules)

Checks presentation requirements:

- selected unit IDs exist in the analysis result;
- requested number of panels is positive;
- explicit selection contains no duplicates;
- labels correspond to selected samples;
- no unsupported layout combination was requested.

### 11.3 Renderer validation (in domain modules)

Checks rendering assumptions:

- arrays are finite where required;
- images are two‑dimensional;
- coordinates have shape `[n, 2]`;
- supplied axes bundle has enough panels;
- style parameters are valid.

---

## 12. Export

### 12.1 File export

```python
@dataclass(frozen=True)
class FigureExportRequest:
    destination: Path
    formats: tuple[Literal["png", "svg", "pdf"], ...] = ("png",)
    dpi: int = 300
    transparent: bool = False
    close_after_save: bool = False


@dataclass(frozen=True)
class ExportedFigure:
    paths: tuple[Path, ...]
    media_types: tuple[str, ...]


def save_figure(
    result: FigureResult,
    request: FigureExportRequest,
) -> ExportedFigure: ...
```

### 12.2 Tracker logging (MLflow, TensorBoard, W&B)

Tracker‑specific logging adapters live in the separate `integrations/`
package, not in `ehp_sn.figures`. This keeps `figures` core
tracker‑independent, as required by the import rules in §1.

```
integrations/
├── mlflow/
│   └── figures.py       # log_figure_to_mlflow(result, artifact_path, ...)
├── tensorboard/
│   └── figures.py       # log_figure_to_tensorboard(result, tag, ...)
└── wandb/
    └── figures.py       # log_figure_to_wandb(result, ...)
```

Each adapter imports its tracker lazily and raises `MissingFigureDependencyError`
if the optional dependency is not installed (see §18).

---

## 13. Testing strategy

### 13.1 Contract tests

```python
def test_mec_figure_returns_expected_axes():
    result = plot_mec_grid_metrics(view)

    assert isinstance(result.figure, Figure)
    assert set(result.axes) >= {"rate_maps", "autocorrelations"}
```

### 13.2 Semantic artist tests

```python
def test_halting_timeline_contains_one_row_per_sample():
    result = plot_halting_timeline(view)
    image = result.axes["timeline"].images[0]

    assert image.get_array().shape[0] == view.active_mask.shape[0]
```

### 13.3 Input validation tests

```python
def test_rejects_mismatched_time_dimensions():
    with pytest.raises(FigureDataShapeError):
        plot_prediction_overlay(invalid_view)
```

### 13.4 Export tests

```python
def test_svg_export(tmp_path):
    result = plot_mec_grid_metrics(view)
    exported = save_figure(
        result,
        FigureExportRequest(
            destination=tmp_path / "figure.svg",
            formats=("svg",),
        ),
    )

    assert exported.paths[0].exists()
    assert exported.paths[0].stat().st_size > 0
```

### 13.5 Image-regression suite (limited)

Use `pytest-mpl` or Matplotlib comparison for a small set of canonical
figures where layout is part of the contract. Keep tolerance explicit and
baselines controlled.

---

## 14. Migration from current selectors

### 14.1 What to split

Each current selector (`figures/selectors/mec.py`, `hpc.py`, `lec.py`, etc.)
combines three responsibilities:

```
current selector
├── trace extraction
├── scientific computation
└── view construction
```

These become:

```
trace extraction
    → analysis runner or analysis adapter

scientific computation
    → analysis (already exists as pure kernels)

view construction (selection, ordering, labelling, extents)
    → domain module in figures (e.g. spatial/mec.py)
```

### 14.2 Example: `figures/selectors/mec.py`

Current:

```
figures/selectors/mec.py
    _compute_mec_grid_metrics()       ← calls compute_gridness() from analysis
    select_mec_grid_metrics()         ← returns MECGridMetricsData
    select_mec_autocorr_mosaic()      ← returns MECAutocorrMosaicData
```

Target:

```
analysis/runners/mec.py
    compute_mec_grid_analysis()       ← calls compute_gridness() internally
                                       ← returns MECGridAnalysis

figures/spatial/mec.py
    MECGridMetricsView                ← typed, immutable, presentation-oriented
    MECAutocorrMosaicView
    build_mec_grid_metrics_view()     ← selects & orders units, builds labels
    build_mec_autocorr_mosaic_view()
    plot_mec_grid_metrics()           ← pure rendering (view → FigureResult)
```

### 14.3 What stays unchanged

- The existing drawing primitives (was `plots/`) — move to `spatial/common.py`
  or inline in domain modules;
- The task‑specific visual encoding functions (was `renders/`) — move to domain
  modules;
- The annotation contracts (`CategoricalLegend`, `ColorStrip`, `ContinuousScale`);
- `DiagnosticEngine` — post‑render validation is correctly separated;
- `sinks.py` → `export.py`.

### 14.4 `PreparedRateMap` classification

`PreparedRateMap` currently straddles analysis and presentation. Resolve as
follows:

- If it represents the raw scientific result (rate map, occupancy,
  Gaussian smoothing performed as a scientific step): rename to
  `SpatialRateMap` and move to `analysis.contracts.spatial`.
- If it represents display‑ready values (masked array, colormap extent,
  smoothing applied for rendering): rename to `RateMapImageView` and
  keep in `figures/spatial/common.py`.

---

## 15. Determinism requirements

A scientific figure must be reproducible from:

```
figure ID
figure view
figure spec (where applicable)
style profile
library versions
random seed, when applicable
```

Avoid:

- randomly selecting units without a seed;
- taking "the first available" run from unordered input;
- dynamically calculating limits from hidden global state;
- current‑time labels;
- environment‑dependent colour assignments;
- relying on dictionary iteration from uncontrolled external data;
- deriving captions from filenames.

Reproducibility metadata is recorded in `FigureProvenance` (see §4.6),
which is a required field of `FigureResult`:

```python
FigureResult(
    figure=fig,
    axes=axes,
    provenance=FigureProvenance(
        figure_id=FigureId("mec_grid_metrics"),
        view_type="MECGridMetricsView",
        selected_ids=selected_units,
        selection_rule="top_grid_score",
        random_seed=42,
        style_profile="paper",
    ),
)
```

---

## 16. Anti-patterns

| Anti‑pattern                                       | Why                                                 | Instead                                  |
| -------------------------------------------------- | --------------------------------------------------- | ---------------------------------------- |
| `plot_*(path: Path)`                               | Couples figures to storage                          | Accept a typed view                      |
| Scientific computation inside renderer             | Makes scores inaccessible except through the figure | Compute before rendering                 |
| Registry definitions owning trace extraction       | Turns registry into implicit workflow engine        | Use `view_type` contracts only           |
| `**kwargs` as main customisation API               | Poor discoverability, weak validation               | Typed spec dataclasses                   |
| `plt.rcParams["font.size"] = 14` in library module | Mutates process‑wide state                          | `with mpl.rc_context(...)`               |
| Saving inside every renderer                       | Prevents composition, introduces side effects       | Return `FigureResult`, export separately |
| Returning only a path                              | Caller loses access to the figure object            | Return `FigureResult`                    |
| Generic `plot_figure(kind, data, **kwargs)`        | Weak contracts, special‑case conditionals           | One function per figure type             |
| `run_id` in `FigureRequest`                        | Ties figures to experiment tracking                 | Keep requests parameter‑free             |

---

## 17. Library dependencies

```
Core:
    matplotlib
    numpy

Useful:
    pandas
    seaborn

Optional:
    plotly + kaleido (interactive extras)

Not necessary as core:
    altair, bokeh, holoviews, panel, dash
```

Recommended dependency groups:

```toml
[project.dependencies]
matplotlib = "..."
numpy = "..."

[project.optional-dependencies]
figures = ["seaborn>=..."]
interactive = ["plotly>=...", "kaleido>=..."]
```

If figures are a required capability, Matplotlib belongs in normal
dependencies rather than an optional `figures` extra.

---

## 18. Optional dependency discipline

Core `figures` modules must import successfully without any optional
dependency installed:

```python
# At module top level — only core imports
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Optional imports — only inside functions that need them
def _try_import_seaborn():
    try:
        import seaborn as sns
        return sns
    except ImportError as exc:
        raise MissingFigureDependencyError(
            "seaborn is required for statistical distribution plots. "
            "Install with: pip install ehp-sn[figures]"
        ) from exc
```

Rules:

- top‑level imports must not trigger `ImportError` for optional packages;
- optional adapters (Plotly, Seaborn helpers not in core path) perform
  local imports;
- missing optional dependencies raise `MissingFigureDependencyError` with
  an actionable message indicating which extra to install;
- registry construction must not eagerly import optional renderers;
- integration adapters (`integrations/mlflow/figures.py`, etc.) each
  lazy‑import their tracker dependency.

```python
class MissingFigureDependencyError(FigureError):
    """Raised when an optional dependency required by a specific figure
    or adapter is not installed."""
```

---

## 19. Summary

### `figures` owns

| Responsibility                                           | Location                   |
| -------------------------------------------------------- | -------------------------- |
| Typed renderer input contracts                           | domain modules             |
| Presentational preparation (selection, ordering, labels) | domain modules (builders)  |
| Visual encoding (axes, colormaps, panels, legends)       | domain modules (renderers) |
| Stable figure ID → definition mapping                    | `registry.py`              |
| Explicit file output (PNG, SVG, PDF)                     | `export.py`                |
| Post‑render layout checks                                | `diagnostics.py`           |

### `figures` does not own

| Non‑responsibility                                  | Owned by                           |
| --------------------------------------------------- | ---------------------------------- |
| Scientific computation (gridness, RSA, projections) | `analysis`                         |
| Trace data extraction                               | `analysis` runners/adapters        |
| Metric aggregation                                  | `analysis`                         |
| Evaluation recipes                                  | `evaluation`                       |
| Experiment tracking (MLflow, TensorBoard, W&B)      | `integrations/` (separate package) |
| Report composition                                  | `reporting` or notebooks           |
| Checkpoint loading                                  | `training`                         |
| Model runtime                                       | `controllers`                      |

### Boundary rule

> `figures` accepts validated, figure‑ready view objects. It does not
> accept raw traces, artifact paths, MLflow runs, checkpoints, or
> arbitrary dictionaries.
