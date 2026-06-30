# Analysis Design Contract

> A deterministic post-evaluation transformation layer that consumes
> evaluation artifacts and produces reproducible, typed, domain-level
> analytical results.

The analysis subsystem owns scientific interpretation — grid scores, place
fields, representational similarity, pathway-conditioned aggregates,
latent-state projections — but owns neither model execution nor visual
rendering. It is the bridge between **what the model produced** (traces,
predictions, metrics) and **what those results mean** in neuroscientific
terms.

---

## 1. Architectural position

```
model/runtime
     ↓
evaluation              produces traces, predictions, benchmark metrics
     ↓
evaluation artifacts    persisted on disk or MLflow
     ↓
analysis ◄───────────── consumes artifacts, produces domain results
     ↓
figures / reports       visual rendering + narrative composition
```

The analysis layer is **one-way downstream** of evaluation. It never
imports from `figures`, `training`, `controllers`, or `objectives`.

Allowed imports:

- `contracts` — shared artifact keys, dependency vocabulary;
- `contracts.artifacts` — shared artifact schemas (`ArtifactKey`, `ProducedArtifact`);
- `contracts.diagnostics` — shared diagnostic types;
- `types` — shared domain types;
- `numpy`, `scipy`, `xarray` — numerical kernels;
- optional `torch`/`cupy` behind explicit optional adapters (analysis must not
  load model checkpoints or depend on training loops).

Zarr and filesystem access are restricted to the **executor** layer via
`ArtifactReader`/`ArtifactWriter` protocols. Pure kernels do not import Zarr.

---

## 2. Lifecycle: definition → request → plan → result → artifact

The entire analysis subsystem follows one lifecycle that the rest of the
repository can adopt as a convention:

```
Definition         — registered reusable capability
     ↓
Request            — user-selected concrete invocation
     ↓
Plan               — fully resolved and validated executable graph
     ↓
Result             — in-memory semantic output (payload + provenance)
     ↓
Artifact           — persisted result with storage references
```

| Stage          | Type                      | Purpose                                                       | Pure? |
| -------------- | ------------------------- | ------------------------------------------------------------- | ----- |
| **Definition** | `AnalysisDefinition`      | Declares name, version, required products, outputs            | Yes   |
| **Request**    | `AnalysisRequest`         | Selects analyses, overrides parameters, points at source data | Yes   |
| **Plan**       | `AnalysisPlan`            | Resolved dependency graph, ordered nodes, cache identities    | Yes   |
| **Result**     | `AnalysisResult`          | In-memory semantic result (payload + provenance)              | N/A   |
| **Artifact**   | `PersistedAnalysisResult` | Persisted result with storage references                      | No    |

---

## 3. Core domain models

### 3.1 AnalysisDefinition — what capability exists

```python
@dataclass(frozen=True)
class AnalysisDefinition:
    name: str
    version: int
    requires: tuple[ProductRequirement, ...]
    produces: tuple[ArtifactKey, ...]
    parameter_schema: str  # e.g. "ehp_sn.analysis.representations.mec.MECGridSpec"
```

The definition is declarative data. It does not reference a concrete analyzer
implementation. That mapping lives in the `AnalysisBinding`.

Example:

```python
AnalysisDefinition(
    name="mec.grid",
    version=1,
    requires=(
        ProductRequirement("spatial_population_activity", version=1),
        ProductRequirement("spatial_occupancy", version=1),
    ),
    produces=(
        ArtifactKey(ArtifactKind.ANALYSIS, "mec.grid_scores"),
        ArtifactKey(ArtifactKind.ANALYSIS, "mec.spatial_autocorrelation"),
    ),
    parameter_schema="ehp_sn.analysis.representations.mec.MECGridSpec",
)
```

### 3.2 AnalysisBinding — definition → implementation

```python
AnalyzerFactory = Callable[[], Analyzer]

@dataclass(frozen=True)
class AnalysisBinding:
    definition: AnalysisDefinition
    factory: AnalyzerFactory
```

The binding couples a definition to the factory that constructs its analyzer.
Keeping them separate preserves the definition as pure data and allows the
registry to evolve its factory strategy independently.

### 3.3 AnalysisRequest — what the user asks for

```python
@dataclass(frozen=True)
class AnalysisRequest:
    analyses: tuple[str, ...]
    source: ArtifactSelector
    parameters: Mapping[str, Mapping[str, JSONValue]] = field(default_factory=dict)
```

Lean — identifies analyses, the data source, and parameter overrides.

### 3.4 Product model — requirements, descriptors, and inventory

```python
@dataclass(frozen=True)
class ProductRequirement:
    key: str
    schema_version: int

@dataclass(frozen=True)
class ProductDescriptor:
    key: str
    schema_version: int
    media_type: str  # e.g. "application/zarr"
    dimensions: tuple[str, ...]
    producer_identity: str | None = None
```

The compiler uses a **product inventory**, not a flat set of requirements:

```python
class ProductInventory(Protocol):
    def lookup(self, requirement: ProductRequirement) -> ProductDescriptor | None: ...
    def list_available(self) -> tuple[ProductDescriptor, ...]: ...
```

A `ProductRequirement` expresses a scientific need. A `ProductDescriptor`
describes what is actually available (dimensions, media type, provenance).
A `ProductInventory` maps between them.

### 3.5 AnalysisPlan and AnalysisNode — the resolved graph

```python
@dataclass(frozen=True)
class AnalysisNode:
    analysis_name: str
    analysis_version: int
    parameters: Mapping[str, JSONValue]
    inputs: tuple[ArtifactRef, ...]
    outputs: tuple[ArtifactKey, ...]
    cache_key: str

@dataclass(frozen=True)
class AnalysisPlan:
    request: AnalysisRequest
    nodes: tuple[AnalysisNode, ...]
    digest: str
    diagnostics: tuple[Diagnostic, ...]
```

### 3.6 AnalysisPayload, AnalysisResult, PersistedAnalysisResult

The analyzer returns a **semantic payload** — no persistence, no
artifact references:

```python
@dataclass(frozen=True)
class AnalysisPayload:
    scalars: Mapping[str, Scalar]
    tables: Mapping[str, xr.Dataset | pa.Table]
    arrays: Mapping[str, xr.DataArray]
    diagnostics: tuple[Diagnostic, ...] = ()
```

The executor wraps it in an `AnalysisResult` with provenance:

```python
@dataclass(frozen=True)
class AnalysisResult:
    analysis_name: str
    analysis_version: int
    parameters: Mapping[str, JSONValue]
    provenance: Provenance
    payload: AnalysisPayload
```

Persistence returns a `PersistedAnalysisResult`:

```python
@dataclass(frozen=True)
class PersistedAnalysisResult:
    result: AnalysisResult
    artifacts: tuple[ProducedArtifact, ...]
```

**This resolves the manifest circularity**: `AnalysisResult` does not contain
output artifact references because those do not exist until persistence
completes. Output references live in `PersistedAnalysisResult.artifacts`.
Warnings are carried as `Diagnostic` objects on the plan, the payload, and
the execution context — each at the appropriate lifecycle stage — not
duplicated across both provenance and result.

---

## 4. Core responsibilities

### 4.1 Validate compatibility

Validation is layered:

**Definition-level** (within `AnalysisRegistry.validate()`):

- `parameter_schema` names resolve to importable types;
- produced artifact keys are unique within the registry;
- required product keys reference known product schemas.

**Repository-level** (within `RepositoryDefinitions.validate()`):

- cross-registry references (figure → analysis, analysis → product schema)
  are satisfiable;
- no ownership conflicts on produced artifact keys.

**Request-level** (within `compile_analysis()`):

- requested analyses are registered;
- source inventory satisfies all required products or can be provisioned;
- parameter overrides match declared parameter schemas;
- no circular dependencies among requested analyses.

### 4.2 Resolve dependencies

Transitively walk analysis requirements and collect:

- required artifact products (aggregate, probe, trace);
- required model views for any on-device aggregation;
- required record fields from step records.

Resolution stays within the **product requirement** abstraction — it does
not resolve specific filesystem paths or MLflow runs.

### 4.3 Compile a plan

Given a request, a registry, and a product inventory, produce an
`AnalysisPlan` whose nodes are topologically sorted and carry stable
cache digests.

The compiler must be **pure**:

```python
def compile_analysis(
    request: AnalysisRequest,
    *,
    registry: AnalysisRegistry,
    inventory: ProductInventory,
) -> AnalysisPlan:
    ...
```

It must not:

- open Zarr stores;
- query MLflow;
- inspect the filesystem;
- construct model, accumulator, or analyzer instances.

### 4.4 Execute analytical kernels

```python
def execute_analysis(
    plan: AnalysisPlan,
    *,
    registry: AnalysisRegistry,
    reader: ArtifactReader,
    writer: ArtifactWriter,
    context: AnalysisContext,
) -> tuple[PersistedAnalysisResult, ...]:
    ...
```

For each node:

1. **load** inputs via `reader.read_all(node.inputs)`;
2. **construct** the analyzer via `registry.create(node.analysis_name)`;
3. **compute** `analyzer.analyze(inputs, node.parameters, context=context)` → `AnalysisPayload`;
4. **wrap** in `AnalysisResult` with provenance;
5. **persist** via `writer.write_result(result, node.outputs)` → `PersistedAnalysisResult`.

The executor owns I/O. The analyzer is a pure transformation.

### 4.5 Expose a stable public API

```python
from ehp_sn.analysis import analyze, AnalysisRequest

bundle = analyze(
    AnalysisRequest(
        analyses=("mec.grid", "hpc.place"),
        source=evaluation_ref,
    )
)
```

For framework integrators:

```python
from ehp_sn.analysis.api import (
    AnalysisDefinition,
    AnalysisRegistry,
    AnalysisBinding,
    compile_analysis,
    execute_analysis,
)
```

---

## 5. Product requirements: separating what from how

Analyses declare what they need, not how to collect it.

```
Analysis definition
    │  requires=("spatial_population_activity",)
    ▼
Compiler resolves against product inventory:
    │
    ├── Already produced? → Reuse artifact ref.
    │
    └── Not produced? → Determine compatible provider:
        │
        └── EvaluationConsumer → on-device accumulation → aggregate artifact
```

The compiler does not construct providers. It states what is required.
The executor (or upstream orchestration) maps requirements to compatible
capture bindings.

### Product vs. artifact vocabulary

| Concept              | Role                                                    |
| -------------------- | ------------------------------------------------------- |
| `ProductRequirement` | Scientific need: "I need spatial population statistics" |
| `ProductDescriptor`  | Schema-level definition of what the product contains    |
| `ProductInventory`   | Maps requirements to available descriptors              |
| `ArtifactRef`        | Locator for one stored instance of a product            |
| `ProducedArtifact`   | In-storage representation with provenance               |

This separation prevents analysis definitions from coupling to storage
backends, tracker systems, or accumulator implementations.

---

## 6. Shared artifact contracts

Once analysis and figures consume artifact types, they must not live under
`eval`. They belong in a shared location:

```
ehp_sn/contracts/artifacts.py
    ArtifactKey
    ArtifactKind
    ArtifactRequirement
    ProducedArtifact
    ArtifactRef

ehp_sn/contracts/provenance.py
    Provenance
```

```
analysis ─┐
eval ─────┼──► contracts.artifacts
figures ──┘
```

This move is part of migration phase 1. The `eval` package may re-export for
backward compatibility during the transition.

---

## 7. Registry

A registry stores definitions and bindings, not runtime state.

```python
class AnalysisRegistry:
    def register(
        self,
        definition: AnalysisDefinition,
        factory: AnalyzerFactory,
    ) -> None: ...

    def definition(self, name: str) -> AnalysisDefinition: ...
    def create(self, name: str) -> Analyzer: ...
    def contains(self, name: str) -> bool: ...
    def names(self) -> tuple[str, ...]: ...
    def validate(self) -> tuple[Diagnostic, ...]: ...
```

Rules:

1. Registries store definition + factory bindings only — no model instances,
   no file handles, no pre-constructed analyzers.
2. Registries do not execute anything.
3. Built-in registration is explicit and deterministic on first call.
4. Duplicate registration raises `ValueError`.
5. Cross-registry validation (e.g. analysis→product schema existence) belongs
   in a `RepositoryDefinitions` aggregate, not in individual registries.

Composite validation:

```python
@dataclass(frozen=True)
class RepositoryDefinitions:
    analyses: AnalysisRegistry
    evaluations: EvaluationDefinitionRegistry
    figures: FigureRegistry

    def validate(self) -> tuple[Diagnostic, ...]:
        """Run cross-registry checks."""
        ...
```

---

## 8. Pure kernels vs. analyzers

The scientific inner loops and the orchestration that invokes them are
separate concepts with separate responsibilities.

### Pure kernel (no I/O, no framework dependencies)

```python
def compute_gridness(
    autocorrelation: NDArray[np.floating],
    geometry: SpatialBinGeometry,
    *,
    inner_fraction: float = 0.10,
    outer_fraction: float = 0.55,
) -> GridnessResult:
    """Pure NumPy gridness score from an autocorrelogram."""
    ...
```

### Analyzer (transformation only, no I/O)

```python
class MECGridAnalyzer:
    """Semantic transformation — no loading, no persistence."""

    name = "mec.grid"

    def analyze(
        self,
        inputs: AnalysisInputs,
        parameters: Mapping[str, object],
        *,
        context: AnalysisContext,
    ) -> AnalysisPayload:
        activity = inputs["spatial_population_activity"]
        occupancy = inputs["spatial_occupancy"]

        result = compute_mec_grid_analysis(
            activity=activity,
            occupancy=occupancy,
            spec=MECGridSpec(**parameters),
        )

        return AnalysisPayload(
            scalars=result.summary,
            arrays={"grid_scores": result.grid_scores, ...},
        )
```

**Pure kernels are testable without storage, registries, or MLflow.**
Analyzers add parameter resolution but remain I/O-free. The executor
owns loading and persistence.

---

## 9. Analyzer protocol

```python
from typing import Protocol, Mapping

class Analyzer(Protocol):
    """Structural contract for all analysis implementations."""

    @property
    def name(self) -> str: ...

    def analyze(
        self,
        inputs: AnalysisInputs,
        parameters: Mapping[str, object],
        *,
        context: AnalysisContext,
    ) -> AnalysisPayload:
        ...
```

Concrete analyzers remain ordinary classes or modules — the Protocol is
for type-checking registries, executors, and factories.

Do not create an abstract `BaseAnalyzer` unless it contains meaningful
shared lifecycle behavior that all analyses benefit from.

---

## 10. Analysis context

The context carries infrastructure capabilities that cross-cut analysis
execution but should not be accessed by pure kernels:

```python
@dataclass(frozen=True)
class AnalysisContext:
    cache: AnalysisCache | None = None
    logger: AnalysisLogger | None = None
    execution_id: str | None = None
```

The context does **not** contain a reader or writer. Those are owned by
the executor and passed explicitly. The context is injected by the executor
and forwarded to analyzers for logging, cache lookups, and execution
identity — not for I/O.

---

## 11. I/O separation

The long-term pattern separates load, compute, and persist:

```
                 ┌─────────────────────┐
                 │  ArtifactReader     │  ← protocol (executor injects impl)
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  load_mec_inputs()  │  ← effectful read
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  compute_mec_grid() │  ← pure kernel
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  persist_mec_grid() │  ← effectful write
                 └─────────────────────┘
```

| Operation | Nature          | Test strategy          |
| --------- | --------------- | ---------------------- |
| `load`    | effectful read  | mock reader            |
| `compute` | pure            | synthetic NumPy arrays |
| `persist` | effectful write | mock writer            |
| all three | integration     | real reader + writer   |

---

## 12. Provenance

Every analysis result carries immutable provenance:

```python
@dataclass(frozen=True)
class Provenance:
    operation_name: str
    operation_version: str
    implementation: str          # e.g. "ehp_sn.analysis.representations.mec"
    code_revision: str | None    # git SHA
    plan_digest: str
    parameter_digest: str
    input_artifacts: tuple[ArtifactRef, ...]
```

Provenance answers:

- Which operation and version produced this?
- Which implementation (module path)?
- Which code revision?
- Which plan and parameter digests?
- Which input artifacts were consumed?

It does **not** include output artifact references (those exist only after
persistence) or duplicate a separate warnings list. Diagnostics are carried
on the plan, the context, and the `AnalysisPayload` separately — each at
the appropriate lifecycle stage.

Provenance is:

- persisted alongside the result;
- logged to MLflow as a JSON artifact;
- accessible from notebooks;
- included in report metadata.

---

## 13. Diagnostics

A single diagnostic model avoids duplication across plan, result, and
provenance:

```python
@dataclass(frozen=True)
class Diagnostic:
    code: DiagnosticCode
    severity: DiagnosticSeverity
    message: str
    subject: str | None = None
    context: Mapping[str, JSONValue] = field(default_factory=dict)
```

Diagnostics appear at multiple lifecycle points:

| Stage       | Carried by                    | Examples                                      |
| ----------- | ----------------------------- | --------------------------------------------- |
| Compilation | `AnalysisPlan.diagnostics`    | missing product, incompatible schema version  |
| Execution   | `AnalysisContext.logger`      | slow operation, cache miss, retry             |
| Computation | `AnalysisPayload.diagnostics` | zero valid cells, degenerate input, exclusion |

The final result aggregates them without duplicating the storage model.

---

## 14. Multi-resolution API surface

Three tiers of API for three audiences:

```
Tier 1: Ordinary consumer
─────────────────────────
from ehp_sn.analysis import analyze, AnalysisRequest

results = analyze(
    AnalysisRequest(
        analyses=("mec.grid", "hpc.place"),
        source=evaluation_ref,
    )
)

Tier 2: Framework integrator
────────────────────────────
from ehp_sn.analysis.api import (
    AnalysisDefinition,
    AnalysisRegistry,
    AnalysisBinding,
    compile_analysis,
    execute_analysis,
)

plan = compile_analysis(request, registry=registry, inventory=inventory)
results = execute_analysis(
    plan, registry=registry, reader=reader, writer=writer, context=ctx
)

Tier 3: Scientific developer
─────────────────────────────
from ehp_sn.analysis.spatial.gridness import compute_gridness

result = compute_gridness(
    autocorrelation=autocorr_array,
    geometry=geometry,
)
```

Tier 3 is the fastest iteration path for research. Tier 2 adds orchestration,
caching, and provenance. Tier 1 is the simplest entry point. All tiers use
`AnalysisRequest` as the canonical invocation object — no decomposed keyword
arguments that duplicate the request model.

---

## 15. Figure integration

Figures declare what analysis products they need. The figure compilation
path becomes a consumer of the analysis layer, not the other way around.

```
FigureRequest
    │  declares: requires analysis products X, Y, Z
    ▼
FigureCompiler
    │  resolves FigureRequest → AnalysisRequest
    ▼
AnalysisCompiler
    │  produces AnalysisPlan
    ▼
AnalysisExecutor
    │  executes plan → tuple[PersistedAnalysisResult, ...]
    ▼
FigureRenderer
    │  selects slices from results, renders
    ▼
matplotlib figure
```

The existing figure-driven evaluation pipeline (request figures → compile
figure plan → run evaluation + analyses → render) remains intact. The
difference is internal: figure compilation delegates to the analysis
compiler for artifact-dependent figures instead of duplicating the
resolution logic.

---

## 16. Submodule structure

Packages are created only when at least two cohesive modules exist or a
distinct public extension boundary is needed. Anticipatory nearly-empty
directories are avoided.

```
analysis/
├── __init__.py              ← public API only
├── api.py                   ← analyze(), compile_analysis(), execute_analysis()
├── models.py                ← AnalysisRequest, AnalysisPlan, AnalysisResult,
│                               AnalysisPayload, PersistedAnalysisResult,
│                               AnalysisNode, AnalysisDefinition, AnalysisBinding
├── protocols.py             ← Analyzer, ArtifactReader, ArtifactWriter,
│                               AnalysisCache, ProductInventory
├── compiler.py              ← pure dependency graph compilation
├── execution.py             ← plan execution, I/O orchestration
├── registry.py              ← AnalysisRegistry
├── validation.py            ← cross-registry checks
├── builtins.py              ← register_builtin_analysis_specs()
│
├── spatial/                 ← existing: geometry, gridness, ratemaps, etc.
│   ├── __init__.py
│   ├── geometry.py
│   ├── gridness.py
│   ├── ratemaps.py
│   ├── ratemap_stats.py
│   └── autocorrelation.py   ← moved from figures.plots.autocorr (phase 1)
│
├── representations/         ← per-stream analysis kernels
│   ├── __init__.py
│   ├── mec.py
│   ├── hpc.py
│   ├── lec.py
│   └── pfc.py
│
└── aggregates/              ← aggregate definitions
    ├── __init__.py
    └── definitions.py
```

Subpackages such as `comparisons/` and `deliberation/` are deferred until
there are at least two cohesive modules that justify a distinct namespace.
Existing files not listed above (`runners/`, `specs.py`, `tem_representations.py`)
are retained during migration and deprecated once their content is rehomed.

---

## 17. Dependency rules

```
Allowed:
    analysis → contracts.*
    analysis → contracts.artifacts
    analysis → contracts.diagnostics
    analysis → contracts.provenance
    analysis → types
    analysis → numpy / scipy / xarray
    analysis → torch (optional, behind explicit adapter,
                      must not load checkpoints or training state)

Forbidden:
    analysis → figures.*
    analysis → eval.accumulators
    analysis → eval.inspection
    analysis → eval.cli
    analysis → eval.execution
    analysis → zarr                         (restricted to executor)
    analysis → training.*
    analysis → controllers.*
    analysis → objectives.*

Allowed downstream:
    figures → analysis.protocols
    figures → analysis.models
    figures → analysis.spatial              (pure functions for inline computation)
    reporting → analysis.*
    notebooks → analysis.*
```

Enforced by an architecture test:

```python
FORBIDDEN_ANALYSIS_DEPENDENCIES = {
    "ehp_sn.figures",
    "ehp_sn.training",
    "ehp_sn.controllers",
    "ehp_sn.objectives",
    "ehp_sn.eval.cli",
    "ehp_sn.eval.execution",
    "ehp_sn.eval.accumulators",
    "ehp_sn.eval.inspection",
    "zarr",
}
```

---

## 18. Configuration

Analysis parameters are Pydantic `BaseModel` with `extra="forbid"` when
they originate from external configuration (TOML/YAML/JSON). Frozen
dataclasses are preferred for programmatically constructed specs.

```python
class MECGridSpec(BaseModel, extra="forbid", frozen=True):
    min_occupancy: int = 5
    smoothing_sigma: float | None = None
    grid_score_threshold: float = 0.3
```

The `parameter_schema` field on `AnalysisDefinition` references the
importable path of this type (e.g.
`"ehp_sn.analysis.representations.mec.MECGridSpec"`).
The registry resolves it for validation and the executor resolves it for
digest computation.

Configuration describes analysis-owned choices only:

```python
# Good — analysis-owned
class GridnessSpec(BaseModel, frozen=True):
    inner_fraction: float = 0.10
    outer_fraction: float = 0.55

# Bad — duplicates model or task spec
class GridnessSpec(BaseModel, frozen=True):
    hrm_num_layers: int       # belongs in model config
```

---

## 19. Migration path

### Phase 1 — enforce boundaries (current cycle)

1. Move `compute_spatial_autocorrelogram` from `figures.plots.autocorr` to
   `analysis.spatial.autocorrelation`. Re-export from old location with
   deprecation warning.
2. Extract shared artifact types (`ArtifactKey`, `ArtifactKind`,
   `ArtifactRequirement`, `ProducedArtifact`, `ArtifactRef`) into
   `contracts.artifacts`. Re-export from `eval.contracts` for backward
   compatibility.
3. Extract shared diagnostic types into `contracts.diagnostics`.
4. Remove `analysis → eval.inspection` import.
5. Remove `analysis → eval.accumulators` import (use lazy factory
   registration in `builtins.py`).
6. Add architecture test that rejects forbidden imports.

### Phase 2 — add analysis contracts

1. Add `AnalysisDefinition`, `AnalysisBinding`, `ProductRequirement`,
   `ProductDescriptor`, `ProductInventory` to `models.py`.
2. Add `Analyzer` Protocol to `protocols.py`.
3. Add `AnalysisRequest`, `AnalysisPlan`, `AnalysisNode`.
4. Add `AnalysisPayload`, `AnalysisResult`, `PersistedAnalysisResult`.
5. Add `Provenance` to `contracts.provenance`.
6. Adapt one analysis (MEC grid) end-to-end through the new contract.

### Phase 3 — separate I/O from kernels

1. Split `compute_mec_grid_analysis` (Path → artifact) into
   `load_mec_inputs`, `compute_mec_grid`, `persist_mec_grid`.
2. Repeat for `compute_hpc_place_analysis`.

### Phase 4 — add analysis API

1. Implement `compile_analysis()` (pure compiler for analysis-only graphs).
2. Implement `execute_analysis()` (plan execution with reader/writer/cache).
3. Implement `analyze()` (top-level consumer facade).
4. Make figure compilation delegate to the analysis compiler for
   artifact-dependent figures.

### Phase 5 — introduce xarray

Convert persisted trace and population-statistic schemas to xarray-compatible
datasets, one schema at a time. Start with spatial population activity.

### Phase 6 — unify lifecycle vocabulary

Apply the same `definition → request → plan → result → artifact` vocabulary
to evaluation and reporting once analysis is stable. Do not extract a generic
`BaseCompiler` or `BaseExecutor` until at least two packages demonstrate
genuinely identical behavior.

---

## 20. What analysis does not own

- Model construction, checkpoint loading, device placement.
- Task rollout, environment interaction, action selection.
- Primary benchmark metric computation (those belong to `eval`/`metrics`).
- Visual rendering, plot styling, color palettes.
- Notebook or report orchestration.
- MLflow run creation or experiment management.
- Filesystem path resolution or Zarr store management.

These concerns belong to `eval`, `figures`, `reporting`, or infrastructure
adapters. Analysis sits between them: consuming evaluation artifacts,
producing domain results, and letting figures/reports consume those results.

---

## 21. The metric vs. analysis boundary

```
Metric:
    A bounded measurement that is part of evaluation correctness.
    sequences_exact, accuracy_ancestral_revisit, step_match

Analysis:
    A post-hoc transformation that interprets, aggregates, compares,
    conditions, or decomposes evaluation products.
    gridness scores, place-field statistics, pathway-conditioned accuracy,
    bootstrap CI across seeds, ACT halt-step distributions, latent projections
```

A metric can become an analysis input (e.g. pathway-conditioned accuracy is
an analysis computed from per-episode accuracy metrics). Analysis must not
redefine the canonical benchmark metric.

---

## 22. Summary: design principles

| Principle                           | Rationale                                                                  |
| ----------------------------------- | -------------------------------------------------------------------------- |
| **Pure compiler**                   | Plan compilation does no I/O — inspectable, cacheable, testable            |
| **Binding separates def from impl** | `AnalysisBinding` couples definition to factory; neither owns execution    |
| **Executor-owned I/O**              | Load and persist are executor concerns; analyzers are semantic transforms  |
| **Products, not implementations**   | Declare what data is needed, not how it is collected                       |
| **Protocols over base classes**     | Structural typing, no forced inheritance                                   |
| **Frozen dataclasses for domain**   | Immutable, hashable, serializable                                          |
| **Pydantic at boundaries only**     | Validate external input, convert to domain dataclass                       |
| **xarray at data boundaries**       | Named dimensions for safety; NumPy inside hot kernels                      |
| **Provenance on every result**      | `Provenance` answers what, how, from what — without output refs            |
| **Three API tiers**                 | Consumer, integrator, scientist — each with appropriate complexity         |
| **Figure-driven is not wrong**      | Existing figure compilation is valid; analysis delegation makes it cleaner |
| **No anticipatory packages**        | Create subpackages only when cohesive modules exist                        |
