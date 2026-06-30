# Diagnostics Architecture

> Canonical design for the EHC diagnostic subsystem — how model internals
> are inspected, checked, captured, assessed, and persisted.

The diagnostics subsystem answers three distinct questions:

1. **Probe question**: _Does a specific mechanism inside the trained model
   work?_ (e.g. "Can the Hebbian attractor memory M self-retrieve?")
2. **Trace question**: _What does the model's internal state look like
   across a population of evaluation cases?_ (e.g. "What is the mean L2
   norm of pfc/z_H across 128 MazeHard episodes?")
3. **Health question**: _Is the model numerically stable, resource-safe,
   and well-behaved during training?_ (e.g. "Are any gradients NaN?
   Is memory usage growing without bound?")

These require different tools, different triggers, and different consumers.
The architecture distinguishes them explicitly.

---

## 1. Three diagnostic paths

```
PATH A: OFFLINE PROBES        PATH B: HEALTH CHECKS        PATH C: EVALUATION TRACES
─────────────────────         ──────────────────────       ────────────────────────────

Purpose: Deep, targeted       Purpose: Runtime safety      Purpose: Population-level
         single-episode       and sanity during            evidence across cases
         mechanism test       training/evaluation

Trigger: Manual               Trigger: Lifecycle hooks     Trigger: ehp eval run
         (script/notebook)    (after forward/loss/step)    (recipe-driven)

Model:   Loaded, eval()       Model:   Live training       Model:   Loaded by offline
                                        or eval model               runner

Scope:   1 episode             Scope:   1 step/batch/       Scope:   N cases (e.g. 128),
                                         episode                    M steps each

Output:  Pydantic Result      Output:  DiagnosticFinding   Output:  Zarr archive +
         (JSON)                list, DiagnosticReport               metrics.csv

Consumer: Researcher           Consumer: Trainer,           Consumer: Report notebooks,
                                        experiment builder,          figure renderer
                                        CI/CD gate
```

---

## 2. Core diagnostic contracts

### 2.1 Findings and severity

Every diagnostic — whether from a probe, a health check, or a trace
assessment — produces **findings**, not raw metrics or free-form strings.

```python
from enum import StrEnum
from dataclasses import dataclass, field
from typing import Mapping, Optional

class DiagnosticSeverity(StrEnum):
    """Severity of a diagnostic finding.

    Ordered by increasing concern: PASS < WARNING < ERROR < CRITICAL.
    """
    PASS = "pass"          # Expected behaviour confirmed
    WARNING = "warning"    # Anomaly detected, not blocking
    ERROR = "error"        # Definite problem, likely degrades results
    CRITICAL = "critical"  # Training/inference should stop
    SKIPPED = "skipped"    # Check could not run (missing dependency, etc.)
    INTERNAL = "internal"  # Diagnostic itself failed

class DiagnosticCode(StrEnum):
    """Canonical diagnostic codes.

    Codes follow a ``subsystem.short_name`` convention.
    Codes carry semantic meaning independent of the probe or check
    that produced them, so consumers can filter and route without
    knowing the producing module.
    """
    # TEM memory
    MEMORY_DYNAMICS = "tem.memory_dynamics"
    ATTRACTOR_DIVERGENCE = "tem.attractor_divergence"
    QUERY_ALIGNMENT_FAILURE = "tem.query_alignment_failure"
    CONVERGENCE_DEPTH = "tem.convergence_depth"
    WEAK_SENSORY_CORRECTION = "tem.weak_sensory_correction"
    # TEM pathway
    PATHWAY_CONTENT_LOSS = "tem.pathway_content_loss"
    PATHWAY_LATE_DECAY = "tem.pathway_late_decay"
    PATHWAY_PLACE_CODE_LOSS = "tem.pathway_place_code_loss"
    PATHWAY_DECODER_MISMATCH = "tem.pathway_decoder_mismatch"
    # HRM dynamics
    HRM_HIGH_LEVEL_COLLAPSE = "hrm.high_level_collapse"
    HRM_LOW_LEVEL_COLLAPSE = "hrm.low_level_collapse"
    HRM_DELTA_RATIO_ANOMALY = "hrm.delta_ratio_anomaly"
    # Numerical health
    NAN_IN_GRADIENTS = "numerical.nan_in_gradients"
    NAN_IN_ACTIVATIONS = "numerical.nan_in_activations"
    EXPLODING_GRADIENTS = "numerical.exploding_gradients"
    VANISHING_GRADIENTS = "numerical.vanishing_gradients"
    PARAMETER_UPDATE_RATIO = "numerical.parameter_update_ratio"
    # Replay / carry health
    CARRY_STALE_STATE = "replay.carry_stale_state"
    HALT_REACTIVATION = "replay.halt_reactivation"
    SLOT_UNDERUTILIZATION = "replay.slot_underutilization"
    # Resource health
    CUDA_MEMORY_GROWTH = "resource.cuda_memory_growth"
    STEP_TIME_REGRESSION = "resource.step_time_regression"

@dataclass(frozen=True, slots=True)
class DiagnosticFinding:
    """One standardised diagnostic observation.

    Every probe, check, and trace assessment produces findings.
    Consumers filter, aggregate, and escalate findings without
    knowing which module produced them.
    """
    code: DiagnosticCode
    severity: DiagnosticSeverity
    message: str
    details: Optional[str] = None
    source: str = ""                # e.g. "tem_memory_probe", "grad_check"
    observed: Mapping[str, float] = field(default_factory=dict)
    expected: Mapping[str, float] = field(default_factory=dict)
    threshold: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """Aggregated diagnostic report from one run or check cycle."""
    findings: tuple[DiagnosticFinding, ...]
    worst_severity: DiagnosticSeverity
    source_hint: str = ""

    @property
    def has_errors(self) -> bool:
        return self.worst_severity in (
            DiagnosticSeverity.ERROR,
            DiagnosticSeverity.CRITICAL,
        )

def summary_report(findings: list[DiagnosticFinding]) -> DiagnosticReport:
    """Build the summary from a list of findings."""
    if not findings:
        return DiagnosticReport(
            findings=(),
            worst_severity=DiagnosticSeverity.PASS,
        )
    severities = [f.severity for f in findings]
    order = [
        DiagnosticSeverity.PASS,
        DiagnosticSeverity.WARNING,
        DiagnosticSeverity.ERROR,
        DiagnosticSeverity.CRITICAL,
    ]
    worst = max(severities, key=lambda s: order.index(s))
    return DiagnosticReport(
        findings=tuple(findings),
        worst_severity=worst,
    )
```

**Design rules for findings**:

- Every probe, check, and trace assessment returns findings.
- Findings are small, serialisable, and immutable.
- `observed` and `expected` carry the numeric evidence — consumers
  decide how to present it.
- `code` is a semantic identifier, not a free-form message.
- `severity` follows the ordered enum: PASS < WARNING < ERROR < CRITICAL.

### 2.2 Policy — separating measurement from interpretation

Measurement thresholds are **not embedded** in probe or check logic.
They live in separate policy objects:

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class QueryAlignmentPolicy:
    """Thresholds for classifying TEM query-alignment findings."""
    minimum_self_retrieval_accuracy: float = 0.3
    maximum_iteration_degradation: float = 0.1
    minimum_depth_improvement: float = 0.15
    minimum_grid_nn_accuracy: float = 0.3
    weak_correction_delta: float = 0.05

@dataclass(frozen=True, slots=True)
class NumericalHealthPolicy:
    """Thresholds for numerical health checks."""
    max_gradient_norm: float = 100.0
    min_gradient_norm: float = 0.0
    max_parameter_update_ratio: float = 0.1
    nan_action: DiagnosticSeverity = DiagnosticSeverity.CRITICAL

@dataclass(frozen=True, slots=True)
class ReplayHealthPolicy:
    """Thresholds for replay/carry health checks."""
    max_carry_age_steps: int = 1000
    min_slot_utilization: float = 0.1
    halt_reactivation_action: DiagnosticSeverity = DiagnosticSeverity.WARNING

@dataclass(frozen=True, slots=True)
class ResourceHealthPolicy:
    """Thresholds for resource health checks."""
    max_memory_growth_gb: float = 4.0
    max_step_time_regression_factor: float = 2.0
```

The assessment function uses the policy:

```python
def assess_query_alignment(
    result: QueryAlignmentResult,
    policy: QueryAlignmentPolicy,
) -> list[DiagnosticFinding]:
    """Convert raw probe observations into standardised findings."""
    ...
```

This decouples:

- **measurement** — `QueryAlignmentResult` (observed data)
- **policy** — `QueryAlignmentPolicy` (what thresholds matter)
- **interpretation** — `list[DiagnosticFinding]` (standardised output)

Policies can vary by experiment, model family, or strictness without
changing the probe implementation.

### 2.3 Probe definition — a common registration contract

Every probe is described by a `ProbeDefinition`:

```python
from collections.abc import Callable
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")

@dataclass(frozen=True, slots=True)
class ProbeDefinition(Generic[InputT, ResultT]):
    """Metadata and callables for one registered diagnostic probe."""
    name: str
    model_family: str                    # "tem-v1", "tem-v2", "hrm-v1", etc.
    description: str
    result_type: type[ResultT]
    produce: Callable[..., ResultT]
    load: Callable[[str | Path], ResultT]

# Registry — populated at import time by probe modules.
_PROBE_REGISTRY: dict[str, ProbeDefinition] = {}

def register_probe(defn: ProbeDefinition) -> None:
    _PROBE_REGISTRY[defn.name] = defn

def list_probes(*, model_family: Optional[str] = None) -> list[ProbeDefinition]:
    result = list(_PROBE_REGISTRY.values())
    if model_family:
        result = [p for p in result if p.model_family == model_family]
    return result
```

Each probe module registers itself:

```python
# diagnostics/probes/tem_memory.py

PROBE = ProbeDefinition(
    name="tem_memory",
    model_family="tem-v1",
    description="Hebbian memory self-retrieval test",
    result_type=MemoryProbeResult,
    produce=produce_tem_memory_probe,
    load=load_tem_memory_probe,
)
register_probe(PROBE)
```

This preserves the existing functional API (`produce_*`, `load_*`) while
adding discovery and tooling support.

### 2.4 Generic persistence

Probes share a single serialisation path instead of duplicating
`persist_*_probe` / `load_*_probe` per probe:

```python
import json
from pathlib import Path
from pydantic import BaseModel

def persist_probe_result(
    result: BaseModel,
    path: Path,
) -> Path:
    """Write any Pydantic probe result to JSON."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    return path

def load_probe_result(
    result_type: type[ResultT],
    path: str | Path,
) -> ResultT:
    """Load any Pydantic probe result from JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return result_type.model_validate(raw)
```

Probe-specific convenience wrappers remain optional:

```python
# Thin convenience — delegates to generic functions above.
def persist_tem_memory_probe(
    result: MemoryProbeResult,
    output_dir: Path,
    *,
    filename: str = "memory_probe.json",
) -> Path:
    return persist_probe_result(result, output_dir / filename)
```

---

## 3. Probe path (detailed)

### 3.1 Probes overview

| Probe ID              | Model Family       | Question                         | Result Type            | Findings Source                           |
| --------------------- | ------------------ | -------------------------------- | ---------------------- | ----------------------------------------- |
| `tem_memory`          | `tem-v1`, `tem-v2` | Can M self-retrieve?             | `MemoryProbeResult`    | Memory health, self-retrieval metrics     |
| `tem_pathway`         | `tem-v1`, `tem-v2` | Where is content lost in decode? | `PathwayProbeResult`   | Per-stage contrast ratios, logit accuracy |
| `tem_query_alignment` | `tem-v1`, `tem-v2` | Does mec_to_hpc(g) index memory? | `QueryAlignmentResult` | Per-query-type NN accuracy, sharpening    |
| `cue_recall`          | `hrm-v1`           | Can HRM recall cued WM item?     | `CueRecallProbeResult` | Recall correctness, z_H/z_L traces        |

All probe results extend `BaseModel` with `extra="forbid"` and are
JSON-serialisable with NaN/Inf support.

### 3.2 TEM Memory Probe — the Hebbian self-retrieval test

**Question**: _Can the learned Hebbian attractor memory M retrieve its own
stored posterior place codes?_

```
                    p_post[t]  ──── query ────►  Hebbian M  ────►  p_self_recall[t]
                         │                              ▲               │
                         │                    ┌─────────┴─────────┐     │
                         │                    │  matrix stats:     │     │
                         │                    │  - Frobenius norm   │     │
                         │                    │  - entry std        │     │
                         │                    │  - effective rank   │     │
                         │                    │  - saturation frac  │     │
                         │                    └───────────────────┘     │
                         │                                              ▼
                         └──────── candidates bank ◄──── NN evaluation ──►
                                                                    │
                                               ┌────────────────────┴────────────────────┐
                                               │  NN accuracy         top-5/10 accuracy   │
                                               │  temporal distance   same-obs fraction   │
                                               │  same-position frac  spatial distance    │
                                               └─────────────────────────────────────────┘
```

**Findings produced** (via assessment):

| Code                  | Severity | Condition                                              |
| --------------------- | -------- | ------------------------------------------------------ |
| `tem.memory_dynamics` | ERROR    | `nn_accuracy` < threshold                              |
| `tem.memory_dynamics` | WARNING  | `memory_saturation_fraction` > 0.9                     |
| `tem.memory_dynamics` | WARNING  | `memory_effective_rank` < 2                            |
| `tem.memory_dynamics` | WARNING  | `retrieved_vs_path_delta` < 0.01 (correction inactive) |

### 3.3 TEM Pathway Probe — locating content loss

**Question**: _Where is retrieved/ancestral content lost between the HPC
place code and the final observation logits?_

The model decodes through 5 stages. The probe measures representation
quality at each stage for all three pathways (inference, retrieved,
ancestral).

```
  p_post / p_recall / p_path
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ Stage 0:  Place Code           (T, S)  flattened multi-freq code │
  │ Stage 1:  Inverse Projection   lec_to_hpc.inverse(p_*)           │
  │ Stage 2:  Single-Freq Slice    inv[prediction_freq]              │
  │ Stage 3:  Decoder Input        w_x * code + b_x                  │
  │ Stage 4:  Logits               MLPDecoder(code) or raw           │
  └──────────────────────────────────────────────────────────────────┘

  Per-stage metrics:  norm_mean, feature_std_mean,
                      pairwise_cosine_mean, effective_rank

  Contrast ratios:    retrieved / inference, ancestral / inference
```

**Findings produced** (via assessment):

| Code                           | Severity | Condition                                                         |
| ------------------------------ | -------- | ----------------------------------------------------------------- |
| `tem.pathway_place_code_loss`  | ERROR    | contrast ratio at stage 0 < 0.3                                   |
| `tem.pathway_late_decay`       | WARNING  | contrast at stage 0 ~1.0 but stage 4 < 0.3                        |
| `tem.pathway_decoder_mismatch` | WARNING  | contrast preserved through stage 3 but logit accuracy near chance |
| `tem.pathway_content_loss`     | WARNING  | ancestral/inference ratio < 0.2 across all stages                 |

### 3.4 TEM Query Alignment Probe — grid-to-memory indexing

**Question**: _Does `mec_to_hpc(g)` produce a valid query into the Hebbian
attractor memory?_

Tests 7 query types against the attractor memory M at multiple iteration
depths (1, 2, 3, 5, 10, 20):

| Query Key            | Source                              | What It Tests                               |
| -------------------- | ----------------------------------- | ------------------------------------------- |
| `p_post`             | Posterior place code                | Gold standard — self-retrieval from M       |
| `p_recall`           | Generative (retrieved) code         | Corrected retrieval quality                 |
| `p_path`             | Ancestral (prior) code              | Path integration without sensory correction |
| `mec_to_hpc_g_post`  | MEC posterior grid → HPC projection | **Primary question**                        |
| `mec_to_hpc_g_prior` | MEC prior grid → HPC projection     | Uncorrected grid query                      |
| `lec_to_hpc_x`       | LEC sensory code → HPC projection   | Sensory-driven query                        |
| `random`             | Random same-norm baseline           | Chance-level attractor behaviour            |

**Findings produced** (via `assess_query_alignment(result, policy)`):

| Code                          | Severity | Typical condition                                    |
| ----------------------------- | -------- | ---------------------------------------------------- |
| `tem.memory_dynamics`         | ERROR    | p_post self-retrieval nn_accuracy < 0.3              |
| `tem.attractor_divergence`    | ERROR    | p_post cosine degrades from iter 1→5 by >0.1         |
| `tem.query_alignment_failure` | ERROR    | p_post/lec_to_hpc(x) succeed but mec_to_hpc(g) fails |
| `tem.convergence_depth`       | WARNING  | mec_to_hpc(g) improves from iter 1→20 by >0.15       |
| `tem.weak_sensory_correction` | WARNING  | g_post ≈ g_prior (delta < 0.05)                      |

### 3.5 HRM Dynamics — latent state metrics

**Question**: _How do high-level vs low-level PFC states evolve during
deliberation?_

```python
from pydantic import BaseModel, Field

class HRMDynamicsMetrics(BaseModel, extra="forbid"):
    """Typed HRM latent-dynamics measurement."""
    h_state_norm_mean: float = Field(default=float("nan"))
    l_state_norm_mean: float = Field(default=float("nan"))
    h_state_delta_mean: float = Field(default=float("nan"))
    l_state_delta_mean: float = Field(default=float("nan"))
    h_l_delta_ratio: float = Field(default=float("nan"))
```

**Findings produced** (via assessment):

| Code                      | Severity | Condition                       |
| ------------------------- | -------- | ------------------------------- |
| `hrm.high_level_collapse` | WARNING  | `h_state_norm_mean` < 0.001     |
| `hrm.low_level_collapse`  | WARNING  | `l_state_norm_mean` < 0.001     |
| `hrm.delta_ratio_anomaly` | WARNING  | `h_l_delta_ratio` > 10 or < 0.1 |

### 3.6 Cue Recall Probe — HRM block-encoding WM test

**Question**: _Can the HRM v1 recall a cued item from block-encoded working
memory?_

Uses **block encoding**: three 300-slot blocks each carrying one item
prototype (A, B, C), with the cued block scaled at cue step.

```python
class CueRecallProbeResult(BaseModel, extra="forbid"):
    """Typed cue-recall probe output."""
    model_family: str = "hrm-v1"
    target_item: int
    predicted_item: Optional[int] = None
    recall_correct: bool = False
    trace: Optional[object] = None  # TraceTree
    findings: tuple[DiagnosticFinding, ...] = ()
```

The trace is embedded. An adapter wraps the result as an
`EvaluationCaseResult` for the evaluation artifact pipeline.

---

## 4. Health check path

### 4.1 Lifecycle-aware checks

Health checks execute at specific points in the training or evaluation
cycle. Each check receives context appropriate to its lifecycle hook:

```
  after_forward       → activations, hidden states, raw outputs
  after_loss          → loss value, logits, targets
  after_backward      → gradient norms, gradient histogram extremes
  after_optimizer     → parameter update ratios, weight statistics
  after_rollout_step  → carry state, halted mask, slot occupancy
  on_episode_end      → episode-level aggregations
  on_run_end          → resource summaries, final statistics
```

### 4.2 Numerical health checks

Checks execute after forward, loss, backward, and optimizer steps.
They produce findings with `numerical.*` diagnostic codes.

**Check catalogue**:

| Check ID                 | Lifecycle Hook    | What It Tests                                   | Default Severity |
| ------------------------ | ----------------- | ----------------------------------------------- | ---------------- | --- | --- | --- | --- | --- | --- | ------------------- | ------- |
| `nan_activations`        | `after_forward`   | Any activation contains NaN                     | CRITICAL         |
| `nan_gradients`          | `after_backward`  | Any gradient is NaN                             | CRITICAL         |
| `exploding_gradients`    | `after_backward`  | Max gradient norm > threshold                   | WARNING          |
| `vanishing_gradients`    | `after_backward`  | Min gradient norm < threshold (non-zero params) | WARNING          |
| `parameter_update_ratio` | `after_optimizer` | `                                               |                  | Δθ  |     | /   |     | θ   |     | ` exceeds threshold | WARNING |
| `dead_activations`       | `after_forward`   | Fraction of zero activations > threshold        | WARNING          |
| `loss_divergence`        | `after_loss`      | Loss increased by >10× in one step              | ERROR            |
| `latent_collapse`        | `after_forward`   | Mean hidden norm < threshold                    | WARNING          |

**Implementation pattern**:

```python
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

class HealthCheck(Protocol):
    """A lifecycle-aware health check."""
    name: str
    hook: str  # "after_forward", "after_backward", etc.

    def __call__(
        self,
        context: Any,
        policy: NumericalHealthPolicy,
    ) -> list[DiagnosticFinding]:
        ...

@dataclass(frozen=True, slots=True)
class HealthCheckDefinition:
    """Registered health check with metadata."""
    name: str
    hook: str
    check: Callable[..., list[DiagnosticFinding]]
    model_families: frozenset[str] | None = None  # None = all

# Registry
_HEALTH_CHECK_REGISTRY: dict[str, HealthCheckDefinition] = {}

def register_health_check(defn: HealthCheckDefinition) -> None:
    _HEALTH_CHECK_REGISTRY[defn.name] = defn

def run_health_checks(
    hook: str,
    *,
    context: Any,
    policy: NumericalHealthPolicy,
    model_family: Optional[str] = None,
) -> DiagnosticReport:
    """Run all registered health checks for a lifecycle hook."""
    findings: list[DiagnosticFinding] = []
    for defn in _HEALTH_CHECK_REGISTRY.values():
        if defn.hook != hook:
            continue
        if (
            defn.model_families is not None
            and model_family not in defn.model_families
        ):
            continue
        try:
            findings.extend(defn.check(context, policy))
        except Exception as exc:
            findings.append(
                DiagnosticFinding(
                    code=DiagnosticCode.INTERNAL,
                    severity=DiagnosticSeverity.INTERNAL,
                    message=f"Health check {defn.name!r} failed: {exc}",
                    source=defn.name,
                )
            )
    return summary_report(findings)
```

### 4.3 Replay / carry health checks

Checks on recurrent rollout state:

| Check ID                | Lifecycle Hook       | What It Tests                                   | Default Severity |
| ----------------------- | -------------------- | ----------------------------------------------- | ---------------- |
| `carry_stale_state`     | `after_rollout_step` | Any active slot older than max_carry_age        | WARNING          |
| `halt_reactivation`     | `after_rollout_step` | A slot that halted is now marked active again   | WARNING          |
| `slot_underutilization` | `on_episode_end`     | Fraction of slots that never became active      | WARNING          |
| `tbptt_boundary`        | `after_rollout_step` | TBPTT boundary state is inconsistent with carry | ERROR            |

### 4.4 Resource health checks

Checks on system resources during training:

| Check ID                           | Lifecycle Hook       | What It Tests                       | Default Severity |
| ---------------------------------- | -------------------- | ----------------------------------- | ---------------- |
| `cuda_memory_growth`               | `on_run_end`         | Peak CUDA memory growth since start | WARNING          |
| `step_time_regression`             | `after_rollout_step` | Step time increased > 2× baseline   | WARNING          |
| `gradient_accumulation_saturation` | `after_backward`     | Accumulation buffer > max capacity  | ERROR            |

### 4.5 Health check integration

Health checks integrate into the training loop through a thin callback:

```python
class DiagnosticHealthCallback(Callback):
    """Lightning callback that runs health checks at each lifecycle hook."""

    def __init__(
        self,
        numerical_policy: NumericalHealthPolicy | None = None,
        replay_policy: ReplayHealthPolicy | None = None,
        resource_policy: ResourceHealthPolicy | None = None,
    ) -> None:
        self._numerical = numerical_policy or NumericalHealthPolicy()
        self._replay = replay_policy or ReplayHealthPolicy()
        self._resource = resource_policy or ResourceHealthPolicy()

    def on_after_backward(self, trainer, module) -> None:
        report = run_health_checks(
            "after_backward",
            context=module,
            policy=self._numerical,
            model_family=getattr(module, "model_family", None),
        )
        if report.has_errors:
            trainer.should_stop = True
```

The callback does not depend on `diagnostics/` importing Lightning —
the callback lives in `lightning/` and imports from `diagnostics/`.

---

## 5. Trace path (detailed)

### 5.1 Trace capture lifecycle

```
Recipe TOML                          Runtime                              Artifact
──────────                          ───────                              ────────

[capture]                    ┌─────────────────────┐               ┌──────────────┐
profile = "diagnostic"  ───► │ TraceSpec.build()   │               │ traces.zarr/ │
max_cases = 16          ───► │  → list[TraceField] │               │ trace_index  │
max_units = 512         ───► └──────────┬──────────┘               │   .json      │
                                        │                          └──────────────┘
                                        ▼
                             ┌─────────────────────┐
                             │ TraceObserver       │  per step:
                             │  .observe(ctx)      │  field.get(ctx) → tensor
                             │  .flush(sink)       │  sink.write(step, values)
                             └──────────┬──────────┘
                                        │
                                        ▼
                             ┌─────────────────────┐
                             │ TraceSink           │
                             │  InMemoryTraceSink  │  → TraceTree (small recall)
                             │  ZarrTraceSink      │  → Zarr archive (large eval)
                             │  ParquetEventSink   │  → Parquet (event data)
                             └─────────────────────┘
```

### 5.2 Trace field definition

Every trace field is defined in `traces/specs.py` as a `TraceField` with:

- A **name** (canonical dot-separated path string, e.g. `"pfc/z_H"`)
- A **getter** function `(ctx: StepContext) → Tensor | None`
- A **storage** hint (`"dense"` or `"meta"`)

Trace field paths are **defined exactly once** in `traces/keys.py` and
imported by selectors, templates, figures, and consumers — never duplicated
as literal strings.

### 5.3 Trace fields by paradigm

**TEM paradigm** (Arena):

| Trace key                      | Source            | Shape          | Description                              |
| ------------------------------ | ----------------- | -------------- | ---------------------------------------- |
| `diagnostic/lec/cells`         | LEC module        | `(B, n_cells)` | LEC activations by frequency             |
| `diagnostic/lec/filtered`      | LEC filter        | `(B, n_cells)` | LEC filtered by frequency                |
| `diagnostic/lec/sensory_code`  | LEC sensory input | `(B, D)`       | Raw sensory code entering LEC inference  |
| `diagnostic/mec/location_mean` | MEC module        | `(B, n_cells)` | MEC location codes by frequency          |
| `diagnostic/hpc/location_mean` | HPC module        | `(B, n_cells)` | HPC grounded-location codes              |
| `diagnostic/hpc/memory`        | HPC memory        | `(B, S, S)`    | HPC memory matrix                        |
| `pred/observation_id/post`     | Posterior decoder | `(B, n_obs)`   | Observation logits (sensory-conditioned) |
| `pred/observation_id/recall`   | Retrieved decoder | `(B, n_obs)`   | Observation logits (grid-corrected)      |
| `pred/observation_id/path`     | Ancestral decoder | `(B, n_obs)`   | Observation logits (prior)               |

**HRM paradigm** (MazeHard, Goaltrace, Routebind):

| Trace key | Source      | Shape          | Description             |
| --------- | ----------- | -------------- | ----------------------- |
| `pfc/z_H` | PFC scratch | `(T, B, S, D)` | High-level memory state |
| `pfc/z_L` | PFC scratch | `(T, B, S, D)` | Low-level memory state  |

**Shared** (all paradigms):

| Trace key                 | Source      | Description                                                               |
| ------------------------- | ----------- | ------------------------------------------------------------------------- |
| `act/halted`              | Controller  | Whether each slot halted this step                                        |
| `act/steps`               | Controller  | Step count per slot                                                       |
| `world_step/location_ids` | Environment | Current location per slot                                                 |
| `world_step/observation`  | Environment | Current observation per slot                                              |
| `arena/*`                 | Arena task  | Wall mask, observation ids, trajectory, revisit mask, actions, valid mask |

### 5.4 Trace naming convention

Trace keys follow the namespace hierarchy:

```
act/*          — ACT halting/stepping signals
pred/*         — model output predictions
value/*        — value / Q-function outputs
reward/*       — environment reward signals
policy/*       — policy / action outputs
dopamine/*     — reward prediction error signals
pfc/*          — PFC hidden-state diagnostics
world_step/*   — environment state
arena/*        — arena-specific metadata
diagnostic/*   — model internals (LEC, MEC, HPC)
```

### 5.5 Trace sink implementations

| Sink                         | Module           | Use case                      | Format                              |
| ---------------------------- | ---------------- | ----------------------------- | ----------------------------------- |
| `InMemoryTraceSink`          | `traces/sink.py` | Small episodes, probe outputs | In-memory `TraceTree`               |
| `UnboundedInMemoryTraceSink` | `traces/sink.py` | Long episodes                 | In-memory, no step limit            |
| `ZarrTraceSink`              | `traces/sink.py` | Large evaluation runs         | Zarr on disk with Blosc compression |
| `ParquetEventSink`           | `traces/sink.py` | Event-based analysis          | Parquet per step                    |

### 5.6 TraceConsumer — bridging to evaluation

The `TraceConsumer` (in `eval/consumers.py`) wraps `TraceObserver +
TraceSink` as an `EvaluationConsumer`, enabling it to participate in the
standard evaluation lifecycle:

```
  begin_run    begin_case    update(ctx)    end_case    finalize()    close()
      │            │             │              │            │           │
      │            │       observer.observe()   │    sink.finalize()     │
      │            │       sink.write()         │    write trace_index   │
      ▼            ▼             ▼              ▼            ▼           ▼
```

The `trace_index.json` sidecar maps case IDs to (start, length) offsets
within the concatenated Zarr arrays, enabling random access per case.

### 5.7 Trace assessment — from traces to findings

The trace assessment bridge converts raw trace data into standardised
findings:

```
   eval artifact directory
            │
            ▼
   load_artifact_run_cases(artifact_dir)
            │
            ▼
   compute_*_metrics_from_trace(case.trace)
            │
            ▼
   assess_*_dynamics(metrics, policy) → list[DiagnosticFinding]
            │
            ▼
   aggregate across cases → DiagnosticReport
```

This replaces the current `diagnostics/reporting.py` module.
The assessment logic moves to `diagnostics/assessments/` and produces
findings, not presentation-ready tables. Presentation moves to
`reporting/`.

---

## 6. Package structure

### 6.1 Target layout

```
ehp_sn/diagnostics/
├── __init__.py              ← narrow public API
│
├── findings.py              ← DiagnosticFinding, DiagnosticCode,
│                                DiagnosticSeverity, DiagnosticReport
├── policies.py              ← QueryAlignmentPolicy, NumericalHealthPolicy,
│                                ReplayHealthPolicy, ResourceHealthPolicy
├── models.py                ← Shared Pydantic result base types
├── registration.py          ← ProbeDefinition, register_probe, list_probes
├── serialization.py         ← persist_probe_result, load_probe_result
│
├── probes/                  ← Deep offline mechanism probes
│   ├── __init__.py
│   ├── tem_memory.py        ← MemoryProbeResult, produce/cue_recall
│   ├── tem_pathway.py       ← PathwayProbeResult
│   ├── tem_query_alignment.py ← QueryAlignmentResult
│   └── cue_recall.py        ← CueRecallProbeResult
│
├── assessments/             ← Convert probe/trace results → findings
│   ├── __init__.py
│   ├── tem_memory.py        ← assess_memory_probe(result, policy)
│   ├── tem_query.py         ← assess_query_alignment(result, policy)
│   ├── hrm_dynamics.py      ← assess_hrm_dynamics(metrics, policy)
│   └── trace_assessment.py  ← assess_traces(artifact_dir, policy)
│
├── checks/                  ← Lifecycle health checks
│   ├── __init__.py
│   ├── numerical.py         ← nan_*, exploding_gradients, etc.
│   ├── replay.py            ← carry_stale, halt_reactivation, etc.
│   ├── resource.py          ← CUDA memory, step-time regression
│   └── registry.py          ← register_health_check, run_health_checks
│
└── dynamics/                ← HRM latent-dynamics computation
    ├── __init__.py
    └── hrm.py               ← compute_hrm_dynamics_metrics(_from_trace)
```

### 6.2 What stays where

| Module                       | Location                           | Notes                                 |
| ---------------------------- | ---------------------------------- | ------------------------------------- |
| `TraceField`, `TraceSpec`    | `traces/`                          | Unchanged                             |
| `TraceObserver`, `TraceSink` | `traces/`                          | Unchanged                             |
| `TraceConsumer`              | `eval/consumers.py`                | Unchanged                             |
| Capture profile config       | `config/evaluation/recipes/*.toml` | Unchanged                             |
| Figure rendering             | `figures/`                         | Unchanged                             |
| Report-data package          | `reporting/`                       | Presentational layer                  |
| Markdown table formatting    | `reporting/diagnostics.py`         | Moved from `diagnostics/`             |
| HRM dynamics → report rows   | `reporting/derived.py`             | Moved from `diagnostics/reporting.py` |
| Notebook analysis            | `notebooks/`                       | Unchanged                             |

### 6.3 What moves

| Current location                           | New location                                | Reason                                |
| ------------------------------------------ | ------------------------------------------- | ------------------------------------- |
| `diagnostics/reporting.py`                 | `reporting/derived.py`                      | Presentation belongs in reporting     |
| `diagnostics/tem_memory_probe.py`          | `diagnostics/probes/tem_memory.py`          | Probes grouped                        |
| `diagnostics/tem_pathway_probe.py`         | `diagnostics/probes/tem_pathway.py`         | Probes grouped                        |
| `diagnostics/tem_query_alignment_probe.py` | `diagnostics/probes/tem_query_alignment.py` | Probes grouped                        |
| `diagnostics/cue_recall_probe.py`          | `diagnostics/probes/cue_recall.py`          | Probes grouped                        |
| `diagnostics/hrm_dynamics.py`              | `diagnostics/dynamics/hrm.py`               | Computation separate from assessment  |
| `_classify_diagnosis()` in probes          | `diagnostics/assessments/tem_query.py`      | Assessment decoupled from measurement |
| `format_*_probe_table()`                   | `reporting/diagnostics.py`                  | Presentation not diagnostics          |

### 6.4 Package-level public API

```python
# ehp_sn/diagnostics/__init__.py

from .findings import (
    DiagnosticCode,
    DiagnosticFinding,
    DiagnosticReport,
    DiagnosticSeverity,
    summary_report,
)
from .registration import (
    ProbeDefinition,
    list_probes,
    register_probe,
)
from .serialization import (
    load_probe_result,
    persist_probe_result,
)
from .policies import (
    NumericalHealthPolicy,
    QueryAlignmentPolicy,
    ReplayHealthPolicy,
    ResourceHealthPolicy,
)

__all__ = [
    # Findings and severity
    "DiagnosticCode",
    "DiagnosticFinding",
    "DiagnosticReport",
    "DiagnosticSeverity",
    "summary_report",
    # Probe infrastructure
    "ProbeDefinition",
    "list_probes",
    "register_probe",
    # Serialization
    "load_probe_result",
    "persist_probe_result",
    # Policies
    "NumericalHealthPolicy",
    "QueryAlignmentPolicy",
    "ReplayHealthPolicy",
    "ResourceHealthPolicy",
]
```

Domain-specific APIs remain scoped:

```python
# Specific probes
from ehp_sn.diagnostics.probes import (
    produce_tem_memory_probe,
    load_tem_memory_probe,
    MemoryProbeResult,
)

# Specific assessments
from ehp_sn.diagnostics.assessments import (
    assess_query_alignment,
    assess_hrm_dynamics,
)

# Health checks
from ehp_sn.diagnostics.checks import (
    run_health_checks,
)
```

---

## 7. Complete diagnostic contract map

| Concern                                | Module                                        | Input                                 | Output                                  |
| -------------------------------------- | --------------------------------------------- | ------------------------------------- | --------------------------------------- |
| **Core contracts**                     |                                               |                                       |                                         |
| Findings, severity, codes              | `diagnostics/findings.py`                     | —                                     | `DiagnosticFinding`, `DiagnosticReport` |
| Probe definition                       | `diagnostics/registration.py`                 | `ProbeDefinition`                     | Registry entry                          |
| Generic persistence                    | `diagnostics/serialization.py`                | `BaseModel`                           | JSON file                               |
| Policies                               | `diagnostics/policies.py`                     | —                                     | Policy dataclasses                      |
| **Offline probes**                     |                                               |                                       |                                         |
| Hebbian memory self-retrieval          | `diagnostics/probes/tem_memory.py`            | `TEMModelV1`, `TEMInputV1`            | `MemoryProbeResult`                     |
| Place-code-to-logit pathway            | `diagnostics/probes/tem_pathway.py`           | `TEMModelV1`, `TEMInputV1`, `obs_ids` | `PathwayProbeResult`                    |
| Grid-query alignment                   | `diagnostics/probes/tem_query_alignment.py`   | `TEMModelV1`, `TEMInputV1`            | `QueryAlignmentResult`                  |
| HRM latent dynamics computation        | `diagnostics/dynamics/hrm.py`                 | `np.ndarray (T,B,S,D)`                | `HRMDynamicsMetrics`                    |
| HRM cue recall (block encoding)        | `diagnostics/probes/cue_recall.py`            | `HRModelV1`                           | `CueRecallProbeResult`                  |
| **Assessments**                        |                                               |                                       |                                         |
| TEM memory → findings                  | `diagnostics/assessments/tem_memory.py`       | `MemoryProbeResult`, `Policy`         | `list[DiagnosticFinding]`               |
| Query alignment → findings             | `diagnostics/assessments/tem_query.py`        | `QueryAlignmentResult`, `Policy`      | `list[DiagnosticFinding]`               |
| HRM dynamics → findings                | `diagnostics/assessments/hrm_dynamics.py`     | `HRMDynamicsMetrics`, `Policy`        | `list[DiagnosticFinding]`               |
| Traces → findings                      | `diagnostics/assessments/trace_assessment.py` | Artifact directory, `Policy`          | `list[DiagnosticFinding]`               |
| **Health checks**                      |                                               |                                       |                                         |
| Numerical checks                       | `diagnostics/checks/numerical.py`             | Lifecycle context, `Policy`           | `list[DiagnosticFinding]`               |
| Replay/carry checks                    | `diagnostics/checks/replay.py`                | Lifecycle context, `Policy`           | `list[DiagnosticFinding]`               |
| Resource checks                        | `diagnostics/checks/resource.py`              | Lifecycle context, `Policy`           | `list[DiagnosticFinding]`               |
| Check orchestration                    | `diagnostics/checks/registry.py`              | Hook name, context                    | `DiagnosticReport`                      |
| **Trace capture**                      |                                               |                                       |                                         |
| Trace field definitions                | `traces/specs.py`                             | Paradigm name                         | `TraceSpec`                             |
| Trace key constants                    | `traces/keys.py`                              | —                                     | String constants                        |
| Trace observer                         | `traces/observer.py`                          | `StepContext`                         | Per-step values                         |
| Trace persistence                      | `traces/sink.py`                              | Value stream                          | Zarr / Parquet / in-memory              |
| Trace→evaluation bridge                | `eval/consumers.py`                           | `StepContext`                         | `ProducedArtifact`                      |
| **Presentation** (outside diagnostics) |                                               |                                       |                                         |
| Markdown table formatting              | `reporting/diagnostics.py`                    | `BaseModel`                           | `str`                                   |
| Derived resources (pathway_metrics)    | `reporting/derived.py`                        | `RegimeArtifactSet`                   | `pd.DataFrame`                          |
| Figure rendering                       | `figures/`                                    | Figure name + context                 | Rendered image                          |
| **Configuration**                      |                                               |                                       |                                         |
| Capture profile selection              | `config/evaluation/recipes/*.toml`            | `[capture]` section                   | `CaptureConfig`                         |
| Probe policy overrides                 | `config/training/*.toml`                      | `[diagnostics.policy]`                | Policy model                            |
| Health check enable/disable            | `config/training/*.toml`                      | `[diagnostics.checks]`                | `list[str]`                             |

---

## 8. Probe vs trace vs check — when to use each

| Scenario                                                   | Probe | Health Check | Trace |
| ---------------------------------------------------------- | ----- | ------------ | ----- |
| You want to know if M self-retrieves                       | ✅    | ❌           | ❌    |
| You want to find where content is lost in the decode chain | ✅    | ❌           | ❌    |
| You want to know if mec_to_hpc(g) indexes memory           | ✅    | ❌           | ❌    |
| You want to detect NaN in gradients during training        | ❌    | ✅           | ❌    |
| You want to monitor CUDA memory growth                     | ❌    | ✅           | ❌    |
| You want to detect stale carry state                       | ❌    | ✅           | ❌    |
| You want to compare multi-case aggregated z_H norms        | ❌    | ❌           | ✅    |
| You want to visualise place fields across test episodes    | ❌    | ❌           | ✅    |
| You want to test a new architectural variant for WM recall | ✅    | ❌           | ❌    |
| You want to enforce CI/CD quality gates                    | ❌    | ✅           | ❌    |

---

## 9. Adding a new diagnostic capability

### 9.1 Adding a new probe

1. Create `diagnostics/probes/<name>.py` with:
   - A Pydantic result schema (`BaseModel`, frozen, `extra="forbid"`)
   - A `produce_*_probe(model, ...) → Result` function
   - A `ProbeDefinition` and call to `register_probe()`

2. Create `diagnostics/assessments/<name>.py` with:
   - An `assess_*_probe(result, policy) → list[DiagnosticFinding]` function

3. Add presentation formatting (if needed) in `reporting/diagnostics.py`.

### 9.2 Adding a new health check

1. Create a check function in `diagnostics/checks/<domain>.py`:

```python
def check_nan_activations(
    context: Any,
    policy: NumericalHealthPolicy,
) -> list[DiagnosticFinding]:
    # context.activations, context.module, etc.
    ...
```

2. Register it:

```python
register_health_check(
    HealthCheckDefinition(
        name="nan_activations",
        hook="after_forward",
        check=check_nan_activations,
    )
)
```

### 9.3 Adding a new trace field

1. Define the field key constant in `traces/keys.py`.
2. Write a getter function in `traces/specs.py`.
3. Register the `TraceField` and add to the paradigm's `*_TRACE_FIELDS` tuple.
4. Optionally add the field to the `"diagnostic"` capture profile.

---

## 10. Ownership summary

| Concern                                    | Owner                                            |
| ------------------------------------------ | ------------------------------------------------ |
| Diagnostic codes, severity enum            | `diagnostics/findings.py`                        |
| Diagnostic findings, reports               | `diagnostics/findings.py`                        |
| Assessment policies                        | `diagnostics/policies.py`                        |
| Probe definitions and registration         | `diagnostics/registration.py`                    |
| Generic probe serialization                | `diagnostics/serialization.py`                   |
| Probe result schemas                       | `diagnostics/probes/`                            |
| Probe logic (run model, extract internals) | `diagnostics/probes/`                            |
| Probe → findings assessment                | `diagnostics/assessments/`                       |
| Latent dynamics computation                | `diagnostics/dynamics/`                          |
| Numerical health checks                    | `diagnostics/checks/numerical.py`                |
| Replay/carry health checks                 | `diagnostics/checks/replay.py`                   |
| Resource health checks                     | `diagnostics/checks/resource.py`                 |
| Health check orchestration                 | `diagnostics/checks/registry.py`                 |
| Trace field definitions                    | `traces/specs.py`                                |
| Trace key constants                        | `traces/keys.py`                                 |
| Trace observer / sink                      | `traces/observer.py`, `traces/sink.py`           |
| Trace → evaluation bridge                  | `eval/consumers.py`                              |
| Capture profile config                     | `config/evaluation/recipes/*.toml`               |
| Report request config                      | `config/reporting/*.toml`                        |
| Report-data package preparation            | `reporting/`                                     |
| Derived resource construction              | `reporting/derived.py` + `tasks/*/inspection.py` |
| Probe/check result presentation            | `reporting/diagnostics.py`                       |
| Figure rendering                           | `figures/`                                       |
| Notebook analysis                          | `notebooks/`                                     |
| Lightning health callback                  | `lightning/` (imports from `diagnostics/`)       |

---

## 11. Key design rules

1. **Three distinct paths.** Probes, health checks, and traces serve
   different questions, triggers, and consumers. Do not force them into
   one abstraction.

2. **Findings, not free-form strings.** Every diagnostic observation is a
   `DiagnosticFinding` with a semantic code, severity, and numeric
   evidence. Consumers filter and escalate without knowing the producer.

3. **Policy is separate from measurement.** Thresholds live in policy
   dataclasses, not inside `_classify_diagnosis`. The same probe
   measurement can be assessed under different policies.

4. **Probes produce typed results; assessments produce findings.** The
   probe measures; the assessment interprets. A probe result is scientific
   data; a finding is an operational signal.

5. **Presentation belongs in reporting.** `format_*_table()` and
   report-row construction live in `reporting/`, not `diagnostics/`.

6. **Trace fields are defined once.** `traces/keys.py` is the single
   source of truth for all trace path strings.

7. **Trace fields are paradigm-scoped.** TEM and HRM fields live in
   separate `*_TRACE_FIELDS` tuples.

8. **Health checks are lifecycle-aware.** Each check declares which
   hook it runs on and receives hook-appropriate context.

9. **Generic persistence.** One `persist_probe_result` / `load_probe_result`
   pair serves all probes. Probe-specific wrappers are optional convenience.

10. **Diagnostics do not import training or Lightning.** Checks are
    imported by a Lightning callback in `lightning/`, not the reverse.
