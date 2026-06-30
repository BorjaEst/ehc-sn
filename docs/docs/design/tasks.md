---
title: Tasks Design Contract
description: Domain layer for ehp-sn tasks — task identity, semantic contracts, target derivation, and evaluation semantics
---

# Tasks Design Contract (`ehc_sn.tasks`)

> The task-domain layer of the repository: **what problem is being solved,
> what constitutes a valid task instance, what information is exposed to the
> learner, what reference outputs or semantic signals exist, and how success
> is interpreted.**

The task package is the semantic foundation of every experiment. It must
remain independent of model architecture, controller implementation, training
loops, optimizer configuration, rollout scheduling, metric accumulation,
reporting, and corpus lifecycle management.

---

## 1. Architectural position

```
ehc_sn.contracts                    (cross-package protocols, runtime DTOs)
        ^
ehc_sn.tasks                        (this package — task-domain semantics)
        ^
        ├── ehc_sn.adapters         (task→model translation)
        ├── ehc_sn.objectives       (differentiable scoring against targets)
        ├── ehc_sn.metrics          (stateful measurement)
        ├── ehc_sn.evaluation       (benchmark orchestration)
        ├── ehc_sn.rollouts         (temporal execution)
        └── ehc_sn.controllers      (control transitions)
```

### 1.1 Dependency rules

`ehc_sn.tasks` shared core may depend on:

| Dependency          | Rationale                             |
| ------------------- | ------------------------------------- |
| Python standard lib | —                                     |
| `numpy`             | Corpus builder and oracle math        |
| `torch`             | Tensor-valued contracts and schemas   |
| `ehc_sn.types`      | Shared type aliases (`Batch`)         |
| `ehc_sn.contracts`  | Cross-package protocols (lightweight) |

`ehc_sn.tasks` shared core must **not** depend on:

| Forbidden dependency | Rationale                                   |
| -------------------- | ------------------------------------------- |
| `ehc_sn.models`      | Tasks must be model-agnostic                |
| `ehc_sn.controllers` | Controllers consume tasks, not vice versa   |
| `ehc_sn.objectives`  | Objectives consume task targets             |
| `ehc_sn.metrics`     | Metrics measure task results (see §2.1)     |
| `ehc_sn.rollouts`    | Rollouts consume task runtimes              |
| `ehc_sn.training`    | Training imports targets, not task identity |
| `ehc_sn.reporting`   | Reporting consumes evaluated results        |
| `ehc_sn.figures`     | Figures consume evaluated results           |
| `lightning`          | Lightning is a training-layer concern       |
| `mlflow`             | Experiment tracking is infrastructure       |

Per-family integration modules may depend outward in controlled ways:

| Module         | May depend on                    |
| -------------- | -------------------------------- |
| `builder.py`   | `ehc_sn.data` (corpus I/O)       |
| `providers.py` | `ehc_sn.eval.contracts`          |
| `traces.py`    | `ehc_sn.traces` contracts        |
| `runtime.py`   | `ehc_sn.contracts` (TaskRuntime) |

These are integration boundaries, not core task semantics.

### 1.2 Task identity namespace

The stable task namespace is **`ehp`** — the project's intended branding —
not the current Python import namespace `ehc_sn`. A task key should survive
package renaming:

```python
ARENA_KEY = TaskKey(namespace="ehp", family="arena", variant="structural")
```

while imports remain:

```python
from ehc_sn.tasks import ...
```

### 1.3 Relationship to existing contracts

The repository already defines three execution-time contracts. Their
relationship to task definitions is:

```
tasks.protocols                 — semantic definition capabilities
    TaskDefinition, InputProjector, TargetProjector, ...

contracts.task_runtime           — execution-time value/control integration
    TaskRuntime, RuntimeReset, StepFeedback

contracts.task_step              — deliberation-step evaluation seam
    TaskStepEvaluator, StepEvaluation

contracts.task_environment       — external environment/runtime adapter
    TaskEnvironmentAdapter
```

A `TaskDefinition` says _what the problem is_. A `TaskRuntime` or
`TaskStepEvaluator` says _how to bind that meaning into an execution context_.

---

## 2. Current state assessment

The repository already respects the core architectural rule: tasks own
semantic contracts, not models, objectives, or training loops. The
dependency direction is correct — `tasks/` never imports `models/`,
`controllers/`, `objectives/`, `rollouts/`, `lightning/`, or `mlflow`.

What exists and is correct:

| Element                                    | Assessment                        |
| ------------------------------------------ | --------------------------------- |
| Per-family vertical modules                | ✅ Cohesive domain structure      |
| `contracts.py` typed dataclasses           | ✅ Clean, model-agnostic          |
| `runtime.py` batch keys + extractors       | ✅ Clean boundary                 |
| `evaluation.py` MetricSpec + ScoreReport   | ✅ Correct ownership              |
| `supervision.py` → learning structs        | ✅ Correct placement              |
| `providers.py` → EvaluationSourceProvider  | ✅ Correct seam                   |
| `reward.py` stateless projectors           | ✅ Correct pattern                |
| `capabilities/replay.py` execution binding | ✅ Correct separation             |
| `scoring.py` central aggregation           | ✅ Correct — can evolve           |
| Channel group constants (routebind)        | ✅ Excellent information boundary |

What is needed but missing:

| Gap                                      | Impact                                   |
| ---------------------------------------- | ---------------------------------------- |
| No structured task identity              | Task identity is raw strings             |
| No self-describing task metadata         | Metadata scattered across modules        |
| No shared task definition protocol       | Conventions are per-family, not enforced |
| Target construction in `builder.py`      | Target semantics mixed with I/O          |
| `ValidationRuntimeConfig` in `training/` | Tasks import training config             |

### 2.1 Scoring metadata placement

`MetricSpec` and `TaskScoringSpec` currently live in `ehc_sn.metrics.spec`.
They are declarative DTOs with no metric implementation dependency — no
stateful accumulation, no distributed reduction. This makes them lightweight
enough to live in a neutral contracts location:

```
ehc_sn.contracts.scoring
```

This is the recommended home. Alternatively, if the declarative DTOs are kept
in `metrics`, task scoring declarations should reference them through
`contracts` rather than importing `metrics` directly. The canonical
`TaskScoringSpec` aggregation remains in `scoring.py` as a migration bridge.

---

## 3. Top-level package structure

```
src/ehc_sn/tasks/
├── __init__.py         ← Public API — minimal stable surface
├── identity.py         ← TaskKey, TaskMode, TaskTrait
├── specs.py            ← TaskSpec — immutable metadata
├── protocols.py        ← TaskDefinition, InputProjector, TargetProjector, …
├── catalog.py          ← Immutable built-in task catalog
├── errors.py           ← TaskError hierarchy
├── scoring.py          ← Legacy string→scoring bridge
│
├── arena/              ← Structural navigation (replay-based)
├── mazehard/           ← Token prediction over maze layouts
├── goaltrace/          ← Goal-conditioned prospective field (DAG only)
├── routebind/          ← Goal-conditioned spatial prospective field
└── seqmaze/            ← Transition-graph edge-lookup & path-prediction
```

Small modules are combined where they share a single concept (`identity.py`
replaces separate `keys.py`, `kinds.py`, `capabilities.py`). Schemas and
validation are deferred until concrete consumers exist. The registry is an
immutable catalog, not a mutable global registry.

---

## 4. Shared submodules

### 4.1 `identity.py` — Task identity, mode, and traits

```python
from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, order=True, slots=True)
class TaskKey:
    namespace: str
    family: str
    variant: str | None = None

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("Task namespace must not be empty")
        if not self.family:
            raise ValueError("Task family must not be empty")

    def __str__(self) -> str:
        base = f"{self.namespace}:{self.family}"
        return base if self.variant is None else f"{base}/{self.variant}"


class TaskMode(StrEnum):
    """Dominant execution shape — mutually exclusive categories."""
    STATIC = "static"               # Single-step field prediction
    SEQUENTIAL = "sequential"       # Multi-step episodic (replay, unroll)
    INTERACTIVE = "interactive"     # Live environment stepping


class TaskTrait(StrEnum):
    """Orthogonal semantic properties — a task may have many."""
    VARIABLE_LENGTH = "variable_length"
    STRUCTURED_OUTPUT = "structured_output"
    DELIBERATIVE = "deliberative"
    REPLAYABLE = "replayable"
    MEMORY_CONDITIONED = "memory_conditioned"
```

`ONLINE` and `STOCHASTIC` are intentionally omitted. `ONLINE` overlaps with
`TaskMode.INTERACTIVE` (both describe live execution). `STOCHASTIC` is too
imprecise — a task may have stochastic transitions, observations, rewards,
or targets, and a single flag conflates them. Add precise traits later when
needed (e.g. `STOCHASTIC_TRANSITIONS`).

Design rules:

- `TaskMode` is mutually exclusive — one mode per task. `MEMORY_CONDITIONED`
  is a trait, not a mode, because a task can be memory-conditioned AND
  sequential or interactive.
- `TaskTrait` values describe semantic or operational properties not directly
  inferable from an interface. Method-presence concepts (`INPUT_DERIVATION`,
  `TARGET_DERIVATION`, `VALIDATION`, `ACTION_MASKING`, `INTERACTIVE_SESSION`,
  `MEMORY_ACCESS`) are removed — they duplicate structural typing and can
  disagree with the actual protocol implementation.
- `SEQUENTIAL` replaces `SEQUENCE` for consistency with `STATIC`.

Recommended classifications:

| Family      | Mode         | Traits                            |
| ----------- | ------------ | --------------------------------- |
| `arena`     | `SEQUENTIAL` | `VARIABLE_LENGTH`, `REPLAYABLE`   |
| `mazehard`  | `SEQUENTIAL` | `VARIABLE_LENGTH`, `DELIBERATIVE` |
| `goaltrace` | `STATIC`     | `STRUCTURED_OUTPUT`               |
| `routebind` | `STATIC`     | `STRUCTURED_OUTPUT`               |
| `seqmaze`   | `SEQUENTIAL` | `VARIABLE_LENGTH`, `DELIBERATIVE` |

### 4.2 `specs.py` — Immutable task metadata

```python
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TaskSpec:
    key: TaskKey
    schema_version: int
    mode: TaskMode
    traits: frozenset[TaskTrait] = frozenset()
    description: str = ""

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError(
                f"schema_version must be >= 1, got {self.schema_version}"
            )
```

`schema_version` versions the task's externally observable semantic
contract — inputs, targets, masks, and success semantics — not the
corpus storage format. Changing target construction (e.g. a decay
formula) increments this version. Extracting target code from
`builder.py` without changing output arrays does not. Corpus schemas
retain their own independent versions.

`TaskSpec` is purely semantic — it does not embed `TaskScoringSpec`.
Scoring metadata lives separately (see §4.3) to keep the dependency
direction clean. Schema descriptors (`TensorFieldSchema`, `StructSchema`)
are deferred until a concrete consumer justifies them.

Per-family example:

```python
# ehc_sn/tasks/goaltrace/spec.py

from ehc_sn.tasks.identity import TaskKey, TaskMode, TaskTrait
from ehc_sn.tasks.specs import TaskSpec

GOALTRACE_KEY = TaskKey(namespace="ehp", family="goaltrace")

GOALTRACE_SPEC = TaskSpec(
    key=GOALTRACE_KEY,
    schema_version=1,
    mode=TaskMode.STATIC,
    traits=frozenset({
        TaskTrait.STRUCTURED_OUTPUT,
    }),
    description="Goal-conditioned prospective field prediction over a DAG.",
)
```

### 4.3 Scoring metadata — separation from TaskSpec

`MetricSpec` and `TaskScoringSpec` are lightweight declarative DTOs.
The recommended home is a neutral contracts module so neither `tasks`
nor `metrics` imports the other:

```
ehc_sn.contracts.scoring
```

If kept in `ehc_sn.metrics.spec`, task modules should reference them
indirectly. The canonical aggregation lives in `scoring.py` as a migration
bridge:

```python
def scoring_spec_for_task_family(task_family: str) -> TaskScoringSpec:
    """Legacy: resolve scoring spec by raw family name."""
    ...

def scoring_spec_for_key(key: TaskKey) -> TaskScoringSpec:
    """Resolve scoring spec by task key."""
    ...
```

A separate join table owned by `evaluation` or a neutral contracts module
is cleaner than embedding scoring inside `TaskSpec`.

### 4.4 `protocols.py` — Segregated semantic protocols

Truly small, focused capability protocols. `TaskDefinition` is a metadata-only
base — it does not mandate input derivation, target derivation, or validation:

```python
from typing import Protocol, TypeVar

InstanceT = TypeVar("InstanceT", contravariant=True)
InputT = TypeVar("InputT", covariant=True)
TargetT = TypeVar("TargetT", covariant=True)
OutputT = TypeVar("OutputT", contravariant=True)
DecodedT = TypeVar("DecodedT", covariant=True)


class TaskDefinition(Protocol):
    """Every task exposes immutable metadata."""

    @property
    def spec(self) -> TaskSpec:
        ...


class InstanceValidator(Protocol[InstanceT]):
    """Task that can validate an instance."""

    def validate_instance(self, instance: InstanceT) -> None:
        ...


class InputProjector(Protocol[InstanceT, InputT]):
    """Task that derives model-facing inputs from an instance."""

    def make_inputs(self, instance: InstanceT) -> InputT:
        ...


class TargetProjector(Protocol[InstanceT, TargetT]):
    """Task that derives reference targets from an instance."""

    def make_targets(self, instance: InstanceT) -> TargetT:
        ...


class OutputDecoder(Protocol[OutputT, DecodedT]):
    """Task that interprets raw model outputs."""

    def decode_output(self, output: OutputT) -> DecodedT:
        ...


class ActionConstrainedTask(Protocol):
    """Task that provides legal-action masks."""

    def action_mask(self, state: object) -> object:
        ...
```

Consumers request only the protocols they need:

```python
def build_supervised_case(
    task: InputProjector[InstanceT, InputT],
    instance: InstanceT,
) -> InputT:
    return task.make_inputs(instance)
```

Families implement what applies:

- Static families (`goaltrace`, `routebind`): `TaskDefinition`, `InputProjector`, `TargetProjector`.
- Replay families (`arena`): `TaskDefinition`, `InstanceValidator`. Replay families
  may also implement `InputProjector` and `TargetProjector` when semantic
  derivation exists independently of the stored corpus representation.
- Interactive families: `TaskDefinition` only (no precomputed targets).
- Future memory families: add `TargetProjector` or use an oracle.

A family whose input is already the canonical instance does not implement
`InputProjector`. One whose targets arrive with instances does not implement
`TargetProjector`. A task with only a `TaskStepEvaluator` does not implement
`TargetProjector`.

### 4.5 `catalog.py` — Immutable built-in catalog

An immutable, startup-time catalog is clearer than a mutable global registry
for a research repository where all tasks are internal and statically known:

```python
from collections.abc import Mapping
from types import MappingProxyType

from ehc_sn.tasks.arena.spec import ARENA_KEY, ARENA_SPEC
from ehc_sn.tasks.goaltrace.spec import GOALTRACE_KEY, GOALTRACE_SPEC
from ehc_sn.tasks.mazehard.spec import MAZEHARD_KEY, MAZEHARD_SPEC
from ehc_sn.tasks.routebind.spec import ROUTEBIND_KEY, ROUTEBIND_SPEC
from ehc_sn.tasks.seqmaze.spec import SEQMAZE_KEY, SEQMAZE_SPEC


_TASK_SPECS: dict[TaskKey, TaskSpec] = {
    ARENA_KEY: ARENA_SPEC,
    GOALTRACE_KEY: GOALTRACE_SPEC,
    MAZEHARD_KEY: MAZEHARD_SPEC,
    ROUTEBIND_KEY: ROUTEBIND_SPEC,
    SEQMAZE_KEY: SEQMAZE_SPEC,
}

TASK_SPECS: Mapping[TaskKey, TaskSpec] = MappingProxyType(_TASK_SPECS)


def task_spec(key: TaskKey) -> TaskSpec:
    """Return the TaskSpec for *key*, or raise KeyError."""
    try:
        return TASK_SPECS[key]
    except KeyError as exc:
        raise TaskNotFoundError(key) from exc


def built_in_task_specs() -> tuple[TaskSpec, ...]:
    """Return all built-in task specifications."""
    return tuple(TASK_SPECS.values())
```

No lock is needed — the catalog is populated once at import time. Introduce
mutable registration only when third-party plugins are a concrete requirement.

### 4.6 `errors.py` — Typed exception hierarchy

```python
class TaskError(Exception):
    """Base exception for all task-domain errors."""

class TaskNotFoundError(TaskError):
    """Raised when a TaskKey is not in the catalog."""

class InvalidTaskSpecError(TaskError):
    """Raised when a TaskSpec fails validation."""

class InvalidTaskInstanceError(TaskError):
    """Raised when a task instance fails validation."""
```

---

## 5. Per-family structure

Each family includes only modules it needs. Not every module in the template
is required.

```
tasks/<family>/
├── __init__.py         ← Small public surface
├── contracts.py        ← Domain DTOs (frozen dataclasses) — always present
├── spec.py             ← FAMILY_KEY, FAMILY_SPEC constants
├── targets.py          ← Pure target construction — when targets exist
├── runtime.py          ← Batch → typed input/target extraction — always present
├── supervision.py      ← Target packaging for learning — when learning packaging exists
├── evaluation.py       ← MetricSpec, TaskScoringSpec, ScoreReport — always present
├── reward.py           ← Stateless reward projection — when reward exists
├── validation.py       ← Task-specific validity checks — always present
├── decoding.py         ← Output interpretation — when decoding exists
├── builder.py          ← Corpus materialization (I/O adapter)
├── providers.py        ← Evaluation case providers (integration)
├── environment.py      ← Interactive session — only for interactive tasks
└── capabilities/       ← Optional extensions
    ├── replay.py
    └── ...
```

**Responsibility-based, not template-based:**

| Module           | When required                                                            |
| ---------------- | ------------------------------------------------------------------------ |
| `contracts.py`   | Always — defines semantic value types                                    |
| `runtime.py`     | Expected for benchmark families — batch extraction and runtime constants |
| `evaluation.py`  | Expected for benchmark families — scoring specs and metric names         |
| `validation.py`  | Expected for benchmark families — instance and artifact validation       |
| `spec.py`        | When family metadata needs a canonical home                              |
| `targets.py`     | When precomputed targets exist (not for interactive-only)                |
| `reward.py`      | When task has reward semantics                                           |
| `decoding.py`    | When output interpretation logic exists                                  |
| `builder.py`     | When corpus materialization is needed                                    |
| `providers.py`   | When evaluation case providers exist                                     |
| `supervision.py` | When learning packaging exists                                           |
| `environment.py` | Only for genuinely interactive tasks                                     |
| `capabilities/`  | Only when optional extensions justify it                                 |

### 5.1 `contracts.py` — Domain DTOs

Owns task-domain value types. Rules: frozen dataclasses, no model imports, no
filesystem I/O, no registry mutation, no training configuration, no metric
state.

### 5.2 `spec.py` — Canonical family metadata

Owns the family's `FAMILY_KEY` and `FAMILY_SPEC` constants. When these are
a handful of lines, they may live in `__init__.py` or `contracts.py`. A
separate file is justified when it breaks dependency cycles or owns
substantial metadata.

### 5.3 `targets.py` — Pure target construction

The highest-value extraction from existing `builder.py` modules. Target
functions should be pure — no filesystem access, manifests, model invocation,
or weighted loss computation.

### 5.4 `runtime.py` — Runtime translation boundary

Batch key constants, `Batch` → typed input/target extraction, runtime-specific
shape normalization, compatibility with `TaskRuntime` and `TaskStepEvaluator`.

### 5.5 `builder.py` — Corpus materialization adapter

A corpus I/O adapter, not the primary owner of target semantics. Imports
`targets.py` and `validation.py` from the same family.

### 5.6 `definition.py` — Optional task definition object

Omit unless a family benefits from wrapping functions in an object. Many
current task families are naturally functional. Do not create `definition.py`
solely to satisfy a registry or protocol.

### 5.7 Integration modules

`providers.py`, `traces.py`, `diagnostics.py`, and `inspection.py` are
integration modules, not core task semantics. When they exist, a useful
enforcement rule: core modules within the family must not import from
integration modules. If integration modules proliferate, co-locate them:

```
tasks/<family>/integrations/
├── evaluation.py
├── traces.py
└── corpus.py
```

---

## 6. Family structure per current task

### Arena

```
arena/
├── contracts.py            ← ArenaTaskInput, ArenaTargets, ArenaTaskOutput
├── spec.py                 ← ARENA_KEY, ARENA_SPEC
├── targets.py              ← Revisit derivation, ancestral/path target construction
├── runtime.py              ← Batch keys, extract_arena_task_input, coerce_arena_targets
├── supervision.py          ← ArenaTEMSupervision, build_arena_observation_metrics
├── evaluation.py           ← ArenaScoreReport, ArenaStepScore, MetricSpecs
├── reward.py               ← ArenaRewardProjector
├── validation.py           ← validate_arena_task_sample, validate_stored_sample
├── builder.py              ← Corpus materialization (uses targets.py)
├── providers.py            ← ArenaReplayProvider, ArenaFixedProbeProvider
└── capabilities/
    └── replay.py           ← ArenaReplayCapability
```

Canonical primary metric: `accuracy_ancestral_revisit`.

### MazeHard

```
mazehard/
├── contracts.py            ← MazeHardTaskInput, MazeHardTargets, MazeHardTaskOutput
├── spec.py                 ← MAZEHARD_KEY, MAZEHARD_SPEC
├── runtime.py              ← Batch keys, extract_maze_hard_task_input, TaskRuntime
├── supervision.py          ← MazeHardTokenSupervision, build_mazehard_weights
├── evaluation.py           ← MazeHardScoreReport, MazeHardStepScore, MetricSpecs
├── reward.py               ← MazeHardRewardProjector
├── validation.py           ← validate_all_samples, validate_stored_sample
├── builder.py              ← Corpus materialization
├── providers.py            ← MazeHardReplayProvider, MazeHardFixedProbeProvider
└── evaluators/
    └── step.py             ← MazeHardStepEvaluator (TaskStepEvaluator impl)
```

Canonical primary metric: `sequences_exact`.

### Goaltrace

```
goaltrace/
├── contracts.py            ← GoalTraceTaskInput, GoalTraceTargets, GoalTraceTaskOutput
├── spec.py                 ← GOALTRACE_KEY, GOALTRACE_SPEC
├── targets.py              ← Distance/reachability computation, activation field
├── runtime.py              ← Batch keys, extract_goaltrace_task_input/targets
├── evaluation.py           ← GoaltraceStepScore, field MSE, decay correlation, MetricSpecs
├── validation.py           ← validate_goaltrace_instance
├── builder.py              ← Corpus materialization (uses targets.py)
└── providers.py            ← GoaltraceReplayProvider
```

### Routebind

Routebind is the reference implementation for channel grouping and target
separation:

```
routebind/
├── contracts.py            ← RoutebindTaskInput, RoutebindTargets, RoutebindTaskOutput
├── spec.py                 ← ROUTEBIND_KEY, ROUTEBIND_SPEC
├── schema.py               ← RoutebindCorpusSchema (channel groups: model, target, all)
├── targets.py              ← Oracle-result-to-target encoding (exists ✅)
├── runtime.py              ← Batch keys, extract_routebind_task_input/targets
├── supervision.py          ← RoutebindSupervision
├── evaluation.py           ← RoutebindStepScore, trajectory/waypoint metrics, MetricSpecs
├── oracle.py               ← Product-state 0-1 BFS oracle
├── decoding.py             ← Field → route extraction (exists ✅)
├── validation.py           ← Instance and channel validation
├── builder.py              ← Corpus materialization (uses targets.py, oracle.py)
└── _oracle_kernel.py       ← Numba kernel (private)
```

### SeqMaze

```
seqmaze/
├── contracts.py            ← SeqMazeTaskInput, SeqMazeTargets, SeqMazeProbeInput, …
├── spec.py                 ← SEQMAZE_KEY, SEQMAZE_SPEC
├── targets.py              ← Path encoding, edge label construction
├── runtime.py              ← Batch keys, SeqMazeRuntime (TaskRuntime), extract_* helpers
├── supervision.py          ← SeqMazeTokenSupervision
├── evaluation.py           ← SeqMazeScoreReport, SeqMazeStepScore, MetricSpecs
├── reward.py               ← SeqMazeRewardProjector
├── validation.py           ← validate_all_samples, validate_stored_sample
├── builder.py              ← Corpus materialization (uses targets.py)
└── providers.py            ← SeqMazeReplayProvider
```

`SeqMazeRuntime` may continue implementing `TaskRuntime`, but the imported
`ValidationRuntimeConfig` must not come from `ehc_sn.training`. Move it to
`ehc_sn.contracts` or `ehc_sn.eval`.

---

## 7. Public API

### 7.1 Root package

Minimal stable surface for the first version — concepts expected to be used
throughout the repository:

```python
from .identity import TaskKey, TaskMode, TaskTrait
from .specs import TaskSpec
from .errors import TaskError, TaskNotFoundError, InvalidTaskSpecError
from .catalog import TASK_SPECS, task_spec, built_in_task_specs

__all__ = [
    "TaskKey",
    "TaskMode",
    "TaskSpec",
    "TaskTrait",
    "TaskError",
    "TaskNotFoundError",
    "InvalidTaskSpecError",
    "TASK_SPECS",
    "built_in_task_specs",
    "task_spec",
]
```

Protocols remain available through their submodule:

```python
from ehc_sn.tasks.protocols import TaskDefinition, InputProjector, TargetProjector
```

Registries and factories remain submodule APIs until multiple real consumers
require them.

### 7.2 Family public surface

Concrete task APIs are imported from family namespaces:

```python
from ehc_sn.tasks.goaltrace import (
    GOALTRACE_SPEC,
    GoalTraceInstance,
    GoalTraceTargets,
    GoalTraceTaskInput,
)
```

Each family's `__init__.py` exports a deliberately small surface. Do not
normally export: private graph algorithms, builder internals, corpus writers,
raw channel-key helpers, internal registry hooks, test fixtures, or
compatibility aliases unless intentionally supported.

---

## 8. Configurations

Configuration should exist only where task semantics are genuinely
configurable. Separate configuration categories:

| Category                     | Purpose                                 | Location                      |
| ---------------------------- | --------------------------------------- | ----------------------------- |
| Task configuration           | Changes what the task means             | `contracts.py` or `config.py` |
| Corpus builder configuration | Changes how instances are generated     | `builder.py`                  |
| Runtime configuration        | Changes execution limits or batching    | `runtime.py` or `contracts/`  |
| Objective configuration      | Changes optimization behavior           | `objectives/`                 |
| Evaluation configuration     | Changes benchmark selection/aggregation | `eval/`                       |

These should not be merged into one universal config per family.

Example separation for goaltrace:

```python
@dataclass(frozen=True, slots=True)
class GoalTraceTargetConfig:
    """Semantic target configuration — changes what the target means."""
    decay: float
    anchor_value: float = 1.0
    unreachable_value: float = 0.0
```

Huber weights or ranking-loss margins belong in objectives, not here.

---

## 9. Boundary corrections

### 9.1 `tasks -> training` import

`ValidationRuntimeConfig` must not live in `ehc_sn.training`. Move it to
`ehc_sn.contracts` (runtime execution configuration), `ehc_sn.eval`
(evaluation-only limits), or `ehc_sn.controllers` (controller-specific
settings).

### 9.2 `contracts -> controllers` import

`contracts/task_environment.py` imports `OnlineBootstrapCarry` from
`controllers`. A base contracts package must not depend on a concrete
controller-layer carry type. Use a `TypeVar` or move the generic carry DTO
into `contracts`.

### 9.3 `contracts -> tensordict` import

Acceptable only if `contracts.task_environment` is explicitly a TorchRL
integration contract. If `contracts` is meant to be framework-neutral,
move it to `ehc_sn.adapters.torchrl`.

---

## 10. Recommended migration path

```
Phase 1 — Shared types (additive, no breakage)
├── Add identity.py (TaskKey, TaskMode, TaskTrait)
├── Add specs.py (TaskSpec), errors.py, protocols.py
├── Add catalog.py (immutable built-in catalog)
├── Add per-family spec.py (FAMILY_KEY, FAMILY_SPEC)
├── scoring.py: add TaskKey bridge, keep string-based function
└── No existing code changes — pure addition

Phase 2 — Target extraction (highest practical payoff)
├── arena: extract revisit/target logic builder.py → targets.py
├── goaltrace: extract field encoding builder.py → targets.py
├── seqmaze: extract path encoding builder.py → targets.py
└── routebind: already has targets.py — reference

Phase 3 — Channel group standardization (incremental)
├── Add MODEL_INPUT_CHANNELS, TARGET_CHANNELS, ALL_CHANNELS per family
├── Standardize across arena, mazehard, goaltrace, seqmaze
└── Enforce information-boundary invariant:
    model-visible ∩ (targets ∪ evaluation_meta) = ∅

Phase 4 — Boundary cleanup (targeted fixes)
├── Move ValidationRuntimeConfig out of training/
├── Fix contracts/task_environment.py import direction
└── Remove tasks → training imports

Phase 5 — Schema formalization (when justified)
├── Add TensorFieldSchema / StructSchema only when introspection demanded
└── Use immutable mappings (MappingProxyType) inside frozen specs
```

---

## 11. Responsibility matrix

| Concern                            | Owner                                         |
| ---------------------------------- | --------------------------------------------- |
| Task identity                      | `tasks.identity`                              |
| Task mode                          | `tasks.identity`                              |
| Semantic traits                    | `tasks.identity`                              |
| Immutable task metadata            | `tasks.specs`                                 |
| Semantic protocols                 | `tasks.protocols`                             |
| Built-in task catalog              | `tasks.catalog`                               |
| Family-domain DTOs                 | `<family>/contracts.py`                       |
| Canonical family metadata          | `<family>/spec.py`                            |
| Channel and tensor schemas         | `<family>/schema.py` (deferred)               |
| Instance → input conversion        | `<family>/definition.py` or `runtime.py`      |
| Instance → target construction     | `<family>/targets.py`                         |
| Batch → typed runtime input        | `<family>/runtime.py`                         |
| Target packaging for learning      | `<family>/supervision.py`                     |
| Task-native scoring declarations   | `<family>/evaluation.py`                      |
| Task-semantic rewards              | `<family>/reward.py`                          |
| Output interpretation              | `<family>/decoding.py`                        |
| Task-specific validation           | `<family>/validation.py`                      |
| Corpus materialization             | `<family>/builder.py`                         |
| Interactive execution adapter      | `<family>/environment.py`                     |
| Replay or optional semantics       | `<family>/capabilities/`                      |
| Evaluation case providers          | `<family>/providers.py` (integration)         |
| Traces and supplements             | `<family>/traces.py` (integration)            |
| Datasets, storage, loading         | `ehc_sn.data`                                 |
| Rollout loops, recurrent unrolling | `ehc_sn.rollouts`                             |
| Losses, weights, regularization    | `ehc_sn.objectives`                           |
| Stateful metric computation        | `ehc_sn.metrics`                              |
| Scoring metadata DTOs              | `ehc_sn.contracts.scoring` (recommended)      |
|                                    | _(owns the MetricSpec/TaskScoringSpec types)_ |
| Task-native scoring declarations   | `<family>/evaluation.py`                      |
|                                    | _(owns concrete declarations for that task)_  |
| Benchmark orchestration            | `ehc_sn.eval`                                 |
| Controller behavior                | `ehc_sn.controllers`                          |
| Model computation                  | `ehc_sn.models`                               |
| Task-model translation             | `ehc_sn.adapters`                             |

The five task families and their canonical identities:

| Family    | Mode       | Primary metric                    | Definition style  |
| --------- | ---------- | --------------------------------- | ----------------- |
| arena     | SEQUENTIAL | `accuracy_ancestral_revisit`      | Replay-based      |
| mazehard  | SEQUENTIAL | `sequences_exact`                 | Token prediction  |
| goaltrace | STATIC     | `field_mse`                       | Single-step field |
| routebind | STATIC     | `balanced_trajectory_field_error` | Single-step field |
| seqmaze   | SEQUENTIAL | `sequence_exact`                  | Path prediction   |
