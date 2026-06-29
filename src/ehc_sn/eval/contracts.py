"""Public contracts for top-level replay evaluation execution.

Task packages implement provider types against this contract. Lightning model
families expose ``execute_evaluation_batch`` against this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import torch

from ehc_sn.contracts.dependencies import Dependency
from ehc_sn.rollouts.materialization import EvaluatedChunk, ObservedStep
from ehc_sn.traces.observer import StepContext, TraceSpec
from ehc_sn.traces.trace_tree import TraceTree
from ehc_sn.types import Batch, JsonValue


# =============================================================================
@dataclass(frozen=True)
class EvaluationCaseBatch:
    """One provider-owned evaluation batch and associated source context.

    ``batch`` contains one or more dataset samples (``n_samples >= 1``).
    The artifact persists one manifest *case* per batch, which may
    represent multiple samples when ``batch_size > 1``.
    """

    batch: Batch
    case_id: str
    n_samples: int = 1
    source_context: object | None = None


# =============================================================================
# Phase 2 — typed artifact vocabulary
# =============================================================================


class ArtifactKind(StrEnum):
    TRACE = "trace"
    AGGREGATE = "aggregate"
    ANALYSIS = "analysis"
    PROBE = "probe"
    METRIC_TABLE = "metric_table"


@dataclass(frozen=True, order=True)
class ArtifactKey:
    """Identifies a produced artifact by kind and name.

    ``manifest_key()`` returns the canonical dict key for artifact
    manifests (e.g. ``"aggregate/mec_spatial_population"``).
    """

    kind: ArtifactKind
    name: str

    def manifest_key(self) -> str:
        return f"{self.kind.value}/{self.name}"


@dataclass(frozen=True)
class ArtifactRequirement:
    """Declares that a figure or analysis requires a specific artifact."""

    key: ArtifactKey
    schema_version: int = 1


@dataclass(frozen=True)
class ProducedArtifact:
    """One named, typed artifact produced by an evaluation consumer or analysis runner.

    All fields are frozen and required except ``metadata``.  The
    ``producer_digest`` identifies the parameters and implementation that
    created the artifact; ``content_digest`` identifies the payload bytes.
    """

    key: ArtifactKey
    schema_version: int
    path: Path
    media_type: str
    producer_digest: str
    content_digest: str
    source_path: Path | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


# =============================================================================
# Phase 2 — run/case context carriers
# =============================================================================


@dataclass(frozen=True)
class EvaluationRunContext:
    """Context for one evaluation run, provided to consumers via ``begin_run()``."""

    run_id: str
    checkpoint_sha256: str
    dataset_identity: object | None = None


@dataclass(frozen=True)
class EvaluationCaseContext:
    """Context for one evaluation case, provided to consumers via ``begin_case()``."""

    case_id: str
    batch_size: int = 1


@dataclass(frozen=True)
class AggregateBuildContext:
    """Context passed to an ``AggregateSpec.factory``.

    Provides all runtime resources needed to construct an
    ``EvaluationConsumer``: the pre-resolved experiment, the run context,
    the output root path, and the device.
    """

    experiment: Any
    run: EvaluationRunContext
    output_root: Path
    device: torch.device


# =============================================================================
# Phase 2 — EvaluationConsumer (replaces the old Protocol + MergeableAccumulator)
# =============================================================================


class EvaluationConsumer(ABC):
    """Run-scoped evaluation consumer with lifecycle hooks.

    Subclasses must call ``super().__init__(name=name)`` with a non-empty
    string.  ``dependencies`` declares what model views, record fields,
    and run metadata the consumer requires.
    """

    def __init__(self, *, name: str) -> None:
        normalized = name.strip()
        if not normalized:
            raise ValueError("EvaluationConsumer.name must be non-empty.")
        self._name = normalized

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def dependencies(self) -> frozenset[Dependency]: ...

    # -- Lifecycle -----------------------------------------------------------

    def begin_run(self, context: EvaluationRunContext) -> None:
        pass

    def begin_case(self, context: EvaluationCaseContext) -> None:
        pass

    @abstractmethod
    def update(self, context: StepContext) -> None: ...

    def end_case(self, context: EvaluationCaseContext) -> None:
        pass

    @abstractmethod
    def finalize(self) -> tuple[ProducedArtifact, ...]: ...

    def close(self) -> None:
        pass


# =============================================================================
@dataclass(frozen=True)
class EvaluationTraceRequest:
    """Trace materialization request for an evaluation batch execution."""

    trace_spec: TraceSpec
    trace_meta: Mapping[str, object] | None = None


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Result payload returned by ``execute_evaluation_batch``.

    Represents one evaluated provider batch.  Contains outputs and
    optional trace trees for ``n_samples`` dataset samples.
    """

    case_id: str
    evaluated: EvaluatedChunk
    n_samples: int = 1
    source_context: object | None = None
    trace: TraceTree | None = None
    consumer_results: dict[str, object] = field(default_factory=dict)


# =============================================================================
@dataclass(frozen=True)
class EvaluationRegimeResult:
    """Aggregate result payload for one named evaluation regime."""

    regime_id: str
    case_results: tuple[EvaluationCaseResult, ...]
    summary: Mapping[str, object]


# =============================================================================
class EvaluationSourceProvider(Protocol):
    """Task-owned provider that yields replay case batches for one regime."""

    def provide_cases(
        self,
        *,
        max_batches: int = 0,
        max_samples: int | None = None,
    ) -> Iterator[EvaluationCaseBatch]:
        """Yield replay case batches in deterministic provider-owned order.

        Args:
            max_batches: If > 0, yield at most this many batches.
            max_samples: If not None, yield at most this many samples
                (summed across batches).  ``max_batches`` takes precedence
                if both are set.
        """


# =============================================================================
class EvaluationExecutor(Protocol):
    """Family-facing execution seam consumed by future evaluation callbacks."""

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        case: EvaluationCaseBatch,
        *,
        consumers: Sequence[EvaluationConsumer] = (),
        trace_request: EvaluationTraceRequest | None = None,
        metric_observer: Callable[[ObservedStep], None] | None = None,
    ) -> EvaluationCaseResult:
        """Execute one provider case and return objective-scored outputs.

        Args:
            case: Provider-owned case batch.
            consumers: Run-scoped evaluation consumers that receive per-case
                lifecycle events and observe each rollout step.  The owning
                evaluation runner finalizes them after all cases complete.
            trace_request: Optional trace capture specification.
            metric_observer: Optional ephemeral per-step observer for live
                metric updates.  Has no lifecycle or persistence responsibility.
        """


# =============================================================================
@dataclass(frozen=True)
class ProviderSpec:
    """Provider specification — not a resolved provider instance.

    Providers may own file handles, datasets, or device-local resources.
    Instantiation is deferred to the runner, close to execution time.

    ``batch_size`` is a first-class field separated from ``settings`` for
    config-level validation.  It is forwarded to the provider constructor
    alongside ``settings`` at resolution time.
    """

    ref: str
    settings: dict[str, Any]
    batch_size: int = 1


@dataclass(frozen=True)
class EvaluationIdentity:
    """Stable artifact-identity fields for one evaluated experiment.

    Every field is required — no optional fallbacks.  Downstream consumers
    (reporting, figure selectors, inspection) use this identity for
    discoverability and compatibility checks.

    Note
    ----
    This is the *runtime* identity attached to an ``EvaluationExperiment``
    and used by inspection.  A separate ``EvaluationIdentity`` exists in
    ``eval.artifact_models`` for the *persisted manifest* with a different
    field set (``evaluation_id``, ``alias``, ``task``, ``model_family``).
    """

    task: str
    model_family: str
    trace_paradigm: str


@dataclass(frozen=True)
class EvaluationExperiment:
    """Complete resolved evaluation experiment returned by a builder.

    The generic runner receives this as a single typed input.  All provider,
    regime, trace, and identity resolution has already been performed by the
    experiment-specific builder.
    """

    executor: Any
    provider_spec: ProviderSpec
    regime_id: str
    regime_kind: str
    trace_spec: TraceSpec | None = None
    capture_profile: str = "metrics_only"
    capture_max_cases: int | None = None
    identity: EvaluationIdentity | None = None


# =============================================================================
__all__ = [
    "AggregateBuildContext",
    "ArtifactKey",
    "ArtifactKind",
    "ArtifactRequirement",
    "EvaluationCaseContext",
    "EvaluationConsumer",
    "EvaluationExperiment",
    "EvaluationIdentity",
    "EvaluationRunContext",
    "ProducedArtifact",
    "EvaluationCaseResult",
    "EvaluationCaseBatch",
    "EvaluationExecutor",
    "EvaluationRegimeResult",
    "EvaluationSourceProvider",
    "EvaluationTraceRequest",
    "ProviderSpec",
]
