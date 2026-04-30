"""Shared evaluation regime contracts for Lightning training surfaces.

This module defines the public evaluation regime layer that separates
fit-path validation from named diagnostic evaluation runs. Four stable
contract parts live here:

- :class:`EvaluationTraceRequest` — which trace keys a regime needs.
- :class:`EvaluationScheduleSettings` — when the regime runs.
- :class:`EvaluationRegimeSettings` — one named regime's full config.
- :class:`EvaluationCaseBatch` — one unit of work from a provider.
- :class:`EvaluationBatchArtifacts` — one unit of output from the evaluation surface.
- :class:`EvaluationSourceProvider` — protocol for task- or probe-owned case sources.
- :class:`SupportsEvaluationRegimes` — protocol that all Lightning families implement.

Metric namespace rules (fixed, not negotiable):

- ``val/`` is owned exclusively by fit-path validation.
- Diagnostic regimes log under ``diag/<regime_id>/``.
- Benchmark regimes (future) log under ``bench/<regime_id>/``.
- No named regime may write into ``val/`` metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from torchmetrics import MetricCollection

from ehc_sn.traces.trace_tree import TraceTree
from ehc_sn.types import Batch


# =================================================================================================
class EvaluationTraceRequest(BaseModel, extra="forbid"):
    """Declares which trace keys a named evaluation regime needs.

    Attributes:
        enabled: Whether trace collection is active for this regime.
        trace_keys: Set of semantic trace key names the regime consumer requires.
    """

    enabled: bool = Field(default=False, description="Activate trace collection for this regime.")
    trace_keys: list[str] = Field(default_factory=list, description="Semantic trace key names required.")

    def key_set(self) -> set[str]:
        """Return trace keys as a set."""
        return set(self.trace_keys)


# =================================================================================================
class EvaluationScheduleSettings(BaseModel, extra="forbid"):
    """Controls when a named evaluation regime is triggered.

    Attributes:
        every_n_epochs: Run the regime every N epochs. ``0`` disables epoch scheduling.
        every_n_steps: Run the regime every N global steps. ``0`` disables step scheduling.
        max_batches: Maximum number of provider case batches to consume per run. ``0`` = unlimited.
    """

    every_n_epochs: int = Field(default=0, ge=0, description="Run every N epochs (0 = disabled).")
    every_n_steps: int = Field(default=0, ge=0, description="Run every N global steps (0 = disabled).")
    max_batches: int = Field(default=1, ge=0, description="Provider case batches to consume per run (0 = unlimited).")


# =================================================================================================
class EvaluationRegimeSettings(BaseModel, extra="forbid"):
    """Full configuration for one named evaluation regime.

    Attributes:
        regime_id: Unique identifier for this regime. Used as the metric namespace suffix.
        phase_kind: ``"diag"`` for diagnostic regimes; ``"bench"`` for benchmark regimes (future).
        provider_ref: Dotted import path to a provider factory or class.
        provider_settings: Arbitrary mapping forwarded to the provider factory.
        schedule: Scheduling settings.
        trace_request: Which traces the regime needs.
    """

    regime_id: str = Field(..., description="Unique regime identifier (used as metric namespace suffix).")
    phase_kind: Literal["diag", "bench"] = Field(default="diag", description="Phase kind: 'diag' or 'bench' (future).")
    provider_ref: str = Field(..., description="Dotted import path to a provider factory or class.")
    provider_settings: dict[str, Any] = Field(default_factory=dict, description="Forwarded to the provider factory.")
    schedule: EvaluationScheduleSettings = Field(default_factory=EvaluationScheduleSettings, description="Scheduling settings.")
    trace_request: EvaluationTraceRequest = Field(default_factory=EvaluationTraceRequest, description="Trace collection settings.")

    @property
    def metric_namespace(self) -> str:
        """Derive the metric namespace from phase kind and regime id.

        Returns:
            e.g. ``"diag/my_probe/"`` or ``"bench/b0/"``
        """
        return f"{self.phase_kind}/{self.regime_id}/"


# =================================================================================================
@dataclass
class EvaluationCaseBatch:
    """One unit of work produced by an :class:`EvaluationSourceProvider`.

    Trace capture is controlled at the regime level via
    :attr:`EvaluationRegimeSettings.trace_request`, not per-case.
    The regime runner passes the regime-level trace request to
    :meth:`~SupportsEvaluationRegimes.execute_evaluation_batch`.

    Attributes:
        batch: The raw task batch (same type as fit-path batches).
        case_id: Optional identifier for this case, for logging.
        metadata: Optional provider-specific metadata.
    """

    batch: Batch
    case_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


# =================================================================================================
@dataclass
class EvaluationBatchArtifacts:
    """Output of one :meth:`~SupportsEvaluationRegimes.execute_evaluation_batch` call.

    Attributes:
        regime_id: The regime that produced this artifact. Set to ``"_inline"`` by
            ``execute_evaluation_batch``; overwritten by the runner with the real regime id.
        metric_namespace: The logging namespace (e.g. ``"diag/my_probe/"``).
            Overwritten by the runner before logging.
        evaluated: The objective-scored rollout result (paradigm-specific type).
        apply_to_metrics: Optional callable that updates a
            :class:`~torchmetrics.MetricCollection` with this batch's results.
            Set by ``execute_evaluation_batch`` as a closure over the evaluated chunk
            and the family's route table; the runner calls it with the regime collection.
        trace: Optional trace tree for figure consumers.
        case_id: Forwarded from the :class:`EvaluationCaseBatch` if present.
        source_metadata: Provider-supplied metadata.
    """

    regime_id: str
    metric_namespace: str
    evaluated: Any  # EvaluatedChunk or paradigm equivalent
    apply_to_metrics: Optional[Callable[[MetricCollection], None]] = None
    trace: Optional[TraceTree] = None
    case_id: Optional[str] = None
    source_metadata: Optional[dict[str, Any]] = None


# =================================================================================================
@runtime_checkable
class EvaluationSourceProvider(Protocol):
    """Protocol for task- or probe-owned evaluation case sources.

    Implementors own:
        - where cases come from (corpus, probe fixture, environment)
        - what one case means

    Implementors do **not** own:
        - trainer scheduling
        - metric naming
        - figure rendering

    The provider protocol is intentionally general: it admits corpus-backed,
    probe-based, and environment-backed sources without forcing any particular
    data root convention.
    """

    def provide_cases(self, max_batches: int = 0) -> Iterator[EvaluationCaseBatch]:
        """Yield evaluation case batches.

        Args:
            max_batches: If > 0, yield at most this many batches; otherwise yield all.

        Yields:
            :class:`EvaluationCaseBatch` items, one per case or mini-batch.
        """
        ...

    def description(self) -> str:
        """Return a human-readable description of this provider, for logging."""
        ...


# =================================================================================================
@runtime_checkable
class SupportsEvaluationRegimes(Protocol):
    """Protocol that all Lightning families implement after the regime abstraction is added.

    Adding this surface makes the regime runner family-agnostic: it calls these
    two methods on any Lightning module without caring about TEM, EHC, or HRM specifics.

    Contract:
        - ``build_evaluation_metrics`` must return a fresh metric collection with
          the requested namespace prefix; it must not return or reuse ``self.val_metrics``.
        - ``execute_evaluation_batch`` must run one evaluation without writing into
          ``self.val_metrics``.
        - Both methods must be implemented identically across all Lightning families.
    """

    def build_evaluation_metrics(self, namespace: str) -> MetricCollection:
        """Return a fresh family-appropriate metric collection with the given namespace prefix.

        Args:
            namespace: Metric namespace prefix, e.g. ``"diag/my_probe/"``.

        Returns:
            A :class:`~torchmetrics.MetricCollection` that is separate from ``self.val_metrics``.
        """
        ...

    def execute_evaluation_batch(
        self,
        batch: Batch,
        trace_request: Optional[EvaluationTraceRequest],
    ) -> EvaluationBatchArtifacts:
        """Execute one evaluation batch and return scored artifacts.

        Must not write into ``self.val_metrics``. Returns a scored rollout result
        and an optional trace for figure consumers.

        Args:
            batch: A task batch identical in structure to fit-path validation batches.
            trace_request: Which trace keys to include in the output, or ``None`` for no trace.

        Returns:
            :class:`EvaluationBatchArtifacts` with ``regime_id`` set to ``"_inline"``
            (the caller overwrites it with the actual regime id after the call).
        """
        ...


# =============================================================================
__all__ = [
    "EvaluationBatchArtifacts",
    "EvaluationCaseBatch",
    "EvaluationRegimeSettings",
    "EvaluationScheduleSettings",
    "EvaluationSourceProvider",
    "EvaluationTraceRequest",
    "SupportsEvaluationRegimes",
]
