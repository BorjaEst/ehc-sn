"""Public contracts for top-level replay evaluation execution.

Task packages implement provider types against this contract. Lightning model
families expose ``execute_evaluation_batch`` against this contract.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

from ehc_sn.objectives.rollout import EvaluatedChunk
from ehc_sn.traces.observer import TraceSpec
from ehc_sn.traces.trace_tree import TraceTree
from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class EvaluationCaseBatch:
    """One provider-owned evaluation case batch and associated source context."""

    batch: Batch
    case_id: str
    source_context: object | None = None


# =============================================================================
@dataclass(frozen=True)
class EvaluationTraceRequest:
    """Trace materialization request for an evaluation batch execution."""

    enabled: bool = False
    trace_spec: TraceSpec | None = None
    trace_meta: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.enabled and self.trace_spec is None:
            raise ValueError(
                "EvaluationTraceRequest.enabled=True requires trace_spec."
            )


# =============================================================================
@dataclass(frozen=True)
class EvaluationBatchResult:
    """Result payload returned by ``execute_evaluation_batch``."""

    case_id: str
    evaluated: EvaluatedChunk
    source_context: object | None = None
    trace: TraceTree | None = None


# =============================================================================
@dataclass(frozen=True)
class EvaluationRegimeResult:
    """Aggregate result payload for one named evaluation regime."""

    regime_id: str
    results: tuple[EvaluationBatchResult, ...]


# =============================================================================
class EvaluationSourceProvider(Protocol):
    """Task-owned provider that yields replay case batches for one regime."""

    def provide_cases(
        self,
        max_batches: int = 0,
    ) -> Iterator[EvaluationCaseBatch]:
        """Yield replay case batches in deterministic provider-owned order."""

    def description(
        self,
    ) -> str:
        """Return a human-readable provider description for logs."""


# =============================================================================
class LightningEvaluationExecutor(Protocol):
    """Family-facing execution seam consumed by future evaluation callbacks."""

    def execute_evaluation_batch(
        self,
        case: EvaluationCaseBatch,
        *,
        trace_request: EvaluationTraceRequest | None = None,
    ) -> EvaluationBatchResult:
        """Execute one provider case and return objective-scored outputs."""


# =============================================================================
__all__ = [
    "EvaluationBatchResult",
    "EvaluationCaseBatch",
    "EvaluationRegimeResult",
    "EvaluationSourceProvider",
    "EvaluationTraceRequest",
    "LightningEvaluationExecutor",
]
