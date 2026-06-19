"""Public contracts for top-level replay evaluation execution.

Task packages implement provider types against this contract. Lightning model
families expose ``execute_evaluation_batch`` against this contract.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

from ehc_sn.rollouts.materialization import EvaluatedChunk
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

    trace_spec: TraceSpec
    trace_meta: Mapping[str, object] | None = None


# =============================================================================
@dataclass(frozen=True)
class EvaluationCaseResult:
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
    case_results: tuple[EvaluationCaseResult, ...]
    summary: Mapping[str, object]


# =============================================================================
class EvaluationSourceProvider(Protocol):
    """Task-owned provider that yields replay case batches for one regime."""

    def provide_cases(
        self,
        max_batches: int = 0,
    ) -> Iterator[EvaluationCaseBatch]:
        """Yield replay case batches in deterministic provider-owned order."""


# =============================================================================
class EvaluationExecutor(Protocol):
    """Family-facing execution seam consumed by future evaluation callbacks."""

    def execute_evaluation_batch(  # ------------------------------------------
        self,
        case: EvaluationCaseBatch,
        *,
        trace_request: EvaluationTraceRequest | None = None,
    ) -> EvaluationCaseResult:
        """Execute one provider case and return objective-scored outputs."""


# =============================================================================
__all__ = [
    "EvaluationCaseResult",
    "EvaluationCaseBatch",
    "EvaluationExecutor",
    "EvaluationRegimeResult",
    "EvaluationSourceProvider",
    "EvaluationTraceRequest",
]
