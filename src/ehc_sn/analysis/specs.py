"""Aggregate and analysis specification types for the evaluation planner.

These are code-level, immutable registrations analogous to
``ehp_sn.figures.FigureSpec``.  They encode executable scientific
methods and their dependency contracts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from ehc_sn.contracts.dependencies import Dependency
from ehc_sn.evaluation.contracts import (
    ArtifactKey,
    ArtifactRequirement,
    EvaluationConsumer,
    ProducedArtifact,
)


# =============================================================================
class ConsumerPlacement(StrEnum):
    """Where a consumer's state lives during evaluation.

    ``MODEL_DEVICE``
        State tensors reside on the model device (GPU) during rollout.
        Transfer to CPU happens during ``finalize()``.
    ``CPU_STREAM``
        Consumer processes detached CPU values.  Suitable for lightweight
        observations that do not require device-side accumulation.
    """

    MODEL_DEVICE = "model_device"
    CPU_STREAM = "cpu_stream"


# =============================================================================
@dataclass(frozen=True)
class AggregateSpec:
    """Registration for an on-device or CPU-stream evaluation consumer.

    The ``factory`` receives an ``AggregateBuildContext`` and must return
    an ``EvaluationConsumer`` instance.
    """

    name: str
    schema_version: int
    factory: Callable[..., EvaluationConsumer]
    dependencies: frozenset[Dependency]
    placement: ConsumerPlacement = ConsumerPlacement.MODEL_DEVICE
    produces: ArtifactKey | None = None
    """The single artifact key produced by this aggregate's ``finalize()``.
    ``None`` when the aggregate produces no persisted artifact (e.g. purely
    diagnostic)."""


# =============================================================================
@dataclass(frozen=True)
class AnalysisSpec:
    """Registration for a post-evaluation CPU analysis.

    The ``runner`` receives loaded input artifacts as keyword arguments
    along with any ``parameters`` from configuration.
    """

    name: str
    schema_version: int
    inputs: frozenset[ArtifactRequirement]
    runner: Callable[..., Sequence[ProducedArtifact]]
    produces: frozenset[ArtifactKey] = field(default_factory=frozenset)
    parameters: frozenset[str] = field(default_factory=frozenset)
    """Parameter names accepted by the runner (e.g. ``{"grid_score_threshold"}``)."""
