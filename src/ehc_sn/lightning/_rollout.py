"""Lightning-local helpers for rollout execution and scoring.

These helpers keep learner modules focused on optimizer and scheduler control
while centralizing the repeated runner/objective/observer wiring shared across
Lightning training surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from torchmetrics import MetricCollection

from ehc_sn.metrics import Route, update_metrics_from_step
from ehc_sn.rollouts import EvaluatedChunk, RolloutChunk, Runner, Source, StepController
from ehc_sn.traces import TraceObserver, TraceSpec, TraceTree


# =================================================================================================
class RolloutObjective(Protocol):
    """Protocol for pure objectives that score executed rollout chunks."""

    def __call__(self, chunk: RolloutChunk, **options: Any) -> EvaluatedChunk: ...


# =================================================================================================
@dataclass(frozen=True)
class RolloutEvaluation:
    """Pair an executed rollout chunk with its objective-scored result."""

    chunk: RolloutChunk
    evaluated: EvaluatedChunk


# =================================================================================================
def evaluate_rollout(
    *,
    runner: Runner,
    source: Source,
    controller: StepController,
    carry: Any,
    objective: RolloutObjective,
    max_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
    objective_options: Mapping[str, object] | None = None,
) -> RolloutEvaluation:
    """Execute a rollout chunk and score it with a pure objective."""
    executed = runner.run(
        source=source,
        controller=controller,
        carry=carry,
        max_steps=max_steps,
        options=dict(runner_options or {}),
    )
    evaluated = objective(executed, **dict(objective_options or {}))
    return RolloutEvaluation(chunk=executed, evaluated=evaluated)


# =================================================================================================
def update_metric_collection_from_evaluated_chunk(
    collection: MetricCollection,
    evaluated: EvaluatedChunk,
    routes: list[Route] | tuple[Route, ...],
) -> None:
    """Fold all scored-step metrics from an evaluated chunk into a collection."""
    for step in evaluated.steps:
        update_metrics_from_step(collection, step.outputs.metrics, routes)


# =================================================================================================
def observe_evaluated_chunk(evaluated: EvaluatedChunk, trace_spec: TraceSpec[Any]) -> TraceTree:
    """Build a trace tree from the scored steps of an evaluated chunk."""
    observer = TraceObserver(TraceTree(), trace_spec)
    observer.observe_chunk(evaluated)
    return observer.tree


# =================================================================================================
__all__ = [
    "RolloutEvaluation",
    "RolloutObjective",
    "evaluate_rollout",
    "observe_evaluated_chunk",
    "update_metric_collection_from_evaluated_chunk",
]
