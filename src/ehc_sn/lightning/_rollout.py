""" """

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from torch import Tensor
from torchmetrics import MetricCollection

from ehc_sn.metrics import Route, update_metrics_from_step
from ehc_sn.rollouts import (
    EvaluatedChunk,
    ObjectiveStepOutput,
    ObservedStep,
    RolloutChunk,
    RolloutExecution,
    Runner,
    Source,
    StepController,
    StepRecord,
)


# =============================================================================
class RolloutObjective(Protocol):
    """Protocol for pure objectives that score executed rollout chunks."""

    def __call__(  # ----------------------------------------------------------
        self,
        chunk: RolloutChunk,
        **options: Any,
    ) -> EvaluatedChunk: ...

    def evaluate_step(  # -----------------------------------------------------
        self,
        record: StepRecord,
        **options: Any,
    ) -> ObjectiveStepOutput: ...


# =============================================================================
@dataclass(frozen=True)
class RolloutEvaluation:
    """Pair an executed rollout chunk with its objective-scored result."""

    chunk: RolloutChunk
    evaluated: EvaluatedChunk


# =============================================================================
@dataclass(frozen=True)
class StreamingRolloutEvaluation:
    """Streaming rollout evaluation without full-step materialization."""

    execution: RolloutExecution
    loss: Tensor
    last_step: ObservedStep


# =============================================================================
def evaluate_rollout(  # ------------------------------------------------------
    *,
    runner: Runner,
    source: Source,
    controller: StepController,
    carry: Any,
    objective: RolloutObjective,
    max_rollout_steps: int | None = None,
    hard_max_rollout_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
    objective_options: Mapping[str, object] | None = None,
    metric_collection: MetricCollection | None = None,
    metric_routes: list[Route] | tuple[Route, ...] = (),
) -> RolloutEvaluation:
    """Execute a rollout chunk and score it with a pure objective."""
    executed = runner.run(
        source=source,
        controller=controller,
        carry=carry,
        max_rollout_steps=max_rollout_steps,
        hard_max_rollout_steps=hard_max_rollout_steps,
        options=dict(runner_options or {}),
    )
    if not isinstance(executed, RolloutChunk):
        raise TypeError(
            "Rollout evaluation requires captured step records, but the "
            "runner returned a recordless execution summary."
        )
    evaluated = objective(executed, **dict(objective_options or {}))
    if metric_collection is not None:
        update_metric_collection_from_evaluated_chunk(
            metric_collection, evaluated, metric_routes
        )
    return RolloutEvaluation(chunk=executed, evaluated=evaluated)


# =============================================================================
def evaluate_rollout_streaming(  # --------------------------------------------
    *,
    runner: Runner,
    source: Source,
    controller: StepController,
    carry: Any,
    objective: RolloutObjective,
    max_rollout_steps: int | None = None,
    hard_max_rollout_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
    objective_options: Mapping[str, object] | None = None,
    metric_collection: MetricCollection | None = None,
    metric_routes: list[Route] | tuple[Route, ...] = (),
) -> StreamingRolloutEvaluation:
    """Execute a rollout and score records on the fly without storing the full chunk."""
    objective_options_dict = dict(objective_options or {})
    total_loss: Tensor | None = None
    last_step: ObservedStep | None = None

    def observe_record(record: StepRecord) -> None:
        nonlocal total_loss, last_step
        step_output = objective.evaluate_step(record, **objective_options_dict)
        if metric_collection is not None:
            update_metrics_from_step(
                metric_collection, step_output.metrics, metric_routes
            )
        executed = (
            record.executed_frame
            if record.executed_frame is not None
            else record.batch
        )
        last_step = ObservedStep(
            index=record.index,
            batch=executed,
            executed_frame=executed,
            sampled_input=record.sampled_input,
            snapshot=record.snapshot,
            outputs=step_output,
        )
        total_loss = (
            step_output.loss
            if total_loss is None
            else total_loss + step_output.loss
        )

    executed = runner.run(
        source=source,
        controller=controller,
        carry=carry,
        max_rollout_steps=max_rollout_steps,
        hard_max_rollout_steps=hard_max_rollout_steps,
        options=dict(runner_options or {}),
        record_observer=observe_record,
        capture_records=False,
    )
    if isinstance(executed, RolloutChunk):
        raise TypeError(
            "Streaming rollout evaluation expects a recordless execution "
            "summary, but the runner returned a captured chunk."
        )
    if total_loss is None or last_step is None:
        raise ValueError(
            "Streaming rollout evaluation received no executed steps."
        )
    return StreamingRolloutEvaluation(
        execution=executed, loss=total_loss, last_step=last_step
    )


# =============================================================================
def update_metric_collection_from_evaluated_chunk(  # -------------------------
    collection: MetricCollection,
    evaluated: EvaluatedChunk,
    routes: list[Route] | tuple[Route, ...],
) -> None:
    """Fold all scored-step metrics from an evaluated chunk into a collection."""
    for step in evaluated.steps:
        update_metrics_from_step(collection, step.outputs.metrics, routes)


# =============================================================================
__all__ = [
    "RolloutEvaluation",
    "RolloutObjective",
    "evaluate_rollout",
    "evaluate_rollout_streaming",
    "StreamingRolloutEvaluation",
    "update_metric_collection_from_evaluated_chunk",
]
