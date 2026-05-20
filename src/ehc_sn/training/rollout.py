"""Training-owned rollout orchestration helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping

from torch import Tensor

from ehc_sn.objectives.rollout import (
    EvaluatedChunk,
    ObservedStep,
    RolloutScorer,
    materialize_observed_step,
    score_rollout_chunk,
)
from ehc_sn.rollouts.runtime import (
    RolloutChunk,
    RolloutExecution,
    Runner,
    Source,
    StepController,
    StepRecord,
)


# =============================================================================
@dataclass(frozen=True)
class CapturedRolloutResult:
    """Pair an executed rollout chunk with its objective-scored result."""

    chunk: RolloutChunk
    evaluated: EvaluatedChunk


# =============================================================================
@dataclass(frozen=True)
class StreamingRolloutResult:
    """Streaming rollout evaluation without full-step materialization."""

    execution: RolloutExecution
    loss: Tensor
    last_step: ObservedStep


# =============================================================================
def run_captured_rollout(  # -------------------------------------------------
    *,
    runner: Runner,
    source: Source,
    controller: StepController,
    carry: Any,
    max_rollout_steps: int | None = None,
    hard_max_rollout_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
) -> RolloutChunk:
    """Execute a rollout and return the captured chunk."""
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
            "Captured rollout execution requires step records, but the "
            "runner returned a recordless execution summary."
        )
    return executed


# =============================================================================
def score_captured_rollout(  # -----------------------------------------------
    *,
    runner: Runner,
    source: Source,
    controller: StepController,
    carry: Any,
    objective: RolloutScorer,
    max_rollout_steps: int | None = None,
    hard_max_rollout_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
    objective_options: Mapping[str, object] | None = None,
) -> CapturedRolloutResult:
    """Execute a rollout chunk and score it with a pure objective."""
    executed = run_captured_rollout(
        runner=runner,
        source=source,
        controller=controller,
        carry=carry,
        max_rollout_steps=max_rollout_steps,
        hard_max_rollout_steps=hard_max_rollout_steps,
        runner_options=runner_options,
    )
    evaluated = score_rollout_chunk(
        executed, objective, **dict(objective_options or {})
    )
    return CapturedRolloutResult(chunk=executed, evaluated=evaluated)


# =============================================================================
def score_rollout_streaming(  # ----------------------------------------------
    *,
    runner: Runner,
    source: Source,
    controller: StepController,
    carry: Any,
    objective: RolloutScorer,
    max_rollout_steps: int | None = None,
    hard_max_rollout_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
    objective_options: Mapping[str, object] | None = None,
    observed_step_observer: Callable[[ObservedStep], None] | None = None,
) -> StreamingRolloutResult:
    """Execute a rollout and score records on the fly without storing the full chunk."""
    objective_options_dict = dict(objective_options or {})
    total_loss: Tensor | None = None
    last_step: ObservedStep | None = None

    def observe_record(record: StepRecord) -> None:
        nonlocal total_loss, last_step
        step_output = objective.evaluate_step(record, **objective_options_dict)
        observed = materialize_observed_step(record, step_output)
        if observed_step_observer is not None:
            observed_step_observer(observed)
        last_step = observed
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
        raise ValueError("Streaming rollout evaluation received no steps.")
    return StreamingRolloutResult(
        execution=executed, loss=total_loss, last_step=last_step
    )


# =============================================================================
__all__ = [
    "CapturedRolloutResult",
    "StreamingRolloutResult",
    "run_captured_rollout",
    "score_captured_rollout",
    "score_rollout_streaming",
]
