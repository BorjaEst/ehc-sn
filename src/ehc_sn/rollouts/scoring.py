"""Rollout scoring — protocols and orchestration for scoring executed chunks.

The canonical per-record scoring transition is :func:`score_rollout_record`.
Traversal drivers (captured, streaming) call it once per ``StepRecord``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Protocol, TypeAlias, TypeVar

from torch import Tensor

from ehc_sn.rollouts.materialization import EvaluatedChunk, ObservedStep
from ehc_sn.rollouts.runtime import RolloutChunk, StepRecord

# =============================================================================
# Canonical per-record scoring operation
# =============================================================================

StepOutputT = TypeVar("StepOutputT", bound="ObjectiveStepOutput")


# =============================================================================
@dataclass(frozen=True)
class ScoredRecord(Generic[StepOutputT]):
    """Result of scoring exactly one ``StepRecord``."""

    observed_step: "ObservedStep"
    loss: Tensor


# =============================================================================
def score_rollout_record(
    record: StepRecord,
    scorer: RolloutScorer,
    *,
    input_builder: Callable[[StepRecord], object],
) -> ScoredRecord:
    """Score exactly one ``StepRecord``.

    This is the single canonical definition of the per-step scoring
    transition.  Every traversal driver (captured, streaming) calls this
    function once per record.
    """
    scoring_input = input_builder(record)
    step_output = scorer.evaluate_step(record, inputs=scoring_input)
    observed = materialize_observed_step(record, step_output)
    return ScoredRecord(
        observed_step=observed,
        loss=step_output.loss,
    )


# =============================================================================
class ObjectiveStepOutput(Protocol):
    """Minimal scored-step output exposed outside objective modules."""

    @property
    def loss(self) -> Tensor:
        """Return the scalar loss tensor for this step."""

    metrics: object
    signals: Mapping[str, object]


# =============================================================================
class RolloutScorer(Protocol):
    """Protocol for pure objectives that score executed rollout chunks."""

    def __call__(  # ----------------------------------------------------------
        self,
        chunk: RolloutChunk,
        *,  # force keyword-only for explicit context
        scoring_inputs: object | None = None,
    ) -> EvaluatedChunk: ...

    def evaluate_step(  # -----------------------------------------------------
        self,
        record: StepRecord,
        *,  # force keyword-only for explicit context
        inputs: object | None = None,
    ) -> ObjectiveStepOutput: ...


# =============================================================================
def materialize_observed_step(  # ---------------------------------------------
    record: StepRecord,
    output: ObjectiveStepOutput,
) -> ObservedStep:
    """Materialize an observed step from an executed record and step output."""
    executed = (
        record.executed_frame
        if record.executed_frame is not None
        else record.batch
    )
    return ObservedStep(
        index=record.index,
        batch=executed,
        executed_frame=executed,
        sampled_input=record.sampled_input,
        snapshot=record.snapshot,
        outputs=output,
    )


# =============================================================================
def score_rollout_chunk(  # ---------------------------------------------------
    chunk: RolloutChunk,
    scorer: RolloutScorer,
    *,  # force keyword-only for explicit context
    scoring_input_builder: Callable[[StepRecord], object] | None = None,
) -> EvaluatedChunk:
    """Score an executed rollout chunk and return an evaluated chunk."""
    if scoring_input_builder is None:
        raise ValueError(
            "score_rollout_chunk requires a scoring_input_builder. "
            "Construct one via the regime module's _build_scoring_input method."
        )

    observed_steps: list[ObservedStep] = []
    total_loss: Tensor | None = None

    for record in chunk.records:
        scored = score_rollout_record(
            record,
            scorer,
            input_builder=scoring_input_builder,
        )
        observed_steps.append(scored.observed_step)
        total_loss = (
            scored.loss if total_loss is None else total_loss + scored.loss
        )

    if total_loss is None:
        raise ValueError("Objective received an empty rollout chunk.")

    return EvaluatedChunk(
        steps=tuple(observed_steps),
        loss=total_loss,
        final_carry=chunk.final_carry,
        source_exhausted=chunk.source_exhausted,
    )


# =============================================================================
__all__ = [
    "ObjectiveStepOutput",
    "RolloutScorer",
    "ScoredRecord",
    "materialize_observed_step",
    "score_rollout_chunk",
    "score_rollout_record",
]
