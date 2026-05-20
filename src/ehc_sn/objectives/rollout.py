"""Objective-owned rollout scoring surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Mapping, Protocol, TypeVar

from torch import Tensor

from ehc_sn.rollouts.runtime import (
    CarrySnapshot,
    HaltedCarry,
    RolloutChunk,
    StepRecord,
)
from ehc_sn.types import Batch


# =============================================================================
class ObjectiveStepOutput(Protocol):
    """Minimal scored-step output exposed outside objective modules."""

    @property
    def loss(self) -> Tensor:
        """Return the scalar loss tensor for this step."""

    metrics: object
    signals: Mapping[str, object]


ScoredOutputT = TypeVar("ScoredOutputT", bound=ObjectiveStepOutput)
CarryT = TypeVar("CarryT", bound=HaltedCarry)


# =============================================================================
@dataclass(frozen=True)
class ObservedStep(Generic[ScoredOutputT]):
    """Objective-scored step context consumed by metrics and trace observers.

    Fields
    ------
    batch:
        Backward-compatible alias; always contains the *executed_frame* content.
    sampled_input:
        Raw source-batch proposal; ``None`` when not populated by the runner.
    executed_frame:
        Exact tensors consumed by the model and objective on this step.
        ``None`` when not populated (legacy path).
    """

    index: int
    batch: Batch
    snapshot: CarrySnapshot
    outputs: ScoredOutputT
    sampled_input: Batch | None = None
    executed_frame: Batch | None = None

    @property
    def carry(self) -> CarrySnapshot:
        """Backward-compatible alias for the frozen post-step snapshot."""
        return self.snapshot


# =============================================================================
@dataclass(frozen=True)
class EvaluatedChunk(Generic[CarryT, ScoredOutputT]):
    """Objective-scored rollout fragment returned by a pure objective."""

    steps: tuple[ObservedStep[ScoredOutputT], ...]
    loss: Tensor
    final_carry: CarryT
    source_exhausted: bool = False

    @property
    def last_step(self) -> ObservedStep[ScoredOutputT]:
        """Return the final scored step in the chunk."""
        if not self.steps:
            raise ValueError("EvaluatedChunk has no observed steps.")
        return self.steps[-1]


# =============================================================================
class RolloutScorer(Protocol):
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
    **options: Any,
) -> EvaluatedChunk:
    """Score an executed rollout chunk and return an evaluated chunk."""
    observed_steps: list[ObservedStep] = []
    total_loss: Tensor | None = None

    for record in chunk.records:
        step_output = scorer.evaluate_step(record, **options)
        observed_steps.append(materialize_observed_step(record, step_output))
        total_loss = (
            step_output.loss
            if total_loss is None
            else total_loss + step_output.loss
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
    "EvaluatedChunk",
    "ObjectiveStepOutput",
    "ObservedStep",
    "RolloutScorer",
    "materialize_observed_step",
    "score_rollout_chunk",
]
