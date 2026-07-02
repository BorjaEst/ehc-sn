"""Rollout scoring — protocols and orchestration for scoring executed chunks.

The canonical per-record scoring transition is :func:`score_rollout_record`.
Traversal drivers (captured, streaming) call it once per ``StepRecord``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Protocol, TypeAlias, TypeVar

import torch
from torch import Tensor

from ehc_sn.rollouts.materialization import EvaluatedChunk, ObservedStep
from ehc_sn.rollouts.runtime import HaltedCarry, RolloutChunk, StepRecord

# =============================================================================
# Canonical per-record scoring operation
# =============================================================================

StepOutputT = TypeVar("StepOutputT", bound="ObjectiveStepOutput")
_AccumCarryT = TypeVar("_AccumCarryT", bound="HaltedCarry")
"""TypeVar for RolloutAccumulator's carry type."""


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
        source_exhausted=chunk.source_exhausted,
    )


# =============================================================================
# =============================================================================
# Streaming evaluation accumulator — replaces list[ObservedStep]
# =============================================================================


@dataclass
class RolloutAccumulator(Generic[_AccumCarryT, StepOutputT]):
    """Streaming accumulator for bounded-memory evaluation.

    Folds per-step metrics into sufficient statistics during the rollout
    loop and retains only the final ``ObservedStep`` for downstream consumers
    that need per-step data.  This changes memory scaling from
    ``O(rollout_length × step_state_size)`` to ``O(step_state_size)``.

    After all steps have been processed, call ``finalize(carry)`` to produce
    an ``EvaluatedChunk`` with the same ``loss`` and aggregate metrics as the
    old unbounded list, but only the final step retained in ``steps``.

    Usage::

        accumulator = RolloutAccumulator[_AccumCarryT, StepOutputT]()

        for record in ...:
            accumulator.observe(scored_record)

        chunk = accumulator.finalize(final_carry, source_exhausted)
        # chunk.loss == sum of per-step losses
        # chunk.steps == (final_step_only,)
    """

    loss_sum: Tensor = field(default_factory=lambda: torch.zeros(()))
    """Accumulated sum of per-step scalar losses."""
    loss_count: int = 0
    """Number of steps accumulated."""
    last_step: ObservedStep[StepOutputT] | None = None
    """Most recently scored step (retained as the 'final' step)."""

    def observe(self, scored: ScoredRecord[StepOutputT]) -> None:
        """Accumulate one scored step.

        Adds the step's loss to the running sum and retains the step as
        the new 'last step'.  Previous step references are released.
        """
        self.loss_sum = self.loss_sum + scored.loss
        self.loss_count += 1
        self.last_step = scored.observed_step

    def finalize(
        self,
        final_carry: _AccumCarryT,
        source_exhausted: bool = False,
    ) -> EvaluatedChunk[_AccumCarryT, StepOutputT]:
        """Produce an ``EvaluatedChunk`` from accumulated statistics.

        The returned ``loss`` is a detached CPU scalar tensor so the result
        is storage-safe (no CUDA tensors, no autograd graph).  The live
        recurrent ``final_carry`` is *not* stored in the chunk; it lives
        in the runner's ``RolloutExecution`` or ``RolloutChunk``.

        Returns
        -------
        EvaluatedChunk
            A chunk whose ``loss`` equals the sum of all observed per-step
            losses and whose ``steps`` contains only the final step
            (or an empty tuple if zero steps were observed).

        Raises
        ------
        ValueError
            If no steps were observed.
        """
        if self.loss_count == 0 or self.last_step is None:
            raise ValueError("RolloutAccumulator: no steps were observed.")
        return EvaluatedChunk(
            steps=(self.last_step,),
            loss=self.loss_sum,
            source_exhausted=source_exhausted,
        )


__all__ = [
    "ObjectiveStepOutput",
    "RolloutAccumulator",
    "RolloutScorer",
    "ScoredRecord",
    "materialize_observed_step",
    "score_rollout_chunk",
    "score_rollout_record",
]
