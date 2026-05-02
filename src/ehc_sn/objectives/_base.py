"""Shared objective base abstractions.

This module contains only universal rollout-objective wiring that is valid
across multiple head families. Family-specific logic lives in sibling internal
modules such as ``_token.py`` and ``_variational.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from torch import Tensor, nn

from ehc_sn.rollouts import EvaluatedChunk, ObservedStep, RolloutChunk, StepRecord


# =================================================================================================
class BaseObjective[ConfigT: BaseModel](nn.Module):
    """Minimal wiring base for all rollout objectives.

    Concrete subclasses score executed :class:`~ehc_sn.rollouts.RolloutChunk`
    objects and return an :class:`~ehc_sn.rollouts.EvaluatedChunk` containing
    one scored step result per executed step.
    """

    def __init__(self, config: ConfigT) -> None:
        super().__init__()
        self._config = config

    @property
    def config(self) -> ConfigT:
        """Return the objective configuration."""
        return self._config

    def forward(self, chunk: RolloutChunk, **options: Any) -> EvaluatedChunk:
        """Score an executed rollout chunk and return one observed step per record."""
        observed_steps: list[ObservedStep] = []
        total_loss: Tensor | None = None

        for record in chunk.records:
            step_output = self.evaluate_step(record, **options)
            executed = record.executed_frame if record.executed_frame is not None else record.batch
            observed_steps.append(
                ObservedStep(
                    index=record.index,
                    batch=executed,
                    executed_frame=executed,
                    sampled_input=record.sampled_input,
                    snapshot=record.snapshot,
                    outputs=step_output,
                )
            )
            total_loss = step_output.loss if total_loss is None else total_loss + step_output.loss

        if total_loss is None:
            raise ValueError("Objective received an empty rollout chunk.")

        return EvaluatedChunk(
            steps=tuple(observed_steps),
            loss=total_loss,
            final_carry=chunk.final_carry,
            source_exhausted=chunk.source_exhausted,
        )

    def evaluate_step(self, record: StepRecord, **options: Any) -> Any:
        """Score one executed rollout step."""
        raise NotImplementedError


# =================================================================================================
__all__ = ["BaseObjective"]
