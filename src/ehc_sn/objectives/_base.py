"""Shared objective base abstractions.

This module contains only universal rollout-objective wiring that is valid
across multiple head families. Family-specific logic lives in sibling internal
modules such as ``_token.py`` and ``_variational.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from torch import nn

from ehc_sn.objectives.rollout import EvaluatedChunk, score_rollout_chunk
from ehc_sn.rollouts.runtime import RolloutChunk, StepRecord


# =============================================================================
class BaseObjective[ConfigT: BaseModel](nn.Module):
    """Minimal wiring base for all rollout objectives.

    Concrete subclasses score executed :class:`~ehc_sn.rollouts.RolloutChunk`
    objects and return an :class:`~ehc_sn.objectives.rollout.EvaluatedChunk` containing
    one scored step result per executed step.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: ConfigT,
    ) -> None:
        """Initialize the objective with the given configuration."""
        super().__init__()
        self._config = config

    @property
    def config(self) -> ConfigT:
        """Return the objective configuration."""
        return self._config

    def forward(  # -----------------------------------------------------------
        self,
        chunk: RolloutChunk,
        **options: Any,
    ) -> EvaluatedChunk:
        """Score an executed rollout chunk and return one observed step per record."""
        return score_rollout_chunk(chunk, self, **options)

    def evaluate_step(  # -----------------------------------------------------
        self,
        record: StepRecord,
        **options: Any,
    ) -> Any:
        """Score one executed rollout step."""
        raise NotImplementedError


# =============================================================================
__all__ = ["BaseObjective"]
