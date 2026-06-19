"""Rollout materialization — scored step and chunk containers.

These types are produced by :func:`~ehc_sn.rollouts.scoring.score_rollout_chunk`
and consumed by metrics aggregation, trace observers, and evaluation paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Mapping, TypeVar

from torch import Tensor

from ehc_sn.rollouts.runtime import CarrySnapshot, HaltedCarry
from ehc_sn.types import Batch

ScoredOutputT = TypeVar("ScoredOutputT")
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
__all__ = [
    "EvaluatedChunk",
    "ObservedStep",
]
