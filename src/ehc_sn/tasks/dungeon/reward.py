"""Dungeon task-owned reward semantics.

:class:`DungeonRewardProjector` computes a per-step reward from
task evaluation semantics (:class:`~ehc_sn.tasks.dungeon.evaluation.DungeonStepScore`).
This is a task-owned orthogonal projection — it does not depend on any
capability or execution-binding internals.

Reward formula: sparse binary success reward, optionally scaled.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from torch import Tensor

from .evaluation import DungeonStepScore


# =============================================================================
class DungeonRewardConfig(BaseModel, extra="forbid"):
    """Configuration for the Dungeon task-owned reward projection.

    Attributes:
        success_reward: Scalar reward returned when the agent reaches the
            goal (``step_score.success == True``).
    """

    success_reward: float = Field(
        default=1.0,
        gt=0.0,
        description="Sparse reward emitted on goal-reach.",
    )


# =============================================================================
class DungeonRewardProjector:
    """Orthogonal reward projection over Dungeon task evaluation semantics.

    Consumes :class:`~ehc_sn.tasks.dungeon.evaluation.DungeonStepScore` and
    returns a sparse per-step reward tensor.

    No capability internals are required.  The projector is stateless and
    mode-agnostic.
    """

    def __init__(self, config: DungeonRewardConfig | None = None) -> None:
        """Create a reward projector.

        Args:
            config: Optional reward configuration.  Defaults to
                :class:`DungeonRewardConfig` with standard settings.
        """
        self._config = config or DungeonRewardConfig()

    def project_step_reward(self, step_score: DungeonStepScore) -> Tensor:
        """Return per-slot sparse reward based on goal-reach.

        Args:
            step_score: Per-step semantic state from
                :func:`~ehc_sn.tasks.dungeon.evaluation.build_dungeon_step_score`.

        Returns:
            Reward tensor of shape ``(B, 1)`` float32.  Non-zero when
            ``step_score.success`` is True.
        """
        return (step_score.success.float() * self._config.success_reward).unsqueeze(-1)


# =============================================================================
__all__ = [
    "DungeonRewardConfig",
    "DungeonRewardProjector",
]
