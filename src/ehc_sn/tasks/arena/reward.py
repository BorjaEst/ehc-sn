"""Arena task-owned reward semantics.

:class:`ArenaRewardProjector` computes a per-slot reward from
task evaluation semantics (:class:`~ehc_sn.tasks.arena.evaluation.ArenaStepScore`).
This is a task-owned orthogonal projection — it does not depend on any
execution-binding internals.

Reward formula: per-slot binary correctness scaled by ``accuracy_weight``.
The projector is stateless and may be injected into any execution-binding
capability that needs a reward signal for arena navigation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from torch import Tensor

from .evaluation import ArenaStepScore


# =============================================================================
class ArenaRewardConfig(BaseModel, extra="forbid"):
    """Configuration for the Arena task-owned reward projection.

    Attributes:
        accuracy_weight: Scalar multiplier applied to per-step observation
            accuracy before returning it as reward.
    """

    accuracy_weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Multiplier applied to per-step accuracy when projecting to reward.",
    )


# =============================================================================
class ArenaRewardProjector:
    """Orthogonal reward projection over Arena task evaluation semantics.

    Consumes :class:`~ehc_sn.tasks.arena.evaluation.ArenaStepScore` and
    projects per-step observation accuracy to a scalar reward signal.

    No capability internals are required.  The projector is stateless and
    mode-agnostic.
    """

    def __init__(self, config: ArenaRewardConfig | None = None) -> None:
        """Create a reward projector.

        Args:
            config: Optional reward configuration.  Defaults to
                :class:`ArenaRewardConfig` with standard settings.
        """
        self._config = config or ArenaRewardConfig()

    def project_step_reward(self, step_score: ArenaStepScore) -> Tensor:
        """Return per-slot reward from binary observation correctness.

        Args:
            step_score: Per-slot local semantic state from
                :func:`~ehc_sn.tasks.arena.evaluation.build_arena_step_score`.

        Returns:
            Tensor of shape ``(B, 1)``, dtype ``float32``.  ``1.0 * weight``
            for correct slots, ``0.0`` for incorrect.
        """
        return step_score.is_correct.float().unsqueeze(-1) * self._config.accuracy_weight


# =============================================================================
__all__ = [
    "ArenaRewardConfig",
    "ArenaRewardProjector",
]
