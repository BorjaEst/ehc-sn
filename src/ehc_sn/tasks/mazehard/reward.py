"""MazeHard task-owned reward semantics.

:class:`MazeHardRewardProjector` computes the dense improvement reward from
task evaluation semantics (:class:`~ehc_sn.tasks.mazehard.evaluation.MazeHardStepScore`).
This is a task-owned orthogonal projection — it does not depend on any
capability or execution-binding internals.

Reward formula: ``exp(accuracy_t) - exp(accuracy_{t-1})``.

The projector is injected into
:class:`~ehc_sn.tasks.mazehard.capabilities.deliberation.MazeHardDeliberationCapability`
at construction; the capability remains thin mode wiring only.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel
from torch import Tensor

from .evaluation import MazeHardStepScore


# =============================================================================
class MazeHardRewardConfig(BaseModel, extra="forbid"):
    """Configuration for the MazeHard dense improvement reward."""


# =============================================================================
class MazeHardRewardProjector:
    """Orthogonal reward projection over MazeHard task evaluation semantics.

    Consumes :class:`~ehc_sn.tasks.mazehard.evaluation.MazeHardStepScore`
    and applies the dense improvement formula:
    ``reward = exp(accuracy_t) - exp(accuracy_{t-1})``.

    No capability internals are required.  The projector is stateless;
    ``prev_accuracy`` is threaded externally by the capability runtime.
    """

    def __init__(self, config: MazeHardRewardConfig | None = None) -> None:
        """Create a reward projector.

        Args:
            config: Optional reward configuration.  Defaults to
                :class:`MazeHardRewardConfig` with standard settings.
        """
        self._config = config or MazeHardRewardConfig()

    def project_step_reward(
        self,
        step_score: MazeHardStepScore,
        *,
        prev_accuracy: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(accuracy, reward)`` for one deliberation step.

        Args:
            step_score: Per-sequence correctness summary from
                :func:`~ehc_sn.tasks.mazehard.evaluation.build_maze_hard_step_score`.
            prev_accuracy: Previous step accuracy tensor of shape ``(B, 1)``,
                or ``None`` for the first step (treated as zero).

        Returns:
            A tuple ``(accuracy, reward)`` where both tensors have shape
            ``(B, 1)`` and dtype ``float32``.
        """
        accuracy = step_score.sequence_accuracy.unsqueeze(-1).to(dtype=torch.float32)
        if prev_accuracy is None:
            prev_acc = torch.zeros_like(accuracy)
        else:
            prev_acc = prev_accuracy.to(device=accuracy.device, dtype=torch.float32)
        reward = torch.exp(accuracy) - torch.exp(prev_acc)
        return accuracy, reward


# =============================================================================
__all__ = [
    "MazeHardRewardConfig",
    "MazeHardRewardProjector",
]
