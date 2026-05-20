"""MazeHard task-owned reward semantics.

:class:`MazeHardRewardProjector` computes the stop-time reward from
task evaluation semantics (:class:`~ehc_sn.tasks.mazehard.evaluation.MazeHardStepScore`).
This is a task-owned orthogonal projection — it does not depend on any
capability or execution-binding internals.

Reward formula: terminal success reward plus explicit continue cost, incorrect
halt penalty, and truncation penalty.

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
    """Configuration for the MazeHard stop-time reward."""

    success_reward: float = 1.0
    continue_cost: float = -0.01
    incorrect_halt_penalty: float = -1.0
    truncation_penalty: float = -1.0


# =============================================================================
class MazeHardRewardProjector:
    """Orthogonal reward projection over MazeHard task evaluation semantics.

    Consumes :class:`~ehc_sn.tasks.mazehard.evaluation.MazeHardStepScore`
    and applies the stop-time reward:
    - terminal success reward when halting with exact sequence correctness
    - explicit continue cost when still deliberating
    - explicit incorrect-halt penalty
    - explicit truncation penalty

    No capability internals are required.  The projector is stateless.
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
        terminated: Tensor,
        truncated: Tensor,
    ) -> Tensor:
        """Return the stop-time reward for one deliberation step.

        Args:
            step_score: Per-sequence correctness summary from
                :func:`~ehc_sn.tasks.mazehard.evaluation.build_maze_hard_step_score`.
            terminated: Per-slot termination signal (halt action) of shape ``(B,)``.
            truncated: Per-slot truncation signal (episode horizon) of shape ``(B,)``.

        Returns:
            Reward tensor of shape ``(B, 1)`` and dtype ``float32``.
        """
        sequence_correct = step_score.sequence_is_correct
        device = sequence_correct.device
        batch = sequence_correct.shape[0]

        reward = torch.full(
            (batch, 1),
            float(self._config.continue_cost),
            device=device,
            dtype=torch.float32,
        )

        truncation_reward = torch.full(
            (batch, 1),
            float(self._config.truncation_penalty),
            device=device,
            dtype=torch.float32,
        )
        reward = torch.where(
            truncated.to(device=device, dtype=torch.bool).unsqueeze(-1),
            truncation_reward,
            reward,
        )

        success_reward = torch.full(
            (batch, 1),
            float(self._config.success_reward),
            device=device,
            dtype=torch.float32,
        )
        incorrect_reward = torch.full(
            (batch, 1),
            float(self._config.incorrect_halt_penalty),
            device=device,
            dtype=torch.float32,
        )
        halt_reward = torch.where(
            sequence_correct.to(device=device, dtype=torch.bool).unsqueeze(-1),
            success_reward,
            incorrect_reward,
        )
        reward = torch.where(
            terminated.to(device=device, dtype=torch.bool).unsqueeze(-1),
            halt_reward,
            reward,
        )
        return reward


# =============================================================================
__all__ = [
    "MazeHardRewardConfig",
    "MazeHardRewardProjector",
]
