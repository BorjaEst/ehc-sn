"""SeqMaze task-owned reward semantics.

:class:`SeqMazeRewardProjector` computes the stop-time reward from
task evaluation semantics (:class:`~ehp_sn.tasks.seqmaze.evaluation.SeqMazeStepScore`).

Reward formula: terminal success reward for correct HALT, explicit continue cost,
incorrect-halt penalty, and truncation penalty.

The projector is stateless and injected into
:class:`~ehp_sn.tasks.seqmaze.runtime.SeqMazeRuntime` at construction.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.tasks.seqmaze.evaluation import SeqMazeStepScore


# =============================================================================
class SeqMazeRewardConfig(BaseModel, extra="forbid"):
    """Configuration for the SeqMaze stop-time reward."""

    success_reward: float = Field(
        default=1.0, description="Reward for correct HALT."
    )
    continue_cost: float = Field(
        default=-0.01, description="Cost per CONTINUE step."
    )
    incorrect_halt_penalty: float = Field(
        default=-1.0, description="Penalty for HALT with incorrect answer."
    )
    truncation_penalty: float = Field(
        default=-1.0,
        description="Penalty when episode_horizon is reached.",
    )


# =============================================================================
class SeqMazeRewardProjector:
    """Orthogonal reward projection over SeqMaze task evaluation semantics.

    Consumes :class:`~ehp_sn.tasks.seqmaze.evaluation.SeqMazeStepScore`
    and applies the stop-time reward:

    - terminal success reward when halting with exact path correctness
    - explicit continue cost when still deliberating
    - explicit incorrect-halt penalty
    - explicit truncation penalty

    The reward trains computation allocation, not path construction.
    The supervised objective remains responsible for teaching the path.
    """

    def __init__(self, config: SeqMazeRewardConfig | None = None) -> None:
        self._config = config or SeqMazeRewardConfig()

    def project_step_reward(
        self,
        step_score: SeqMazeStepScore,
        *,
        terminated: Tensor,
        truncated: Tensor,
    ) -> Tensor:
        """Return the stop-time reward for one deliberation step.

        Args:
            step_score: Per-sequence correctness summary.
            terminated: Per-slot termination signal (halt action), shape ``(B,)``.
            truncated: Per-slot truncation signal (episode horizon), shape ``(B,)``.

        Returns:
            Reward tensor of shape ``(B, 1)``, dtype ``float32``.
        """
        sequence_correct = step_score.path_exact
        device = sequence_correct.device
        batch = sequence_correct.shape[0]

        # Default: continue cost
        reward = torch.full(
            (batch, 1),
            float(self._config.continue_cost),
            device=device,
            dtype=torch.float32,
        )

        # Truncation overrides continue cost
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

        # Halt: success if correct, penalty otherwise
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
    "SeqMazeRewardConfig",
    "SeqMazeRewardProjector",
]
