"""MazeHard deliberation-mode capability binding.

Owns the :class:`MazeHardDeliberationConfig` and
:class:`MazeHardDeliberationFinalizer` that wire MazeHard task semantics into
the deliberation actor-critic controller.

This is an *execution-mode* binding, not a task-identity definition.
MazeHard semantics (score, evaluation, contracts) live in the parent task package.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers.deliberation.actor_critic import DeliberationStepFinalizer, DeliberationStepResult
from ehc_sn.tasks.mazehard.contracts import MAZE_HARD_IGNORE_LABEL_ID, MazeHardTargets, MazeHardTaskOutput
from ehc_sn.tasks.mazehard.evaluation import compute_maze_hard_sequence_accuracy
from ehc_sn.types import Batch


# =============================================================================
def _compute_improvement_reward(
    output: MazeHardTaskOutput | Tensor,
    targets: MazeHardTargets | Tensor,
    *,
    prev_accuracy: Tensor | None = None,
    ignore_label_id: int = MAZE_HARD_IGNORE_LABEL_ID,
) -> tuple[Tensor, Tensor]:
    """Return ``(accuracy, reward)`` for the MazeHard dense improvement reward.

    Reward formula: ``exp(acc_t) - exp(acc_{t-1})``.
    When ``prev_accuracy`` is omitted the previous accuracy is treated as zero.
    """
    accuracy = compute_maze_hard_sequence_accuracy(
        output,
        targets,
        ignore_label_id=ignore_label_id,
    ).to(dtype=torch.float32)
    if prev_accuracy is None:
        prev_accuracy = torch.zeros_like(accuracy)
    else:
        prev_accuracy = prev_accuracy.to(device=accuracy.device, dtype=torch.float32)
    reward = torch.exp(accuracy) - torch.exp(prev_accuracy)
    return accuracy, reward


# =============================================================================
class MazeHardDeliberationConfig(BaseModel, extra="forbid"):
    """Task-owned configuration for MazeHard deliberation training.

    Attributes:
        halt_action: Action index the model uses to signal 'done' for a slot.
            Must match the action space configured in the backbone / policy head.
        episode_horizon: Task-owned semantic step budget per slot.  When
            ``steps >= episode_horizon`` the finalizer emits ``truncated=True``.
            Must be > 0.  Owned here; never forwarded to the controller.
    """

    halt_action: int = Field(
        default=0,
        ge=0,
        description="Action index that signals episode termination for MazeHard.",
    )
    episode_horizon: int = Field(
        default=16,
        ge=1,
        description="Task-owned semantic step budget per slot; finalizer emits truncated when steps reach this value.",
    )


# =============================================================================
class MazeHardDeliberationFinalizer:
    """MazeHard implementation of :class:`~ehc_sn.controllers.deliberation.actor_critic.DeliberationStepFinalizer`.

    Owned by the task layer; injected into
    :class:`~ehc_sn.controllers.deliberation.actor_critic.DeliberationACController`
    at wiring time.

    Responsibilities:
        - Compute dense improvement reward from current task logits.
        - Mark per-slot termination when ``action == config.halt_action``.
        - Mark per-slot truncation when ``steps >= config.episode_horizon``.
        - Thread ``prev_accuracy`` as ``runtime_state`` across steps.

    Implements :class:`~ehc_sn.controllers.deliberation.actor_critic.DeliberationStepFinalizer`
    structurally (duck-typed; no Protocol inheritance required for runtime use).
    """

    def __init__(self, config: MazeHardDeliberationConfig) -> None:
        """Create the MazeHard deliberation finalizer.

        Args:
            config: Task-owned config specifying ``halt_action`` and ``episode_horizon``.
        """
        self._halt_action = config.halt_action
        self._episode_horizon = config.episode_horizon

    def finalize_step(
        self,
        data: Batch,
        task_output: object,
        action: Tensor,
        steps: Tensor,
        runtime_state: object | None,
    ) -> DeliberationStepResult:
        """Compute reward and termination for one MazeHard deliberation step.

        Args:
            data: Current per-slot batch dict with ``"labels"`` key (shape ``(B, S)``).
            task_output: Must be a :class:`~ehc_sn.tasks.mazehard.contracts.MazeHardTaskOutput`
                with a ``task_logits`` tensor of shape ``(B, S, V)``.
            action: Sampled action tensor of shape ``(B,)``.
            steps: Per-slot step counters of shape ``(B,)``.
            runtime_state: Previous ``prev_accuracy`` tensor of shape ``(B, 1)``,
                or ``None`` on the first step.

        Returns:
            :class:`~ehc_sn.controllers.deliberation.actor_critic.DeliberationStepResult` with:
                - ``reward``: shape ``(B, 1)``, ``float32``.
                - ``terminated``: ``action == halt_action``, shape ``(B,)``.
                - ``truncated``: ``steps >= episode_horizon``, shape ``(B,)``.
                - ``next_runtime_state``: updated ``prev_accuracy`` tensor, ``(B, 1)``.
        """
        assert isinstance(
            task_output, MazeHardTaskOutput
        ), f"MazeHardDeliberationFinalizer expects MazeHardTaskOutput, got {type(task_output).__name__}"
        labels: Tensor = data["labels"]
        prev_accuracy: Tensor | None = runtime_state if isinstance(runtime_state, Tensor) else None

        accuracy, reward = _compute_improvement_reward(
            task_output,
            labels,
            prev_accuracy=prev_accuracy,
        )

        terminated = action.eq(self._halt_action)
        truncated = steps >= self._episode_horizon

        return DeliberationStepResult(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            next_runtime_state=accuracy.detach(),
        )


# =============================================================================
__all__ = [
    "MazeHardDeliberationFinalizer",
    "MazeHardDeliberationConfig",
]
