"""MazeHard deliberation-family step finalizer.

This module owns the :class:`DeliberationStepFinalizer` implementation for
MazeHard.  It is entirely separate from the online RL runtime helper in
:mod:`~ehc_sn.tasks.mazehard.runtime` so the two seams remain orthogonal.

Reward:        dense improvement reward ``exp(acc_t) - exp(acc_{t-1})``.
Termination:   per-slot when ``action == config.halt_action`` (learned halt).
Truncation:    per-slot when ``steps >= config.episode_horizon`` (task-owned semantic horizon).
Runtime state: ``prev_accuracy`` tensor of shape ``(B, 1)`` threaded across steps.
"""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers.deliberation.actor_critic import DeliberationStepFinalizer, DeliberationStepResult
from ehc_sn.tasks.mazehard.contracts import MazeHardTaskOutput
from ehc_sn.tasks.mazehard.evaluation import compute_maze_hard_improvement_reward
from ehc_sn.types import Batch


# =============================================================================
class MazeHardDeliberationTaskConfig(BaseModel, extra="forbid"):
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
        - Mark per-slot termination when ``action == config.halt_action`` (learned halt).
        - Mark per-slot truncation when ``steps >= config.episode_horizon`` (task-owned semantic horizon).
        - Thread ``prev_accuracy`` as ``runtime_state`` across steps.

    Implements :class:`~ehc_sn.controllers.deliberation.actor_critic.DeliberationStepFinalizer`
    structurally (duck-typed; no Protocol inheritance required for runtime use).
    """

    def __init__(self, config: MazeHardDeliberationTaskConfig) -> None:
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
            steps: Per-slot step counters of shape ``(B,)`` (not used here; present
                for protocol conformance).
            runtime_state: Previous ``prev_accuracy`` tensor of shape ``(B, 1)``,
                or ``None`` on the first step.

        Returns:
            :class:`~ehc_sn.controllers.deliberation.actor_critic.DeliberationStepResult` with:
                - ``reward``: shape ``(B, 1)``, ``float32``.
                - ``terminated``: ``action == halt_action``, shape ``(B,)``.
                - ``truncated``: ``steps >= episode_horizon``, shape ``(B,)``.
                - ``next_runtime_state``: updated ``prev_accuracy`` tensor, ``(B, 1)``.
        """
        assert isinstance(task_output, MazeHardTaskOutput), (
            f"MazeHardDeliberationFinalizer expects MazeHardTaskOutput, got {type(task_output).__name__}"
        )
        labels: Tensor = data["labels"]

        prev_accuracy: Tensor | None = runtime_state if isinstance(runtime_state, Tensor) else None

        accuracy, reward = compute_maze_hard_improvement_reward(
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
    "MazeHardDeliberationTaskConfig",
]
