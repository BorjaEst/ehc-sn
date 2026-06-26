# SPDX-License-Identifier: MIT
"""MazeHard step evaluator — task-owned TaskStepEvaluator implementation.

Canonical owner of :class:`MazeHardStepEvaluatorConfig` and
:class:`MazeHardStepEvaluator` that wire MazeHard task semantics into
the deliberation actor-critic controller via
:class:`~ehc_sn.contracts.task_step.TaskStepEvaluator`.

Task-owned reward semantics live in
:mod:`ehc_sn.tasks.mazehard.reward` (:class:`~ehc_sn.tasks.mazehard.reward.MazeHardRewardProjector`).
This evaluator delegates reward computation to the projector and keeps only
terminated, truncated, reward emission, and runtime-state threading.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.contracts.task_step import StepEvaluation, TaskStepEvaluator
from ehc_sn.tasks.mazehard.contracts import MazeHardTaskOutput
from ehc_sn.tasks.mazehard.evaluation import build_maze_hard_step_score
from ehc_sn.tasks.mazehard.reward import MazeHardRewardProjector
from ehc_sn.types import Batch


# =============================================================================
class MazeHardStepEvaluatorConfig(BaseModel, extra="forbid"):
    """Configuration for MazeHard step evaluation.

    Attributes:
        halt_action: Action index the model uses to signal 'done' for a slot.
            Must match the action space configured in the backbone / policy head.
        episode_horizon: Semantic step budget per slot.  When
            ``steps >= episode_horizon`` the evaluator emits ``truncated=True``.
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
        description="Task-owned semantic step budget per slot; evaluator emits truncated when steps reach this value.",
    )


# =============================================================================
class MazeHardStepEvaluator(TaskStepEvaluator):
    """MazeHard implementation of :class:`~ehc_sn.contracts.task_step.TaskStepEvaluator`.

    Owned by the task layer; injected into
    :class:`~ehc_sn.controllers.deliberation.q_halting.DeliberationQHaltingController`
    at wiring time.

    Responsibilities:
        - Delegate reward computation to :class:`~ehc_sn.tasks.mazehard.reward.MazeHardRewardProjector`.
        - Mark per-slot termination when ``action == config.halt_action``.
        - Mark per-slot truncation when ``steps >= config.episode_horizon``.
        - No reward-local runtime state is threaded across steps.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: MazeHardStepEvaluatorConfig,
        reward_projector: MazeHardRewardProjector,
    ) -> None:
        """Create the MazeHard step evaluator.

        Args:
            config: Task-owned config specifying ``halt_action`` and ``episode_horizon``.
            reward_projector: Task-owned reward projector, injected at wiring time.
                Lives in :mod:`ehc_sn.tasks.mazehard.reward`; the evaluator
                does not construct it internally.
        """
        self._halt_action = config.halt_action
        self._episode_horizon = config.episode_horizon
        self._reward_projector = reward_projector

    def evaluate_step(
        self,
        data: Batch,
        task_output: object,
        action: Tensor,
        steps: Tensor,
        runtime_state: object | None,
    ) -> StepEvaluation:
        """Compute reward and termination for one MazeHard deliberation step.

        Args:
            data: Current per-slot batch dict with ``"labels"`` key (shape ``(B, S)``).
            task_output: Must be a :class:`~ehc_sn.tasks.mazehard.contracts.MazeHardTaskOutput`
                with a ``task_logits`` tensor of shape ``(B, S, V)``.
            action: Sampled action tensor of shape ``(B,)``.
            steps: Per-slot step counters of shape ``(B,)``.
            runtime_state: Unused runtime carry for MazeHard reward semantics.

        Returns:
            :class:`~ehc_sn.contracts.task_step.StepEvaluation` with:
                - ``reward``: shape ``(B, 1)``, ``float32``.
                - ``terminated``: ``action == halt_action``, shape ``(B,)``.
                - ``truncated``: ``steps >= episode_horizon``, shape ``(B,)``.
                - ``metrics``: dict with ``"step_score"`` tensor.
                - ``next_runtime_state``: ``None`` (stateless reward projection).
        """
        if not isinstance(task_output, MazeHardTaskOutput):
            raise TypeError(
                f"MazeHardStepEvaluator.evaluate_step expects MazeHardTaskOutput, "
                f"got {type(task_output).__name__}"
            )
        labels: Tensor = data["labels"]
        terminated = action.eq(self._halt_action)
        truncated = steps >= self._episode_horizon

        step_score = build_maze_hard_step_score(task_output, labels)
        reward = self._reward_projector.project_step_reward(
            step_score,
            terminated=terminated,
            truncated=truncated,
        )

        return StepEvaluation(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            next_runtime_state=None,
            metrics={"step_score": step_score},
        )


# =============================================================================
__all__ = [
    "MazeHardStepEvaluator",
    "MazeHardStepEvaluatorConfig",
]
