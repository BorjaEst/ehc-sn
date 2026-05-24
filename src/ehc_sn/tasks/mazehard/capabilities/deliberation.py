"""MazeHard deliberation capability object and config.

Canonical owner of the :class:`MazeHardDeliberationConfig` and
:class:`MazeHardDeliberationCapability` that wire MazeHard task semantics into
the deliberation actor-critic controller.  This is an *execution binding*, not
a task-identity definition.  MazeHard semantics (score, evaluation, contracts)
live in the parent task package.

Task-owned reward semantics live in
:mod:`ehc_sn.tasks.mazehard.reward` (:class:`~ehc_sn.tasks.mazehard.reward.MazeHardRewardProjector`).
This capability delegates reward computation to the projector and keeps only
terminated, truncated, reward emission, and runtime-state threading.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers.deliberation.actor_critic import DeliberationStepResult
from ehc_sn.tasks.mazehard.contracts import MazeHardTaskOutput
from ehc_sn.tasks.mazehard.evaluation import build_maze_hard_step_score
from ehc_sn.tasks.mazehard.reward import MazeHardRewardProjector
from ehc_sn.types import Batch


# =============================================================================
class MazeHardDeliberationConfig(BaseModel, extra="forbid"):
    """Mode-scoped configuration for MazeHard deliberation capability.

    This is capability config, not task identity.  Task-owned semantics
    (score, evaluation, reward) live in the parent package.

    Attributes:
        halt_action: Action index the model uses to signal 'done' for a slot.
            Must match the action space configured in the backbone / policy head.
        episode_horizon: Semantic step budget per slot.  When
            ``steps >= episode_horizon`` the capability emits ``truncated=True``.
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
        description="Task-owned semantic step budget per slot; capability emits truncated when steps reach this value.",
    )


# =============================================================================
class MazeHardDeliberationCapability:
    """MazeHard implementation of :class:`~ehc_sn.controllers.deliberation.actor_critic.DeliberationStepFinalizer`.

    Owned by the task layer; injected into
    :class:`~ehc_sn.controllers.deliberation.actor_critic.DeliberationACController`
    at wiring time.

    Responsibilities:
        - Delegate reward computation to :class:`~ehc_sn.tasks.mazehard.reward.MazeHardRewardProjector`.
        - Mark per-slot termination when ``action == config.halt_action``.
        - Mark per-slot truncation when ``steps >= config.episode_horizon``.
        - No reward-local runtime state is threaded across steps.

    Implements :class:`~ehc_sn.controllers.deliberation.actor_critic.DeliberationStepFinalizer`
    structurally (duck-typed; no Protocol inheritance required for runtime use).
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: MazeHardDeliberationConfig,
        reward_projector: MazeHardRewardProjector,
    ) -> None:
        """Create the MazeHard deliberation capability.

        Args:
            config: Task-owned config specifying ``halt_action`` and ``episode_horizon``.
            reward_projector: Task-owned reward projector, injected at wiring time.
                Lives in :mod:`ehc_sn.tasks.mazehard.reward`; the capability
                does not construct it internally.
        """
        self._halt_action = config.halt_action
        self._episode_horizon = config.episode_horizon
        self._reward_projector = reward_projector

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
            runtime_state: Unused runtime carry for MazeHard reward semantics.

        Returns:
            :class:`~ehc_sn.controllers.deliberation.actor_critic.DeliberationStepResult` with:
                - ``reward``: shape ``(B, 1)``, ``float32``.
                - ``terminated``: ``action == halt_action``, shape ``(B,)``.
                - ``truncated``: ``steps >= episode_horizon``, shape ``(B,)``.
                - ``next_runtime_state``: ``None`` (stateless reward projection).
        """
        assert isinstance(
            task_output, MazeHardTaskOutput
        ), f"MazeHardDeliberationCapability expects MazeHardTaskOutput, got {type(task_output).__name__}"
        labels: Tensor = data["labels"]
        terminated = action.eq(self._halt_action)
        truncated = steps >= self._episode_horizon

        step_score = build_maze_hard_step_score(task_output, labels)
        reward = self._reward_projector.project_step_reward(
            step_score,
            terminated=terminated,
            truncated=truncated,
        )

        return DeliberationStepResult(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            next_runtime_state=None,
        )


# =============================================================================
__all__ = [
    "MazeHardDeliberationCapability",
    "MazeHardDeliberationConfig",
]
