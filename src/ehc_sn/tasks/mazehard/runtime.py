"""MazeHard task-level runtime helpers.

Owns typed extraction of task dataclasses from generic batch mappings and
raw channel-to-canonical-batch coercion so controller and script code
remains task-agnostic and adapter-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.contracts.task_runtime import (
    RuntimeReset,
    StepFeedback,
    TaskRuntime,
)
from ehc_sn.tasks.mazehard.contracts import (
    MazeHardTargets,
    MazeHardTaskInput,
    MazeHardTaskOutput,
)
from ehc_sn.tasks.mazehard.evaluation import build_maze_hard_step_score
from ehc_sn.tasks.mazehard.reward import MazeHardRewardProjector
from ehc_sn.training.hrm import ValidationRuntimeConfig
from ehc_sn.types import Batch

MAZE_HARD_BATCH_KEYS: Final[tuple[str, ...]] = ("input_ids", "labels")
"""Canonical generic batch keys required by MazeHard rollout consumers."""


# =============================================================================
def extract_maze_hard_task_input(
    batch: Batch,
) -> MazeHardTaskInput:
    """Extract the model-facing MazeHard task input from one generic batch."""
    input_ids, _ = _validate_maze_hard_batch(batch)
    return MazeHardTaskInput(input_ids=input_ids)


# =============================================================================
def extract_maze_hard_targets(
    batch: Batch,
) -> MazeHardTargets:
    """Extract MazeHard supervision targets from one generic batch."""
    _, labels = _validate_maze_hard_batch(batch)
    return MazeHardTargets(labels=labels)


# =============================================================================
def _validate_maze_hard_batch(
    batch: Batch,
) -> tuple[Tensor, Tensor]:
    """Validate and normalize the canonical MazeHard batch mapping."""
    missing = [key for key in MAZE_HARD_BATCH_KEYS if key not in batch]
    if missing:
        raise KeyError(
            "MazeHard batch is missing required keys: "
            + ", ".join(missing)
            + "."
        )

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    if input_ids.ndim != 2:
        raise ValueError(
            "MazeHard input_ids must have shape (B, S), "
            f"got {tuple(input_ids.shape)}."
        )
    if labels.ndim != 2:
        raise ValueError(
            "MazeHard labels must have shape (B, S), "
            f"got {tuple(labels.shape)}."
        )
    if tuple(labels.shape) != tuple(input_ids.shape):
        raise ValueError(
            f"MazeHard labels must have shape {tuple(input_ids.shape)}, "
            f"got {tuple(labels.shape)}."
        )

    return input_ids.to(dtype=torch.int64), labels.to(dtype=torch.int64)


# Private maze SEM vocabulary IDs used by batch coercion.
WALL_ID: int = 1
EMPTY_ID: int = 2
START_ID: int = 3
GOAL_ID: int = 4
PATH_ID: int = 5  # solution label token
SEM_VOCAB_SIZE: int = 5  # base semantic vocabulary (PAD..GOAL)
MAZE_HARD_VOCAB_SIZE: int = PATH_ID + 1  # full vocab including PATH

_MANDATORY_GRID2D_CHANNEL: str = "topology"
_CHANNEL_SOLUTION: str = "solution"


# =============================================================================
def coerce_maze_hard_batch(raw: Mapping[str, Any]) -> Batch:
    """Convert raw MazeHard channels into canonical token and label tensors.

    Task-owned tokenization: raw channel arrays -> ``{"input_ids", "labels"}``.
    Supports both single-maze arrays ``(H, W)`` and aligned stacked arrays
    ``(B, H, W)``.  Spatial dimensions are flattened while any leading batch
    dimensions are preserved.
    """
    channels = _coerce_numpy_channels(raw)
    _validate_channel_stack_shapes(channels)

    grid = _channels_to_grid(channels)
    input_ids = _flatten_spatial_to_tensor(grid, dtype=np.int64, name="grid")
    labels = input_ids.clone()

    if _CHANNEL_SOLUTION in channels:
        solution_mask = _flatten_spatial_to_tensor(
            channels[_CHANNEL_SOLUTION] > 0,
            dtype=np.bool_,
            name=_CHANNEL_SOLUTION,
        ).to(dtype=torch.bool)
        labels = torch.where(
            solution_mask, torch.full_like(labels, PATH_ID), labels
        )

    return {"input_ids": input_ids, "labels": labels}


def _channels_to_grid(channels: dict[str, np.ndarray]) -> np.ndarray:
    topology = channels[_MANDATORY_GRID2D_CHANNEL]
    grid = np.where(topology, EMPTY_ID, WALL_ID).astype(np.int32)
    if "start" in channels:
        grid = np.where(channels["start"], START_ID, grid)
    if "goals" in channels:
        grid = np.where(channels["goals"], GOAL_ID, grid)
    return grid


def _coerce_numpy_channels(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    channels: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            channels[key] = value
            continue
        if isinstance(value, Tensor):
            channels[key] = value.detach().cpu().numpy()
            continue
        raise TypeError(
            f"Unsupported MazeHard channel type for key {key!r}: {type(value).__name__}."
        )
    if _MANDATORY_GRID2D_CHANNEL not in channels:
        raise ValueError(
            f"MazeHard batch must contain the mandatory '{_MANDATORY_GRID2D_CHANNEL}' channel."
        )
    return channels


def _validate_channel_stack_shapes(channels: dict[str, np.ndarray]) -> None:
    reference_name, reference = next(iter(channels.items()))
    mismatched = {
        name: value.shape
        for name, value in channels.items()
        if value.shape != reference.shape
    }
    if mismatched:
        detail = ", ".join(
            f"{name}={shape}" for name, shape in mismatched.items()
        )
        raise ValueError(
            "MazeHard batch requires aligned raw channel shapes; "
            f"expected all channels to match {reference_name}={reference.shape}, got {detail}."
        )


def _flatten_spatial_to_tensor(
    array: np.ndarray, *, dtype: Any, name: str
) -> Tensor:
    if array.ndim not in (2, 3):
        raise ValueError(
            f"MazeHard field {name!r} must have shape (H, W) or (B, H, W), got {array.shape}."
        )
    return torch.from_numpy(
        array.reshape(*array.shape[:-2], -1).astype(dtype, copy=False)
    )


# =============================================================================
# MazeHardRuntime — TaskRuntime implementation
# =============================================================================


class MazeHardRuntimeConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`MazeHardRuntime`.

    Attributes:
        halt_action: Action index the model uses to signal 'done' for a slot.
            Must match the action space configured in the backbone / policy head.
        episode_horizon: Semantic step budget per slot.  When
            ``steps >= episode_horizon`` the runtime emits ``truncated=True``.
            Must be > 0.
    """

    halt_action: int = Field(
        default=0,
        ge=0,
        description="Action index that signals episode termination for MazeHard.",
    )
    episode_horizon: int = Field(
        default=16,
        ge=1,
        description="Task-owned semantic step budget per slot; runtime emits truncated when steps reach this value.",
    )
    validation: ValidationRuntimeConfig = Field(
        default_factory=ValidationRuntimeConfig,
        description="Runner-owned safety limits (max steps, seed).",
    )


# =============================================================================
@dataclass
class _MazeHardRuntimeState:
    """Internal runtime state for MazeHard deliberation.

    Attributes:
        step_count: Per-slot step counter of shape ``(B,)`` int32.
        static_data: The static MazeHard batch (``input_ids``, ``labels``).
            Never changes across steps — MazeHard is a static-instance task.
    """

    step_count: Tensor  # (B,) int32
    static_data: Batch


# =============================================================================
class MazeHardRuntime(TaskRuntime["_MazeHardRuntimeState"]):
    """MazeHard implementation of :class:`~ehp_sn.contracts.task_runtime.TaskRuntime`.

    Owned by the task layer; injected into
    :class:`~ehp_sn.controllers.deliberation.q_halting.DeliberationQHaltingController`
    at wiring time.

    Responsibilities:
        - Own runtime state: step counter plus static MazeHard batch.
        - Delegate reward computation to :class:`~ehp_sn.tasks.mazehard.reward.MazeHardRewardProjector`.
        - Mark per-slot termination when ``action == config.halt_action``.
        - Mark per-slot truncation when ``steps >= config.episode_horizon``.
        - Return the static observation unchanged on every step (static-instance
          deliberation).
        - Support partial reset via ``reset_slots``.

    Stateless contract: no reward-local runtime state is threaded across steps
    beyond the step counter and static data.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: MazeHardRuntimeConfig,
        reward_projector: MazeHardRewardProjector,
    ) -> None:
        """Create the MazeHard runtime.

        Args:
            config: Task-owned config specifying ``halt_action`` and
                ``episode_horizon``.
            reward_projector: Task-owned reward projector, injected at wiring
                time.  Lives in :mod:`ehp_sn.tasks.mazehard.reward`; the runtime
                does not construct it internally.
        """
        if reward_projector is None:
            raise ValueError("MazeHardRuntime requires a reward_projector")
        self._halt_action = config.halt_action
        self._episode_horizon = config.episode_horizon
        self._reward_projector = reward_projector

    def reset(  # -------------------------------------------------------------
        self,
        batch: Batch,
    ) -> RuntimeReset["_MazeHardRuntimeState"]:
        """Initialise runtime for a full batch of new MazeHard episodes.

        The initial observation IS the static batch (input_ids, labels).
        """
        B = _runtime_batch_size(batch)
        device = _runtime_batch_device(batch)
        state = _MazeHardRuntimeState(
            step_count=torch.zeros(B, dtype=torch.int32, device=device),
            static_data=batch,
        )
        return RuntimeReset(observation=batch, state=state)

    def reset_slots(  # -------------------------------------------------------
        self,
        reset_mask: Tensor,  # (B,) bool
        batch: Batch,
        state: _MazeHardRuntimeState,
    ) -> RuntimeReset["_MazeHardRuntimeState"]:
        """Reset halted slots with fresh episodes; preserve continuing slots."""
        B = _runtime_batch_size(batch)
        device = _runtime_batch_device(batch)

        step_count = torch.where(
            reset_mask,
            torch.zeros(B, dtype=torch.int32, device=device),
            state.step_count,
        )

        static_data = {
            key: torch.where(
                reset_mask.view((-1,) + (1,) * (value.ndim - 1)),
                batch[key],
                state.static_data[key],
            )
            for key, value in state.static_data.items()
        }

        new_state = _MazeHardRuntimeState(
            step_count=step_count,
            static_data=static_data,
        )
        return RuntimeReset(observation=static_data, state=new_state)

    def step(  # --------------------------------------------------------------
        self,
        state: _MazeHardRuntimeState,
        task_output: object,
        action: Tensor,  # (B,) int64
        steps: Tensor,  # (B,) int32
    ) -> StepFeedback["_MazeHardRuntimeState"]:
        """Compute reward and termination for one MazeHard deliberation step.

        Args:
            state: Current runtime state with static data and step count.
            task_output: Must be a
                :class:`~ehp_sn.tasks.mazehard.contracts.MazeHardTaskOutput`
                with a ``task_logits`` tensor of shape ``(B, S, V)``.
            action: Sampled action tensor of shape ``(B,)``.
            steps: Per-slot step counters of shape ``(B,)`` (post-advance).

        Returns:
            :class:`StepFeedback` with:
                - ``reward``: shape ``(B, 1)``, ``float32``.
                - ``terminated``: ``action == halt_action``, shape ``(B,)``.
                - ``truncated``: ``steps >= episode_horizon``, shape ``(B,)``.
                - ``next_observation``: same static data (unchanged).
                - ``next_state``: updated step count.
                - ``metrics``: dict with ``"step_score"`` tensor.
        """
        if not isinstance(task_output, MazeHardTaskOutput):
            raise TypeError(
                f"MazeHardRuntime.step expects MazeHardTaskOutput, "
                f"got {type(task_output).__name__}"
            )
        labels: Tensor = state.static_data["labels"]
        terminated = action.eq(self._halt_action)
        truncated = steps >= self._episode_horizon

        step_score = build_maze_hard_step_score(task_output, labels)
        reward = self._reward_projector.project_step_reward(
            step_score,
            terminated=terminated,
            truncated=truncated,
        )

        next_state = _MazeHardRuntimeState(
            step_count=state.step_count + 1,
            static_data=state.static_data,
        )

        return StepFeedback(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            next_observation=state.static_data,
            next_state=next_state,
            metrics={"step_score": step_score},
        )


# =============================================================================
def _runtime_batch_size(batch: Batch) -> int:
    """Infer batch size from the first tensor-valued entry."""
    for value in batch.values():
        if isinstance(value, Tensor):
            return int(value.shape[0])
    raise ValueError("Batch must contain at least one tensor-valued entry.")


def _runtime_batch_device(batch: Batch) -> torch.device:
    """Infer device from the first tensor-valued entry."""
    for value in batch.values():
        if isinstance(value, Tensor):
            return value.device
    raise ValueError("Batch must contain at least one tensor-valued entry.")


# =============================================================================
__all__ = [
    "MAZE_HARD_BATCH_KEYS",
    "MAZE_HARD_VOCAB_SIZE",
    "SEM_VOCAB_SIZE",
    "WALL_ID",
    "EMPTY_ID",
    "START_ID",
    "GOAL_ID",
    "PATH_ID",
    "MazeHardRuntime",
    "MazeHardRuntimeConfig",
    "coerce_maze_hard_batch",
    "extract_maze_hard_targets",
    "extract_maze_hard_task_input",
]
