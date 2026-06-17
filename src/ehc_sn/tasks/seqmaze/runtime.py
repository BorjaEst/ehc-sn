"""SeqMaze task-level runtime helpers and TaskRuntime implementation.

Owns typed extraction of probe and v1 dataclasses from generic batch mappings,
and provides :class:`SeqMazeRuntime` — the
:class:`~ehc_sn.contracts.task_runtime.TaskRuntime` implementation for
actor-critic deliberation on static DAG problems.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.contracts.task_runtime import (
    RuntimeReset,
    StepFeedback,
    TaskRuntime,
)
from ehc_sn.tasks.seqmaze.contracts import (
    SeqMazeProbeInput,
    SeqMazeProbeTargets,
    SeqMazeTargets,
    SeqMazeTaskInput,
    SeqMazeTaskOutput,
)
from ehc_sn.tasks.seqmaze.evaluation import build_seqmaze_step_score
from ehc_sn.tasks.seqmaze.reward import SeqMazeRewardProjector
from ehc_sn.training.hrm import ValidationRuntimeConfig
from ehc_sn.types import Batch

# Canonical batch keys for the seqmaze probe.
SEQUENCE_BATCH_KEYS: Final[tuple[str, ...]] = (
    "node_obs_id",
    "node_candidate_index",
    "node_start_flag",
    "node_goal_flag",
    "successor_indices",
    "successor_mask",
    "node_mask",
    "edge_label",
    "edge_mask",
    "target_path",
    "path_mask",
    "path_length",
)

# Additional batch keys required by the v1 path-prediction task.
SEQUENCE_MAX_V1_BATCH_KEYS: Final[tuple[str, ...]] = (
    "target_path",
    "path_mask",
    "path_length",
)


# =============================================================================
def extract_seqmaze_probe_input(batch: Batch) -> SeqMazeProbeInput:
    """Extract the probe input fields from one generic batch mapping."""
    _validate_batch(batch)
    return SeqMazeProbeInput(
        node_obs_id=batch["node_obs_id"].to(dtype=torch.int64),
        node_candidate_index=batch["node_candidate_index"].to(
            dtype=torch.int64
        ),
        node_start_flag=batch["node_start_flag"].to(dtype=torch.bool),
        node_goal_flag=batch["node_goal_flag"].to(dtype=torch.bool),
        successor_indices=batch["successor_indices"].to(dtype=torch.int64),
        successor_mask=batch["successor_mask"].to(dtype=torch.bool),
        node_mask=batch["node_mask"].to(dtype=torch.bool),
    )


# =============================================================================
def extract_seqmaze_probe_targets(batch: Batch) -> SeqMazeProbeTargets:
    """Extract the probe supervision targets from one generic batch mapping."""
    _validate_batch(batch)
    return SeqMazeProbeTargets(
        edge_label=batch["edge_label"].to(dtype=torch.int64),
        edge_mask=batch["edge_mask"].to(dtype=torch.bool),
    )


# =============================================================================
def _validate_batch(batch: Batch) -> None:
    """Ensure all required keys are present."""
    missing = [key for key in SEQUENCE_BATCH_KEYS if key not in batch]
    if missing:
        raise KeyError(
            "SeqMaze probe batch is missing required keys: "
            + ", ".join(missing)
            + "."
        )


# =============================================================================
def extract_seqmaze_targets(batch: Batch) -> SeqMazeTargets:
    """Extract v1 path supervision targets from one generic batch mapping."""
    for key in SEQUENCE_MAX_V1_BATCH_KEYS:
        if key not in batch:
            raise KeyError(
                f"SeqMaze v1 batch is missing required key: {key!r}."
            )
    return SeqMazeTargets(
        path_index=batch["target_path"].to(dtype=torch.int64),
        path_mask=batch["path_mask"].to(dtype=torch.bool),
        path_length=batch["path_length"].to(dtype=torch.int64),
    )


# =============================================================================
def extract_seqmaze_task_input(batch: Batch) -> SeqMazeTaskInput:
    """Extract v1 task input from one generic batch mapping.

    Uses the same graph-structure fields as the probe input.
    """
    _validate_batch(batch)
    return SeqMazeTaskInput(
        node_obs_id=batch["node_obs_id"].to(dtype=torch.int64),
        node_candidate_index=batch["node_candidate_index"].to(
            dtype=torch.int64
        ),
        node_start_flag=batch["node_start_flag"].to(dtype=torch.bool),
        node_goal_flag=batch["node_goal_flag"].to(dtype=torch.bool),
        successor_indices=batch["successor_indices"].to(dtype=torch.int64),
        successor_mask=batch["successor_mask"].to(dtype=torch.bool),
        node_mask=batch["node_mask"].to(dtype=torch.bool),
    )


# =============================================================================
# SeqMazeRuntime — TaskRuntime implementation for static-DAG deliberation
# =============================================================================


class SeqMazeRuntimeConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`SeqMazeRuntime`.

    Attributes:
        halt_action: Action index the model uses to signal 'done'.
            Must match the action space in the backbone/policy head.
        episode_horizon: Semantic step budget per slot.  When
            ``steps >= episode_horizon`` the runtime emits ``truncated=True``.
    """

    halt_action: int = Field(
        default=0,
        ge=0,
        description="Action index that signals episode termination.",
    )
    episode_horizon: int = Field(
        default=16,
        ge=1,
        description="Task-owned step budget; runtime emits truncated "
        "when steps reach this value.",
    )
    validation: ValidationRuntimeConfig = Field(
        default_factory=ValidationRuntimeConfig,
        description="Runner-owned safety limits (max steps, seed).",
    )


# =============================================================================
@dataclass
class _SeqMazeRuntimeState:
    """Internal runtime state for SeqMaze static deliberation.

    Attributes:
        step_count: Per-slot step counter of shape ``(B,)`` int32.
        static_data: The static SeqMaze batch (all graph + target fields).
            Never changes across steps — SeqMaze is a static-instance task.
    """

    step_count: Tensor  # (B,) int32
    static_data: Batch


# =============================================================================
class SeqMazeRuntime(TaskRuntime["_SeqMazeRuntimeState"]):
    """SeqMaze implementation of
    :class:`~ehc_sn.contracts.task_runtime.TaskRuntime`.

    The SeqMaze problem is a static directed acyclic graph.  The observation
    (graph structure, start/goal flags) is fixed across deliberation steps.
    What changes is the model's internal state and its predicted path.

    Responsibilities:
        - Own runtime state: step counter plus static SeqMaze batch.
        - Delegate reward computation to :class:`SeqMazeRewardProjector`.
        - Mark per-slot termination when ``action == config.halt_action``.
        - Mark per-slot truncation when ``steps >= config.episode_horizon``.
        - Return the static observation unchanged on every step.
        - Support partial reset via ``reset_slots``.
    """

    def __init__(
        self,
        config: SeqMazeRuntimeConfig,
        reward_projector: SeqMazeRewardProjector,
    ) -> None:
        if reward_projector is None:
            raise ValueError("SeqMazeRuntime requires a reward_projector")
        self._halt_action = config.halt_action
        self._episode_horizon = config.episode_horizon
        self._reward_projector = reward_projector
        # Derived from static batch at reset time
        self._eos_id: int | None = None
        self._pad_id: int | None = None
        self._n_max: int | None = None

    def reset(
        self,
        batch: Batch,
    ) -> RuntimeReset["_SeqMazeRuntimeState"]:
        """Initialise runtime for a full batch of new SeqMaze problems."""
        B = _seqmaze_batch_size(batch)
        device = _seqmaze_batch_device(batch)
        self._n_max = int(batch["node_mask"].shape[1])
        self._eos_id = self._n_max
        self._pad_id = self._n_max + 1
        state = _SeqMazeRuntimeState(
            step_count=torch.zeros(B, dtype=torch.int32, device=device),
            static_data=batch,
        )
        return RuntimeReset(observation=batch, state=state)

    def reset_slots(
        self,
        reset_mask: Tensor,
        batch: Batch,
        state: _SeqMazeRuntimeState,
    ) -> RuntimeReset["_SeqMazeRuntimeState"]:
        """Reset halted slots with fresh problems; preserve continuing slots."""
        B = _seqmaze_batch_size(batch)
        device = _seqmaze_batch_device(batch)

        step_count = torch.where(
            reset_mask,
            torch.zeros(B, dtype=torch.int32, device=device),
            state.step_count,
        )

        # Merge static data: reset slots get new batch entries
        merged_data: dict[str, Tensor] = {}
        for key in state.static_data:
            val = state.static_data[key]
            if isinstance(val, Tensor):
                reshape = [-1] + [1] * (val.ndim - 1) if val.ndim > 1 else [-1]
                merged_data[key] = torch.where(
                    (
                        reset_mask.reshape(*reshape).expand_as(val)
                        if val.ndim > 1
                        else reset_mask
                    ),
                    batch[key] if key in batch else val,
                    val,
                )
            else:
                merged_data[key] = val

        new_state = _SeqMazeRuntimeState(
            step_count=step_count,
            static_data=merged_data,
        )
        return RuntimeReset(observation=merged_data, state=new_state)

    def step(
        self,
        state: _SeqMazeRuntimeState,
        task_output: object,
        action: Tensor,
        steps: Tensor,
    ) -> StepFeedback["_SeqMazeRuntimeState"]:
        """Advance the runtime by one deliberation step.

        Sequence:
            1. Increment step counters.
            2. Compute task evaluation (SeqMazeStepScore) from current
               path prediction.
            3. Project reward via SeqMazeRewardProjector.
            4. Determine termination (halt action) and truncation (horizon).
            5. Return static observation unchanged.
        """
        if self._eos_id is None or self._pad_id is None or self._n_max is None:
            raise RuntimeError(
                "SeqMazeRuntime.reset() must be called before step()."
            )

        # 1. Increment step counters
        new_step_count = state.step_count + 1

        # 2. Compute step score
        if not hasattr(task_output, "path_logits"):
            raise TypeError(
                f"SeqMazeRuntime.step expects task_output with path_logits, "
                f"got {type(task_output).__name__}."
            )

        batch = state.static_data
        targets = SeqMazeTargets(
            path_index=batch["target_path"].to(dtype=torch.int64),
            path_mask=batch["path_mask"].to(dtype=torch.bool),
            path_length=batch["path_length"].to(dtype=torch.int64),
        )
        goal_idx = batch["node_goal_flag"].to(dtype=torch.bool)

        step_score = build_seqmaze_step_score(
            task_output,
            targets,
            successor_indices=batch["successor_indices"].to(dtype=torch.int64),
            successor_mask=batch["successor_mask"].to(dtype=torch.bool),
            goal_candidate_index=goal_idx,
            eos_id=self._eos_id,
            pad_id=self._pad_id,
            n_max=self._n_max,
        )

        # 3. Determine termination and truncation
        terminated = action == self._halt_action
        truncated = new_step_count >= self._episode_horizon

        # 4. Project reward
        reward = self._reward_projector.project_step_reward(
            step_score,
            terminated=terminated,
            truncated=truncated,
        )

        # 5. Build next state — observation is static (unchanged batch)
        new_state = _SeqMazeRuntimeState(
            step_count=new_step_count,
            static_data=batch,
        )

        return StepFeedback(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            next_observation=batch,
            next_state=new_state,
        )


# =============================================================================
def _seqmaze_batch_size(batch: Batch) -> int:
    """Return the leading batch size from any seqmaze tensor."""
    for v in batch.values():
        if isinstance(v, Tensor):
            return int(v.shape[0])
    raise ValueError("Cannot determine batch size from empty batch.")


def _seqmaze_batch_device(batch: Batch) -> torch.device:
    """Return the device of the first tensor in the batch."""
    for v in batch.values():
        if isinstance(v, Tensor):
            return v.device
    raise ValueError("Cannot determine device from empty batch.")
