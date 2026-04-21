"""MazeHard task-local runtime helpers.

These helpers own raw-to-canonical MazeHard field coercion and extraction of
task dataclasses from generic batch mappings so controller code remains
task-agnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor

from ehc_sn.data.schema import CHANNEL_SOLUTION, O_ID, validate_npz
from ehc_sn.data.transforms import channels_to_grid
from ehc_sn.types import Batch

from .contracts import MazeHardTargets, MazeHardTaskInput
from .evaluation import compute_maze_hard_improvement_reward

MAZE_HARD_BATCH_KEYS: Final[tuple[str, ...]] = ("input_ids", "labels")
"""Canonical generic batch keys required by MazeHard rollout consumers."""


# =============================================================================
class MazeHardControllerRuntime:
    """Controller-facing MazeHard rollout runtime.

    This object owns the task-shaped TensorDict contracts required by the
    MazeHard reinforcement-learning path so the controller remains generic over
    batch-key semantics while MazeHard reward semantics stay task-owned.
    """

    def build_reset_td(  # ----------------------------------------------------
        self,
        batch: Batch,
    ) -> TensorDict:
        """Return the environment reset TensorDict for one canonical MazeHard batch."""
        input_ids, labels = _validate_maze_hard_batch(batch)
        _ = labels
        batch_size = int(input_ids.shape[0])
        return TensorDict(
            {"input_ids": input_ids},
            batch_size=[batch_size],
            device=input_ids.device,
        )

    def build_env_step_td(  # -------------------------------------------------
        self,
        env_td: TensorDictBase,
        *,
        reset_mask: Tensor,
        action: Tensor,
        task_output: object,
        data: Batch,
    ) -> TensorDictBase:
        """Return the environment input TensorDict for one MazeHard rollout step."""
        _ = task_output
        input_ids, _ = _validate_maze_hard_batch(data)
        input_mask = _broadcast_reset_mask(reset_mask, env_td["input_ids"])
        step_mask = _broadcast_reset_mask(reset_mask, env_td["step_count"])
        step_action = action.unsqueeze(-1) if action.ndim == 1 else action
        step_input_ids = torch.where(input_mask, input_ids, env_td["input_ids"])
        step_count = torch.where(step_mask, torch.zeros_like(env_td["step_count"]), env_td["step_count"])
        return TensorDict(
            {
                "input_ids": step_input_ids,
                "step_count": step_count,
                "action": step_action,
            },
            batch_size=env_td.batch_size,
            device=env_td.device,
        )

    def finalize_env_transition(  # -------------------------------------------
        self,
        previous_env_td: TensorDictBase,
        next_env_td: TensorDictBase,
        *,
        reset_mask: Tensor,
        action: Tensor,
        task_output: object,
        data: Batch,
    ) -> TensorDictBase:
        """Attach task-owned reward metadata to the stepped environment state."""
        _ = action
        _, labels = _validate_maze_hard_batch(data)
        task_logits: Tensor = task_output.task_logits  # type: ignore[union-attr]

        if "prev_accuracy" in previous_env_td.keys():
            prev_accuracy = previous_env_td["prev_accuracy"]
        else:
            prev_accuracy = torch.zeros_like(next_env_td["step_count"], dtype=torch.float32)
        prev_accuracy = prev_accuracy.to(device=next_env_td.device, dtype=torch.float32)
        reset_view = _broadcast_reset_mask(reset_mask, prev_accuracy)
        prev_accuracy = torch.where(reset_view, torch.zeros_like(prev_accuracy), prev_accuracy)

        accuracy, reward = compute_maze_hard_improvement_reward(
            task_logits.detach().to(dtype=torch.float32),
            labels,
            prev_accuracy=prev_accuracy,
        )

        finalized = next_env_td.clone()
        finalized["prev_accuracy"] = accuracy.to(device=next_env_td.device)
        finalized["reward"] = reward.to(device=next_env_td.device)
        return finalized

    def extract_next_step_obs(self, carry: Any) -> Batch:  # ------------------
        """Extract the next-step observation batch from the post-step carry.

        MazeHard observations are static token sequences (the maze layout does
        not change between steps). The post-step ``env_td`` carries the current
        ``input_ids`` while the per-slot carry data retains canonical
        supervision labels for each slot.

        Args:
            carry: The :class:`~ehc_sn.controllers.rl.RLRolloutState` produced
                after the most recent controller step.

        Returns:
            A batch dict with ``"input_ids"`` and ``"labels"`` suitable for a
            follow-up backbone forward pass to compute the TD bootstrap value.
        """
        env_td: TensorDictBase = carry.env_td
        return {
            "input_ids": env_td["input_ids"],
            "labels": carry.data["labels"],
        }


# =============================================================================
def coerce_maze_hard_batch(  # ------------------------------------------------
    raw: Mapping[str, Any],
) -> Batch:
    """Convert raw MazeHard channels into canonical token and label tensors.

    Supports both single-maze arrays ``(H, W)`` and aligned stacked arrays
    ``(B, H, W)``. Spatial dimensions are flattened while any leading batch
    dimensions are preserved.
    """
    channels = _coerce_numpy_channels(raw)
    _validate_channel_stack_shapes(channels)

    grid = channels_to_grid(channels)["grid"]
    input_ids = _flatten_spatial_to_tensor(grid, dtype=np.int64, name="grid")
    labels = input_ids.clone()

    if CHANNEL_SOLUTION in channels:
        solution_mask = _flatten_spatial_to_tensor(
            channels[CHANNEL_SOLUTION] > 0,
            dtype=np.bool_,
            name=CHANNEL_SOLUTION,
        ).to(dtype=torch.bool)
        labels = torch.where(solution_mask, torch.full_like(labels, O_ID), labels)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


# =============================================================================
def extract_maze_hard_task_input(  # ------------------------------------------
    batch: Batch,
) -> MazeHardTaskInput:
    """Extract the model-facing MazeHard task input from one generic batch."""
    input_ids, _ = _validate_maze_hard_batch(batch)
    return MazeHardTaskInput(input_ids=input_ids)


# =============================================================================
def extract_maze_hard_targets(  # ---------------------------------------------
    batch: Batch,
) -> MazeHardTargets:
    """Extract MazeHard supervision targets from one generic batch."""
    _, labels = _validate_maze_hard_batch(batch)
    return MazeHardTargets(labels=labels)


# =============================================================================
def _coerce_numpy_channels(  # ------------------------------------------------
    raw: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Normalize one raw MazeHard channel mapping to numpy arrays."""
    channels: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            channels[key] = value
            continue
        if isinstance(value, Tensor):
            channels[key] = value.detach().cpu().numpy()
            continue
        raise TypeError(f"Unsupported MazeHard channel type for key {key!r}: {type(value).__name__}.")

    validate_npz(channels)
    return channels


# =============================================================================
def _validate_channel_stack_shapes(  # ----------------------------------------
    channels: dict[str, np.ndarray],
) -> None:
    """Validate that all raw MazeHard channels share the same full shape."""
    reference_name, reference = next(iter(channels.items()))
    mismatched = {name: value.shape for name, value in channels.items() if value.shape != reference.shape}
    if mismatched:
        detail = ", ".join(f"{name}={shape}" for name, shape in mismatched.items())
        raise ValueError(
            "MazeHard runtime requires aligned raw channel shapes; "
            f"expected all channels to match {reference_name}={reference.shape}, got {detail}."
        )


# =============================================================================
def _flatten_spatial_to_tensor(  # --------------------------------------------
    array: np.ndarray,
    *,
    dtype: Any,
    name: str,
) -> Tensor:
    """Flatten the spatial tail of a 2D or 3D MazeHard array to a torch tensor."""
    if array.ndim not in (2, 3):
        raise ValueError(f"MazeHard field {name!r} must have shape (H, W) or (B, H, W), got {array.shape}.")
    return torch.from_numpy(array.reshape(*array.shape[:-2], -1).astype(dtype, copy=False))


# =============================================================================
def _broadcast_reset_mask(  # -------------------------------------------------
    reset_mask: Tensor,
    target: Tensor,
) -> Tensor:
    """Return ``reset_mask`` reshaped to broadcast over ``target``."""
    if reset_mask.ndim != 1:
        raise ValueError(f"MazeHard reset_mask must have shape (B,), got {tuple(reset_mask.shape)}.")
    return reset_mask.view((-1,) + (1,) * (target.ndim - 1))


# =============================================================================
def _validate_maze_hard_batch(  # ---------------------------------------------
    batch: Batch,
) -> tuple[Tensor, Tensor]:
    """Validate and normalize the canonical MazeHard batch mapping."""
    missing = [key for key in MAZE_HARD_BATCH_KEYS if key not in batch]
    if missing:
        raise KeyError("MazeHard batch is missing required keys: " + ", ".join(missing) + ".")

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    if input_ids.ndim != 2:
        raise ValueError(f"MazeHard input_ids must have shape (B, S), got {tuple(input_ids.shape)}.")
    if labels.ndim != 2:
        raise ValueError(f"MazeHard labels must have shape (B, S), got {tuple(labels.shape)}.")
    if tuple(labels.shape) != tuple(input_ids.shape):
        raise ValueError(f"MazeHard labels must match input_ids shape {tuple(input_ids.shape)}, got {tuple(labels.shape)}.")

    return input_ids.to(dtype=torch.int64), labels.to(dtype=torch.int64)


# =============================================================================
__all__ = [
    "MAZE_HARD_BATCH_KEYS",
    "MazeHardControllerRuntime",
    "coerce_maze_hard_batch",
    "extract_maze_hard_targets",
    "extract_maze_hard_task_input",
]
