"""MazeHard HRM-family trace extras.

These fields bind MazeHard task semantics to HRM-family-specific output
surfaces. They live here because they read task-facing logits from the adapter
boundary while the generic trace infrastructure stays model-agnostic.

Usage
-----
    from ehc_sn.adapters.mazehard.hrm import (
        MAZE_HARD_HRM_Q_HALTING_TRACE_FIELDS,
        MAZE_HARD_HRM_ACT_TRACE_FIELDS,
    )
    self.trace_spec = build_trace_spec("act", extra_fields=MAZE_HARD_HRM_ACT_TRACE_FIELDS)
    self.trace_spec = build_trace_spec(
        "rl",
        extra_fields=MAZE_HARD_HRM_Q_HALTING_TRACE_FIELDS,
    )
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import Tensor

from ehc_sn.adapters.hrm._base import O_ID
from ehc_sn.traces import TraceField, TraceValue
from ehc_sn.traces.keys import (  # fmt: skip
    GOALTRACE_META_KEY_CURRENT_FLAG,
    GOALTRACE_META_KEY_GOAL_FLAG,
    GOALTRACE_META_KEY_NODE_MASK,
    GOALTRACE_META_KEY_OBSERVATION_ID,
    GOALTRACE_META_KEY_SUCCESSOR_INDICES,
    GOALTRACE_META_KEY_SUCCESSOR_MASK,
    GOALTRACE_META_KEY_TARGET_FIELD,
    GOALTRACE_META_KEY_WEIGHT,
    GOALTRACE_TRACE_KEY_FIRING_FIELD,
    MAZEHARD_META_KEY_GT_OVERLAY,
    MAZEHARD_META_KEY_INPUT_IDS,
    ROUTEBIND_META_KEY_CELL_MASK,
    ROUTEBIND_META_KEY_CELL_TYPE,
    ROUTEBIND_META_KEY_GOAL_FLAG,
    ROUTEBIND_META_KEY_OBSERVATION_ID,
    ROUTEBIND_META_KEY_START_FLAG,
    ROUTEBIND_META_KEY_TARGET_TRAJECTORY,
    ROUTEBIND_META_KEY_TARGET_WAYPOINT,
    SEQMAZE_META_KEY_N_NODES,
    SEQMAZE_META_KEY_NODE_GOAL_FLAG,
    SEQMAZE_META_KEY_NODE_START_FLAG,
    SEQMAZE_META_KEY_NODE_VALID_MASK,
    SEQMAZE_META_KEY_PATH_LENGTH,
    SEQMAZE_META_KEY_PATH_MASK,
    SEQMAZE_META_KEY_TARGET_PATH,
    SEQMAZE_META_KEY_TARGET_PATH_LEN,
)
from ehc_sn.traces.observer import StepContext
from ehc_sn.types import Batch

# =============================================================================
# Getter function
# =============================================================================


def _prediction_overlay_from_task_logits(task_logits: Tensor) -> TraceValue:
    """Return a binary overlay for the MazeHard solution-path token."""
    pred = torch.argmax(task_logits, dim=-1)  # (B, S)
    return (pred == O_ID).to(torch.uint8)


def _get_maze_hard_prediction_overlay_act(
    ctx: StepContext,
) -> TraceValue:
    """Read MazeHard solution-overlay traces from ACT backbone task logits."""
    return _prediction_overlay_from_task_logits(ctx.record.outputs.task.task_logits)


def _get_maze_hard_prediction_overlay_q_halting(
    ctx: StepContext,
) -> TraceValue:
    """Read MazeHard solution-overlay traces from Q-halting task output logits."""
    return _prediction_overlay_from_task_logits(
        ctx.record.outputs.task_output.task_logits
    )


def build_mazehard_hrm_trace_meta(batch: Batch) -> dict[str, object]:
    """Return out-of-band trace metadata required by MazeHard HRM figures."""
    root_key, leaf_key = MAZEHARD_META_KEY_GT_OVERLAY.split("/", maxsplit=1)
    return {
        root_key: {
            leaf_key: (batch["labels"] == O_ID).to(torch.uint8),
        },
        MAZEHARD_META_KEY_INPUT_IDS: batch["input_ids"],
    }


# =============================================================================
# Named field objects
# =============================================================================

MAZE_HARD_HRM_TRACE_PREDICTION_OVERLAY = TraceField(
    name="pred/prediction_overlay",
    get=_get_maze_hard_prediction_overlay_act,
)

MAZE_HARD_HRM_ACT_TRACE_FIELDS: tuple[TraceField, ...] = (
    MAZE_HARD_HRM_TRACE_PREDICTION_OVERLAY,
)

_MAZE_HARD_HRM_TRACE_PREDICTION_OVERLAY_Q_HALTING = TraceField(
    name="pred/prediction_overlay",
    get=_get_maze_hard_prediction_overlay_q_halting,
)

MAZE_HARD_HRM_Q_HALTING_TRACE_FIELDS: tuple[TraceField, ...] = (
    _MAZE_HARD_HRM_TRACE_PREDICTION_OVERLAY_Q_HALTING,
)


# =============================================================================
# Goaltrace HRM ACT trace fields
# =============================================================================


def _get_goaltrace_firing_field_act(
    ctx: StepContext,
) -> TraceValue:
    """Read goaltrace firing field from ACT backbone bridge output."""
    return ctx.record.outputs.task.firing_field


GOALTRACE_HRM_ACT_FIRING_FIELD: TraceField = TraceField(
    name=GOALTRACE_TRACE_KEY_FIRING_FIELD,
    get=_get_goaltrace_firing_field_act,
)

GOALTRACE_HRM_ACT_TRACE_FIELDS: tuple[TraceField, ...] = (
    GOALTRACE_HRM_ACT_FIRING_FIELD,
)


# =============================================================================
# SeqMaze trace metadata builder
# =============================================================================


def build_seqmaze_hrm_trace_meta(batch: Batch) -> dict[str, object]:
    """Return out-of-band trace metadata for SeqMaze HRM v1 figures.

    Includes minimal task context needed to interpret predictions without
    duplicating large model activations.

    Note: batch channel names (e.g. "node_mask") may differ from the trace
    meta key ("node_valid_mask").  The batch name is used for extraction;
    the constant provides the trace meta key name.
    """
    return {
        SEQMAZE_META_KEY_TARGET_PATH: batch[SEQMAZE_META_KEY_TARGET_PATH],
        SEQMAZE_META_KEY_PATH_MASK: batch[SEQMAZE_META_KEY_PATH_MASK],
        SEQMAZE_META_KEY_PATH_LENGTH: batch[SEQMAZE_META_KEY_PATH_LENGTH],
        SEQMAZE_META_KEY_NODE_VALID_MASK: batch["node_mask"],
        SEQMAZE_META_KEY_NODE_START_FLAG: batch[SEQMAZE_META_KEY_NODE_START_FLAG],
        SEQMAZE_META_KEY_NODE_GOAL_FLAG: batch[SEQMAZE_META_KEY_NODE_GOAL_FLAG],
    }


# =============================================================================
# SeqMaze HRM v2 Q-halting trace fields
# =============================================================================

SEQMAZE_HRM_Q_HALTING_TRACE_FIELDS: tuple[str, ...] = (
    "reward",
    "state_value",
    "q_values",
    "sampled_action",
    "terminated",
    "truncated",
)


def build_seqmaze_hrm_q_halting_trace_meta(
    batch: Batch,
) -> dict[str, object]:
    """Return out-of-band trace metadata for SeqMaze HRM v2 Q-halting figures."""
    return {
        SEQMAZE_META_KEY_N_NODES: int(batch["node_mask"].sum()),
        SEQMAZE_META_KEY_TARGET_PATH_LEN: (
            int(batch[SEQMAZE_META_KEY_PATH_LENGTH].max().item())
            if SEQMAZE_META_KEY_PATH_LENGTH in batch
            else 0
        ),
    }


# =============================================================================
# Goaltrace trace metadata builder
# =============================================================================

def build_goaltrace_hrm_trace_meta(
    batch: dict,
) -> dict:
    """Build trace metadata for goaltrace evaluation traces from a batch dict.

    Extracts all sample-constant metadata fields (input channels, targets,
    topology) from the collated batch and returns them as a flat dict of
    tensors ready for insertion into ``TraceTree.attached_meta``.

    The model-input keys (observation_id, weight, current_flag, goal_flag,
    node_mask) are projected explicitly.  Extra batch keys are ignored
    unless they are recognised target or topology keys.
    """
    # Model input channels
    meta: dict[str, object] = {
        GOALTRACE_META_KEY_OBSERVATION_ID: batch["observation_id"],
        GOALTRACE_META_KEY_WEIGHT: batch["weight"],
        GOALTRACE_META_KEY_CURRENT_FLAG: batch["current_flag"],
        GOALTRACE_META_KEY_GOAL_FLAG: batch["goal_flag"],
        GOALTRACE_META_KEY_NODE_MASK: batch["node_mask"],
    }

    # Target channel
    if "target_field" in batch:
        meta[GOALTRACE_META_KEY_TARGET_FIELD] = batch["target_field"]

    # Topology channels (evaluation metadata, not model input)
    if "successor_indices" in batch:
        meta[GOALTRACE_META_KEY_SUCCESSOR_INDICES] = batch["successor_indices"]
    if "successor_mask" in batch:
        meta[GOALTRACE_META_KEY_SUCCESSOR_MASK] = batch["successor_mask"]

    return meta


# =============================================================================
# Routebind HRM ACT trace fields
# =============================================================================


def _get_routebind_trajectory_field_act(
    ctx: StepContext,
) -> TraceValue:
    """Read routebind trajectory field from ACT backbone bridge output."""
    return ctx.record.outputs.task.trajectory_field


ROUTEBIND_HRM_ACT_TRAJECTORY_FIELD: TraceField = TraceField(
    name="routebind/trajectory_field",
    get=_get_routebind_trajectory_field_act,
)

ROUTEBIND_HRM_ACT_TRACE_FIELDS: tuple[TraceField, ...] = (
    ROUTEBIND_HRM_ACT_TRAJECTORY_FIELD,
)
"""Trace fields for Routebind × HRM ACT evaluation."""


def build_routebind_hrm_trace_meta(batch: dict) -> dict:
    """Build trace metadata for routebind evaluation traces from a batch dict.

    Extracts all sample-constant metadata fields (input channels, targets)
    from the collated batch and returns them as a flat dict of tensors
    ready for insertion into ``TraceTree.attached_meta``.
    """
    meta: dict[str, object] = {
        ROUTEBIND_META_KEY_CELL_TYPE: batch["cell_type"],
        ROUTEBIND_META_KEY_OBSERVATION_ID: batch["observation_id"],
        ROUTEBIND_META_KEY_START_FLAG: batch["start_flag"],
        ROUTEBIND_META_KEY_GOAL_FLAG: batch["goal_flag"],
        ROUTEBIND_META_KEY_CELL_MASK: batch["spatial_mask"],
    }

    if "target_trajectory" in batch:
        meta[ROUTEBIND_META_KEY_TARGET_TRAJECTORY] = batch["target_trajectory"]
    if "target_waypoint" in batch:
        meta[ROUTEBIND_META_KEY_TARGET_WAYPOINT] = batch["target_waypoint"]

    return meta


# =============================================================================
__all__ = [
    "build_mazehard_hrm_trace_meta",
    "build_seqmaze_hrm_trace_meta",
    "build_seqmaze_hrm_q_halting_trace_meta",
    "build_goaltrace_hrm_trace_meta",
    "build_routebind_hrm_trace_meta",
    "MAZE_HARD_HRM_Q_HALTING_TRACE_FIELDS",
    "MAZE_HARD_HRM_ACT_TRACE_FIELDS",
    "MAZE_HARD_HRM_TRACE_PREDICTION_OVERLAY",
    "ROUTEBIND_HRM_ACT_TRACE_FIELDS",
    "ROUTEBIND_HRM_ACT_TRAJECTORY_FIELD",
    "SEQMAZE_HRM_Q_HALTING_TRACE_FIELDS",
    "MAZEHARD_META_KEY_GT_OVERLAY",
    "MAZEHARD_META_KEY_INPUT_IDS",
]
