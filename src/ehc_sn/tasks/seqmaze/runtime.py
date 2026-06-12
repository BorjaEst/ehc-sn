"""SeqMaze task-level runtime helpers.

Owns typed extraction of probe and v1 dataclasses from generic batch mappings.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch
from torch import Tensor

from ehc_sn.types import Batch

from .contracts import (
    SeqMazeProbeInput,
    SeqMazeProbeTargets,
    SeqMazeTargets,
    SeqMazeTaskInput,
)

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
