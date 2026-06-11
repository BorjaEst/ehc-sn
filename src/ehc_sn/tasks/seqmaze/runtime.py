"""SeqMaze task-level runtime helpers.

Owns typed extraction of probe dataclasses from generic batch mappings.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch
from torch import Tensor

from ehc_sn.types import Batch

from .contracts import SeqMazeProbeInput, SeqMazeProbeTargets

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
