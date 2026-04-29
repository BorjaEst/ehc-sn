"""Task-owned channel derivation from MazeNdSourceRecord.

Converts a :class:`~ehc_sn.data.substrate.maze_nd.MazeNdSourceRecord` into
the MazeHard task-owned channels (start, goals, solution).  Shared channels
(topology, mask_valid) come from the parent substrate and are not produced here.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from ehc_sn.data.substrate.maze_nd import MazeNdSourceRecord

_SOLUTION_CHARS: frozenset[str] = frozenset({"o", "S", "G"})


def source_record_to_task_channels(record: MazeNdSourceRecord) -> dict[str, ndarray]:
    """Derive task-owned channels from a :class:`MazeNdSourceRecord`.

    Args:
        record: Raw source record for one maze puzzle.

    Returns:
        ``{"start": bool (H,W), "goals": bool (H,W), "solution": int32 (H,W)}``

    Raises:
        ValueError: When inputs/labels grids differ in shape or are not 2-D.
    """
    inputs = np.array(record["inputs"], dtype="U1")
    labels = np.array(record["labels"], dtype="U1")

    if inputs.ndim != 2:
        raise ValueError(f"MazeHard inputs must be a 2-D grid, got shape {inputs.shape}.")
    if inputs.shape != labels.shape:
        raise ValueError(f"MazeHard record has mismatched inputs/labels shapes: {inputs.shape} vs {labels.shape}.")

    start = inputs == "S"
    goals = inputs == "G"
    solution = ((labels == "o") | (labels == "S") | (labels == "G")).astype(np.int32)

    return {"start": start, "goals": goals, "solution": solution}


__all__ = ["source_record_to_task_channels"]
