"""MazeHard sample inspection — human-oriented decoded view of a corpus sample.

Produces :class:`MazeHardSampleInspection` from raw persisted arrays.
Pattern mirrors ``goaltrace/inspection.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# =============================================================================
@dataclass(frozen=True)
class MazeHardSampleInspection:
    """Human-oriented decoded view of one mazehard corpus sample.

    Attributes:
        sample_id: Human-readable sample identifier.
        split: Dataset split label.
        index: Sample index within the split.
        grid_shape: (H, W) grid dimensions.
        wall_density: Fraction of cells that are walls.
        solution_length: Number of solution-path cells.
        n_start_cells: Number of start markers (should be 1).
        n_goal_cells: Number of goal markers (should be ≥ 1).
        warnings: Non-fatal anomalies.
    """

    sample_id: str
    split: str
    index: int

    grid_shape: tuple[int, int]
    wall_density: float
    solution_length: int
    n_start_cells: int
    n_goal_cells: int

    warnings: list[str] = field(default_factory=list)


# =============================================================================
def prepare_sample_inspection(
    sample: dict[str, np.ndarray],
    split: str = "",
    index: int = -1,
) -> MazeHardSampleInspection:
    """Build a :class:`MazeHardSampleInspection` from raw persisted arrays.

    Args:
        sample: Channel dict for one sample.
        split: Split label.
        index: Sample index within the split.

    Returns:
        Filled :class:`MazeHardSampleInspection`.

    Raises:
        KeyError: When a required channel is missing.
    """
    required = ("topology", "start", "goals", "solution")
    for k in required:
        if k not in sample:
            raise KeyError(
                f"prepare_sample_inspection: '{k}' missing from sample."
            )

    topology: NDArray[np.bool_] = np.asarray(sample["topology"])
    start: NDArray[np.bool_] = np.asarray(sample["start"])
    goals: NDArray[np.bool_] = np.asarray(sample["goals"])
    solution: NDArray[np.int32] | NDArray[np.bool_] = np.asarray(
        sample["solution"]
    )

    H, W = topology.shape
    wall_density = float(topology.sum()) / float(H * W)
    solution_length = int((solution > 0).sum())
    n_start_cells = int(start.sum())
    n_goal_cells = int(goals.sum())

    warnings: list[str] = []
    if n_start_cells != 1:
        warnings.append(f"Expected 1 start cell, got {n_start_cells}.")
    if n_goal_cells < 1:
        warnings.append(f"Expected at least 1 goal cell, got {n_goal_cells}.")

    sample_id = f"{split}/{index}"

    return MazeHardSampleInspection(
        sample_id=sample_id,
        split=split,
        index=index,
        grid_shape=(H, W),
        wall_density=wall_density,
        solution_length=solution_length,
        n_start_cells=n_start_cells,
        n_goal_cells=n_goal_cells,
        warnings=warnings,
    )


__all__ = [
    "MazeHardSampleInspection",
    "prepare_sample_inspection",
]
