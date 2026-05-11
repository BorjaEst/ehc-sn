"""MazeHard-specific channel transforms.

Provides :func:`channels_to_grid`, which merges ``topology``, ``start``, and
``goals`` channels into a single SEM-encoded ``int32`` grid.  This is a
task-owned transform — it uses MazeHard vocabulary IDs and must not live in
the storage-layer :mod:`ehc_sn.data.transforms` module.
"""

from __future__ import annotations

import numpy as np

from ehc_sn.tasks.mazehard.runtime import EMPTY_ID, GOAL_ID, START_ID, WALL_ID
from ehc_sn.types import Channels

_CHANNEL_TOPOLOGY: str = "topology"
"""Canonical topology channel name (matches ehc_sn.data.substrate.grid2d.CHANNEL_TOPOLOGY)."""


# =================================================================================================
def channels_to_grid(channels: Channels) -> Channels:
    """Merge structural channels into a single canonical semantic grid.

    Combines ``topology``, ``start``, and ``goals`` into a single ``int32``
    array ``"grid"`` of shape ``(H, W)`` using canonical SEM IDs.  The result
    is returned non-destructively alongside the original channels.

    Priority (later assignments win): ``WALL < EMPTY < START < GOAL``.

    .. note::
        ``"grid"`` is a **derived synthetic key** — it is not present on disk.

    Args:
        channels: Dict of canonical NPZ channel arrays.  Must contain
            ``"topology"`` (bool, H×W).  Optional: ``"start"``, ``"goals"``.

    Returns:
        Input dict extended with ``"grid": int32 array of shape (H, W)``.
    """
    topology = channels[_CHANNEL_TOPOLOGY]
    grid = np.where(topology, EMPTY_ID, WALL_ID).astype(np.int32)
    if "start" in channels:
        grid = np.where(channels["start"], START_ID, grid)
    if "goals" in channels:
        grid = np.where(channels["goals"], GOAL_ID, grid)
    return {**channels, "grid": grid}


# =================================================================================================
__all__ = ["channels_to_grid"]
