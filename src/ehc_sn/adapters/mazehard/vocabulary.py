"""MazeHard semantic vocabulary — re-exports from the task-owned surface.

Canonical SEM IDs are now owned by :mod:`ehc_sn.tasks.mazehard.runtime`.
This module re-exports them for any adapter code that references the old
location, and keeps the char-encoding helpers that are not part of the
task surface.
"""

from __future__ import annotations

from ehc_sn.tasks.mazehard.runtime import (
    EMPTY_ID,
    GOAL_ID,
    MAZE_HARD_VOCAB_SIZE,
    PATH_ID,
)
from ehc_sn.tasks.mazehard.runtime import SEM_VOCAB_SIZE as VOCAB_SIZE
from ehc_sn.tasks.mazehard.runtime import (
    START_ID,
    WALL_ID,
)

# =================================================================================================
PAD_ID: int = 0
MAZE_CHARSET: str = "# SG"
MAZE_CHAR_TO_ID: dict[str, int] = {ch: idx + 1 for idx, ch in enumerate(MAZE_CHARSET)}
MAZE_ID_TO_CHAR: dict[int, str] = {idx: ch for ch, idx in MAZE_CHAR_TO_ID.items()}


# =================================================================================================
__all__ = [
    "MAZE_CHARSET",
    "MAZE_CHAR_TO_ID",
    "MAZE_ID_TO_CHAR",
    "PAD_ID",
    "WALL_ID",
    "EMPTY_ID",
    "START_ID",
    "GOAL_ID",
    "VOCAB_SIZE",
    "PATH_ID",
    "MAZE_HARD_VOCAB_SIZE",
]
