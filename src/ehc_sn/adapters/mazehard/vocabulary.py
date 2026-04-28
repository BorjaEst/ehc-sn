"""Canonical maze semantic (SEM) vocabulary for MazeHard adapters.

Defines the canonical integer IDs used to encode maze grid cells as a
model-agnostic semantic enum.  This vocabulary is owned by the MazeHard
adapter side — it expresses task-level semantics, not storage contracts.

Canonical SEM layout:
    0  PAD   — padding / ignored position
    1  WALL  — impassable cell ("``#``")
    2  EMPTY — passable cell ("`` ``")
    3  START — agent start position ("``S``")
    4  GOAL  — target goal position ("``G``")

Note:
    ``O_ID`` (solution-path cell token) is **not** part of the canonical
    vocabulary.  It is an adapter-owned supervision annotation defined in
    :mod:`ehc_sn.adapters.mazehard.hrm.core`.
"""

from __future__ import annotations

# =================================================================================================
MAZE_CHARSET: str = "# SG"
MAZE_CHAR_TO_ID: dict[str, int] = {ch: idx + 1 for idx, ch in enumerate(MAZE_CHARSET)}
MAZE_ID_TO_CHAR: dict[int, str] = {idx: ch for ch, idx in MAZE_CHAR_TO_ID.items()}
VOCAB_SIZE: int = len(MAZE_CHARSET) + 1  # +1 for PAD at index 0  (= 5)

# =================================================================================================
PAD_ID: int = 0
WALL_ID: int = MAZE_CHAR_TO_ID["#"]
EMPTY_ID: int = MAZE_CHAR_TO_ID[" "]
START_ID: int = MAZE_CHAR_TO_ID["S"]
GOAL_ID: int = MAZE_CHAR_TO_ID["G"]


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
]
