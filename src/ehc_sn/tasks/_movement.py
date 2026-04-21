""" """

from __future__ import annotations

from enum import IntEnum
from typing import Final

ACTION_STAY: Final[int] = 0
ACTION_UP: Final[int] = 1
ACTION_RIGHT: Final[int] = 2
ACTION_DOWN: Final[int] = 3
ACTION_LEFT: Final[int] = 4
MOVEMENT_ACTION_COUNT: Final[int] = 5

# Row/column deltas indexed by action id: stay, up, right, down, left.
_ACTION_DELTAS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),  # STAY
    (-1, 0),  # UP
    (0, 1),  # RIGHT
    (1, 0),  # DOWN
    (0, -1),  # LEFT
)


class MovementAction(IntEnum):
    """Canonical discrete movement action for dungeon-grid tasks."""

    STAY = ACTION_STAY
    UP = ACTION_UP
    RIGHT = ACTION_RIGHT
    DOWN = ACTION_DOWN
    LEFT = ACTION_LEFT


__all__ = [
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_STAY",
    "ACTION_UP",
    "MOVEMENT_ACTION_COUNT",
    "MovementAction",
]
