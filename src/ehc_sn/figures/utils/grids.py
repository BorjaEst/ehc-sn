"""MazeHard figure utilities."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import numpy as np


def reshape_grid(seq: np.ndarray, *, grid_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Reshape a flat sequence into a 2D grid.

    If ``grid_shape`` is not provided, infer a square grid and validate it.
    """
    arr = np.asarray(seq)
    if grid_shape is None:
        n = int(math.isqrt(arr.size))
        if n * n != arr.size:
            raise ValueError(f"Expected square sequence length, got {arr.size}")
        grid_shape = (n, n)
    if grid_shape[0] * grid_shape[1] != arr.size:
        raise ValueError(
            f"Expected sequence length {grid_shape[0] * grid_shape[1]}, got {arr.size}"
        )
    return arr.reshape(grid_shape)


def first_halt_index(halted_t: Iterable[bool]) -> int:
    """Return the first index where halted is True, or the last index if none."""
    halted_arr = np.asarray(list(halted_t), dtype=bool)
    if halted_arr.size == 0:
        return 0
    if halted_arr.any():
        return int(np.argmax(halted_arr))
    return int(halted_arr.size - 1)
