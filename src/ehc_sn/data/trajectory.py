"""Generic random-walk trajectory helper for grid-based maze datasets.

Shared by Arena and Dungeon dataset builders.  Pure NumPy with no task
semantics; callers supply action deltas and the stay-action id so this
module stays below the ``tasks`` layer (``spec/spec-architecture.md``, §4).
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray


def random_walk(
    mask_valid: ndarray,
    start: tuple[int, int],
    n_steps: int,
    *,
    rng: np.random.Generator,
    action_deltas: tuple[tuple[int, int], ...],
    stay_action: int,
) -> tuple[ndarray, ndarray, ndarray]:
    """Random walk over passable cells.

    Args:
        mask_valid: Boolean ``(H, W)`` passable-cell mask.
        start: Starting ``(row, col)`` position.
        n_steps: Number of steps to generate.
        rng: NumPy random generator (mutated in-place).
        action_deltas: Row/col deltas indexed by action id.  The entry at
            ``action_deltas[stay_action]`` must be ``(0, 0)``.
        stay_action: Action id representing the no-op / stay action.

    Returns:
        ``(rows, cols, prev_actions)`` each shaped ``(n_steps,)`` as
        ``int32``.  ``prev_actions[0]`` is always ``stay_action``.
    """
    rows = np.zeros(n_steps, dtype=np.int32)
    cols = np.zeros(n_steps, dtype=np.int32)
    prev_actions = np.zeros(n_steps, dtype=np.int32)

    r, c = start
    rows[0], cols[0] = r, c
    prev_actions[0] = stay_action

    H, W = mask_valid.shape
    for t in range(1, n_steps):
        valid_moves: list[tuple[int, int, int]] = []
        for action, (dr, dc) in enumerate(action_deltas):
            if action == stay_action:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask_valid[nr, nc]:
                valid_moves.append((action, nr, nc))

        if valid_moves:
            action, nr, nc = valid_moves[int(rng.integers(len(valid_moves)))]
        else:
            action, nr, nc = stay_action, r, c

        r, c = nr, nc
        rows[t] = r
        cols[t] = c
        prev_actions[t] = action

    return rows, cols, prev_actions


__all__ = ["random_walk"]
