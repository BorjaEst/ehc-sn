"""Shared replay-build helpers for grid-based task corpus builders.

Provides the random-walk trajectory generator and start-cell selector used
by Arena and Dungeon task builders.  These helpers are pure NumPy with no
task semantics; callers supply action deltas and the stay-action id.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray


def first_true_cell(mask: np.ndarray) -> tuple[int, int] | None:
    """Return the lexicographically first true cell in *mask*."""
    coords = np.argwhere(mask.astype(bool, copy=False))
    if coords.size == 0:
        return None
    row, col = coords[0]
    return int(row), int(col)


def random_valid_cell(
    mask_valid: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample a start cell uniformly from valid cells in row-major order.

    Implements the ``random_valid_cell_v1`` start policy: enumerate valid
    cells in row-major order, then sample uniformly using ``rng``.

    Args:
        mask_valid: Boolean ``(H, W)`` passable-cell mask.
        rng: NumPy random generator (mutated in-place).

    Returns:
        ``(row, col)`` of the sampled start cell.

    Raises:
        RuntimeError: When ``mask_valid`` has no passable cells.
    """
    coords = np.argwhere(mask_valid.astype(bool, copy=False))
    if coords.size == 0:
        raise RuntimeError("Empty valid mask — no passable cells to start from.")
    idx = int(rng.integers(len(coords)))
    return int(coords[idx, 0]), int(coords[idx, 1])


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
        action_deltas: Row/col deltas indexed by action id.
        stay_action: Action id representing the no-op / stay action.

    Returns:
        ``(rows, cols, prev_actions)`` each shaped ``(n_steps,)`` as ``int32``.
        ``prev_actions[0]`` is always ``stay_action``.
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


# ---------------------------------------------------------------------------
# Frozen inverse-action map for no-backtrack walk (Arena v1)
# ---------------------------------------------------------------------------
_INVERSE_ACTION: dict[int, int] = {1: 3, 3: 1, 2: 4, 4: 2}  # UP↔DOWN, RIGHT↔LEFT
"""Inverse movement action map (STAY=0 has no entry; it does not trigger suppression)."""


def random_walk_no_backtrack(
    mask_valid: ndarray,
    start: tuple[int, int],
    n_steps: int,
    *,
    rng: np.random.Generator,
    action_deltas: tuple[tuple[int, int], ...],
    stay_action: int,
) -> tuple[ndarray, ndarray, ndarray]:
    """Random walk with no-immediate-backtrack suppression.

    Implements ``random_walk_no_immediate_backtrack_v1``:

    - Enumerate legal movement actions from the current cell.
    - If the inverse of the previous non-STAY action is legal **and** at least
      one other legal movement exists, exclude that inverse action.
    - If no legal movement remains, emit STAY.
    - Otherwise sample uniformly from the remaining legal actions.

    Inverse map (frozen): UP↔DOWN, RIGHT↔LEFT. STAY has no inverse-suppression
    effect.

    Args:
        mask_valid: Boolean ``(H, W)`` passable-cell mask.
        start: Starting ``(row, col)`` position.
        n_steps: Number of steps to generate.
        rng: NumPy random generator (mutated in-place).
        action_deltas: Row/col deltas indexed by action id.
        stay_action: Action id representing the no-op / stay action.

    Returns:
        ``(rows, cols, prev_actions)`` each shaped ``(n_steps,)`` as ``int32``.
        ``prev_actions[0]`` is always ``stay_action``.
    """
    rows = np.zeros(n_steps, dtype=np.int32)
    cols = np.zeros(n_steps, dtype=np.int32)
    prev_actions = np.zeros(n_steps, dtype=np.int32)

    r, c = start
    rows[0], cols[0] = r, c
    prev_actions[0] = stay_action

    H, W = mask_valid.shape
    last_non_stay = stay_action  # tracks the most recent non-STAY action taken

    for t in range(1, n_steps):
        legal: list[tuple[int, int, int]] = []
        for action, (dr, dc) in enumerate(action_deltas):
            if action == stay_action:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask_valid[nr, nc]:
                legal.append((action, nr, nc))

        # Suppress immediate backtrack when there are alternatives.
        inverse = _INVERSE_ACTION.get(last_non_stay)
        if inverse is not None and len(legal) > 1:
            legal = [(a, nr, nc) for (a, nr, nc) in legal if a != inverse]

        if not legal:
            action, nr, nc = stay_action, r, c
        else:
            action, nr, nc = legal[int(rng.integers(len(legal)))]

        if action != stay_action:
            last_non_stay = action

        r, c = nr, nc
        rows[t] = r
        cols[t] = c
        prev_actions[t] = action

    return rows, cols, prev_actions


__all__ = [
    "first_true_cell",
    "random_valid_cell",
    "random_walk",
    "random_walk_no_backtrack",
]
