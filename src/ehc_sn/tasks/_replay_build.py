"""Shared replay-build helpers for grid-based task corpus builders.

Provides the random-walk trajectory generator and start-cell selector used
by Arena and Dungeon task builders.  These helpers are pure NumPy with no
task semantics; callers supply action deltas and the stay-action id.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy import ndarray


def _cartesian_angle(
    s1_row: int,
    s1_col: int,
    s2_row: int,
    s2_col: int,
    /,
) -> float:
    """Compute the allocentric angle from (s1) to (s2) in Cartesian coordinates.

    Returns angle in radians in ``[-pi, pi]`` where 0 = right, pi/2 = up.
    """
    return float(np.arctan2(s1_row - s2_row, s2_col - s1_col))


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
        raise RuntimeError(
            "Empty valid mask — no passable cells to start from."
        )
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
# Inverse-action map helper (derived from action_deltas, not hardcoded)
# ---------------------------------------------------------------------------
def _build_inverse_action_map(
    action_deltas: tuple[tuple[int, int], ...],
) -> dict[int, int]:
    """Return ``{action_id: inverse_action_id}`` from negatable deltas.

    An action's inverse is the action whose (dr, dc) delta is the exact
    negation of this action's delta.  STAY (delta ``(0, 0)``) is never
    added to the map.  An action whose delta has no matching negation
    does not appear as a key (no spurious suppression).

    Args:
        action_deltas: Tuple of ``(dr, dc)`` per action id.

    Returns:
        Dict mapping action id to its inverse action id.
    """
    inv: dict[int, int] = {}
    n = len(action_deltas)
    for a in range(n):
        dr, dc = action_deltas[a]
        if dr == 0 and dc == 0:
            continue  # STAY has no inverse
        for b in range(n):
            if b == a:
                continue
            dr2, dc2 = action_deltas[b]
            if dr2 == -dr and dc2 == -dc:
                inv[a] = b
                break
    return inv


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

    The inverse action for each non-STAY action is derived from
    ``action_deltas``: action ``b`` is the inverse of action ``a`` when
    ``delta_b == -delta_a``.  This generalises to any action space
    (4-direction grid, 6-direction hex, etc.) without hardcoded mapping.

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
    inverse_map = _build_inverse_action_map(action_deltas)

    for t in range(1, n_steps):
        legal: list[tuple[int, int, int]] = []
        for action, (dr, dc) in enumerate(action_deltas):
            if action == stay_action:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask_valid[nr, nc]:
                legal.append((action, nr, nc))

        # Suppress immediate backtrack when there are alternatives.
        inverse = inverse_map.get(last_non_stay)
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


# ---------------------------------------------------------------------------
# Legacy straight-line angle-bias walk (faithful TEM reproduction)
# ---------------------------------------------------------------------------


def random_walk_straight_bias(
    mask_valid: ndarray,
    start: tuple[int, int],
    n_steps: int,
    *,
    rng: np.random.Generator,
    action_deltas: tuple[tuple[int, int], ...],
    stay_action: int,
    direc_bias: float = 0.2,
    angle_bias_change: float = 0.4,
    diff_angle_min: float | None = None,
    angle_fn: Callable = _cartesian_angle,
) -> tuple[ndarray, ndarray, ndarray]:
    """Walk with a straight-line heading bias (legacy TEM angle-bias policy).

    Implements the angle-bias algorithm from the legacy TEM:

    - Maintain a persistent heading angle theta, initialised uniformly in
      ``[-pi, pi]``.
    - At each step, compute the allocentric angle from the current cell to
      each available neighbour.
    - Pick the neighbour whose angle is closest to theta, as long as the
      angular difference is below ``diff_angle_min``.  If the closest
      neighbour exceeds the threshold (i.e. the agent hit a wall), pick a
      random non-stay neighbour.
    - Perturb theta by ``uniform(-angle_bias_change, +angle_bias_change)``
      **every step** (after both biased and random moves).
    - With probability ``direc_bias``, override the biased neighbour and
      sample uniformly from the available neighbours instead (no bias).

    Args:
        mask_valid: Boolean ``(H, W)`` passable-cell mask.
        start: Starting ``(row, col)`` position.
        n_steps: Number of steps to generate.
        rng: NumPy random generator (mutated in-place).
        action_deltas: Row/col deltas indexed by action id.
        stay_action: Action id representing the no-op / stay action.
        direc_bias: Probability of a random override (default 0.2 means
            20% of steps are random, 80% follow the heading bias).
        angle_bias_change: Maximum angular perturbation per step (default
            0.4 radians).
        diff_angle_min: Angular difference threshold for considering the
            bias neighbour.  Defaults to ``pi/4`` for square/cartesian grids.
        angle_fn: Function ``(r1, c1, r2, c2) -> float`` that computes the
            allocentric angle from (r1, c1) to (r2, c2).  Default is
            :func:`_cartesian_angle`; override for hex grids.

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
    current_angle = rng.uniform(-np.pi, np.pi)
    if diff_angle_min is None:
        diff_angle_min = np.pi / 4  # default for square/cartesian

    for t in range(1, n_steps):
        valid_moves: list[tuple[int, int, int]] = []
        for action, (dr, dc) in enumerate(action_deltas):
            if action == stay_action:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask_valid[nr, nc]:
                valid_moves.append((action, nr, nc))

        if not valid_moves:
            action, nr, nc = stay_action, r, c
        elif rng.random() > direc_bias:
            # Follow heading bias: pick neighbour closest to current angle.
            angles = [
                angle_fn(r, c, nr, nc) if (nr, nc) != (r, c) else 10000.0
                for _, nr, nc in valid_moves
            ]
            a_diffs = [abs(a - current_angle) for a in angles]
            a_diffs = [a if a < np.pi else abs(2 * np.pi - a) for a in a_diffs]
            angle_diff = min(a_diffs)

            if angle_diff < diff_angle_min:
                min_idx = a_diffs.index(angle_diff)
                action, nr, nc = valid_moves[min_idx]
            else:
                # Hit a wall — pick random non-stay neighbour.
                p_angles = [1.0 if a < 100 else 1e-6 for a in angles]
                p_sum = sum(p_angles)
                p_norm = [p / p_sum for p in p_angles]
                idx = rng.choice(len(valid_moves), p=p_norm)
                action, nr, nc = valid_moves[idx]

        else:
            # Random move (override the bias).
            action, nr, nc = valid_moves[int(rng.integers(len(valid_moves)))]

        # Perturb heading every step (after biased or random move), matching
        # legacy TEM behavior where the heading always drifts.
        current_angle += rng.uniform(-angle_bias_change, angle_bias_change)
        current_angle = np.mod(current_angle + np.pi, 2 * np.pi) - np.pi

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
    "random_walk_straight_bias",
]
