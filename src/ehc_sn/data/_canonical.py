"""Deterministic canonicalization helpers for processed maze channels.

These helpers keep source-specific scripts small while centralizing the rules
for valid-mask selection, singleton-cell choice, structural landmarks, and
observation assignment.
"""

from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

# =================================================================================================
_NEIGHBORS4: tuple[tuple[int, int], ...] = ((-1, 0), (0, -1), (0, 1), (1, 0))


# =================================================================================================
def largest_component_mask(topology: np.ndarray) -> np.ndarray:
    """Return the largest 4-connected component of ``topology``.

    Ties are broken by the lexicographically smallest cell so the result is
    deterministic across runs.
    """
    if topology.ndim != 2:
        raise ValueError(f"topology must be a 2-D array, got shape {topology.shape}.")

    topology = topology.astype(bool, copy=False)
    visited = np.zeros_like(topology, dtype=bool)
    best_mask: np.ndarray | None = None
    best_size = -1
    best_anchor: tuple[int, int] | None = None

    for start_row, start_col in np.argwhere(topology):
        if visited[start_row, start_col]:
            continue

        queue: deque[tuple[int, int]] = deque([(int(start_row), int(start_col))])
        cells: list[tuple[int, int]] = []
        visited[start_row, start_col] = True

        while queue:
            row, col = queue.popleft()
            cells.append((row, col))
            for d_row, d_col in _NEIGHBORS4:
                nxt_row, nxt_col = row + d_row, col + d_col
                if nxt_row < 0 or nxt_row >= topology.shape[0] or nxt_col < 0 or nxt_col >= topology.shape[1]:
                    continue
                if visited[nxt_row, nxt_col] or not topology[nxt_row, nxt_col]:
                    continue
                visited[nxt_row, nxt_col] = True
                queue.append((nxt_row, nxt_col))

        anchor = min(cells)
        size = len(cells)
        if size > best_size or (size == best_size and (best_anchor is None or anchor < best_anchor)):
            best_size = size
            best_anchor = anchor
            best_mask = np.zeros_like(topology, dtype=bool)
            for row, col in cells:
                best_mask[row, col] = True

    return np.zeros_like(topology, dtype=bool) if best_mask is None else best_mask


# =================================================================================================
def first_true_cell(mask: np.ndarray) -> tuple[int, int] | None:
    """Return the lexicographically first true cell in ``mask``."""
    coords = np.argwhere(mask.astype(bool, copy=False))
    if coords.size == 0:
        return None
    row, col = coords[0]
    return int(row), int(col)


# =================================================================================================
def singleton_mask(shape: tuple[int, int], cell: tuple[int, int]) -> np.ndarray:
    """Return a boolean mask with exactly one true cell."""
    mask = np.zeros(shape, dtype=bool)
    mask[cell] = True
    return mask


# =================================================================================================
def shortest_path_distances(mask_valid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Return 4-neighbor shortest-path distances over ``mask_valid``.

    Unreachable cells are marked with ``-1``.
    """
    if not mask_valid[start]:
        raise ValueError(f"start cell {start} must lie inside the valid mask.")

    distances = np.full(mask_valid.shape, -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque([start])
    distances[start] = 0

    while queue:
        row, col = queue.popleft()
        for d_row, d_col in _NEIGHBORS4:
            nxt_row, nxt_col = row + d_row, col + d_col
            if nxt_row < 0 or nxt_row >= mask_valid.shape[0] or nxt_col < 0 or nxt_col >= mask_valid.shape[1]:
                continue
            if not mask_valid[nxt_row, nxt_col] or distances[nxt_row, nxt_col] >= 0:
                continue
            distances[nxt_row, nxt_col] = distances[row, col] + 1
            queue.append((nxt_row, nxt_col))

    return distances


# =================================================================================================
def farthest_reachable_cell(mask_valid: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    """Return the farthest reachable cell from ``start``.

    Ties are broken lexicographically.
    """
    distances = shortest_path_distances(mask_valid, start)
    max_distance = int(distances.max())
    if max_distance < 0:
        raise ValueError("valid mask contains no reachable cells.")

    candidates = np.argwhere(distances == max_distance)
    row, col = candidates[0]
    return int(row), int(col)


# =================================================================================================
def canonical_cell_from_mask(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    reference: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Select one canonical true cell from ``mask`` constrained by ``valid_mask``.

    If ``reference`` is provided, the farthest constrained cell from that
    reference is chosen. Otherwise the lexicographically first constrained cell
    is returned.
    """
    constrained = mask.astype(bool, copy=False) & valid_mask.astype(bool, copy=False)
    if not np.any(constrained):
        return None
    if reference is None:
        return first_true_cell(constrained)

    distances = shortest_path_distances(valid_mask.astype(bool, copy=False), reference)
    coords = np.argwhere(constrained & (distances >= 0))
    if coords.size == 0:
        return None

    best_cell: tuple[int, int] | None = None
    best_distance = -1
    for row, col in coords:
        cell = (int(row), int(col))
        distance = int(distances[cell])
        if distance > best_distance or (
            distance == best_distance and (best_cell is None or cell < best_cell)
        ):
            best_distance = distance
            best_cell = cell
    return best_cell


# =================================================================================================
def binary_structural_landmarks(mask_valid: np.ndarray) -> np.ndarray:
    """Return binary junction landmarks encoded as ``int32`` values in ``{0, 1}``."""
    mask_valid = mask_valid.astype(bool, copy=False)
    degrees = np.zeros(mask_valid.shape, dtype=np.int32)
    for d_row, d_col in _NEIGHBORS4:
        src_rows = slice(max(0, -d_row), mask_valid.shape[0] - max(0, d_row))
        src_cols = slice(max(0, -d_col), mask_valid.shape[1] - max(0, d_col))
        dst_rows = slice(max(0, d_row), mask_valid.shape[0] - max(0, -d_row))
        dst_cols = slice(max(0, d_col), mask_valid.shape[1] - max(0, -d_col))
        degrees[dst_rows, dst_cols] += (
            mask_valid[dst_rows, dst_cols] & mask_valid[src_rows, src_cols]
        ).astype(np.int32)
    return np.where(mask_valid & (degrees >= 3), 1, 0).astype(np.int32)


# =================================================================================================
def sample_observations(mask_valid: np.ndarray, n_observations: int, *, seed: int) -> np.ndarray:
    """Assign deterministic random observation ids to valid cells.

    Invalid cells are filled with ``-1``.
    """
    if n_observations <= 0:
        raise ValueError(f"n_observations must be positive, got {n_observations}.")

    rng = np.random.default_rng(seed)
    observations = np.full(mask_valid.shape, -1, dtype=np.int32)
    valid_coords = np.argwhere(mask_valid.astype(bool, copy=False))
    if valid_coords.size == 0:
        return observations

    obs_ids = rng.integers(0, n_observations, size=len(valid_coords), dtype=np.int32)
    for (row, col), obs_id in zip(valid_coords, obs_ids, strict=True):
        observations[int(row), int(col)] = int(obs_id)
    return observations


# =================================================================================================
def stable_text_seed(text: str) -> int:
    """Return a stable 32-bit seed derived from ``text``."""
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


# =================================================================================================
__all__ = [
    "binary_structural_landmarks",
    "canonical_cell_from_mask",
    "farthest_reachable_cell",
    "first_true_cell",
    "largest_component_mask",
    "sample_observations",
    "shortest_path_distances",
    "singleton_mask",
    "stable_text_seed",
]
