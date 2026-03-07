"""Canonical on-disk format contracts for maze NPZ files.

Each processed maze is stored as a single NPZ file containing a dict of named
2D channels with shape ``(H, W)``. This module defines:

- Channel name constants (``CHANNEL_*``).
- Expected numpy dtypes per channel (``CHANNEL_DTYPES``).
- The mandatory channel set (``MANDATORY_CHANNELS``).
- A ``validate_npz`` function that enforces the contract at load time.

See ``spec/spec-architecture.md`` §3.6.2 for the full format specification.
"""

from __future__ import annotations

import numpy as np

# =================================================================================================
CHANNEL_TOPOLOGY: str = "topology"
"""Passable cells (``True``) vs walls (``False``). Mandatory."""

CHANNEL_OBSERVATIONS: str = "observations"
"""Unique observation ID per passable cell. Optional."""

CHANNEL_START: str = "start"
"""Agent start position(s). Optional."""

CHANNEL_GOALS: str = "goals"
"""Target goal position(s). Optional."""

CHANNEL_SOLUTION: str = "solution"
"""Shortest-path distance or step label (0 = not on path). Optional."""

CHANNEL_LANDMARKS: str = "landmarks"
"""Special object IDs ('shiny'). Optional."""

CHANNEL_REGIONS: str = "regions"
"""Room/region ID. Optional."""

CHANNEL_MASK_VALID: str = "mask_valid"
"""Explicit reachability mask. Optional."""

# =================================================================================================
# Vocabulary token IDs
# =================================================================================================

O_ID: int = 5
"""Solution-path overlay token ID used in MazeHard supervised training."""

# =================================================================================================
MANDATORY_CHANNELS: frozenset[str] = frozenset({CHANNEL_TOPOLOGY})

CHANNEL_DTYPES: dict[str, np.dtype] = {
    CHANNEL_TOPOLOGY: np.dtype(bool),
    CHANNEL_OBSERVATIONS: np.dtype(np.int32),
    CHANNEL_START: np.dtype(bool),
    CHANNEL_GOALS: np.dtype(bool),
    CHANNEL_SOLUTION: np.dtype(np.int32),
    CHANNEL_LANDMARKS: np.dtype(np.int32),
    CHANNEL_REGIONS: np.dtype(np.int32),
    CHANNEL_MASK_VALID: np.dtype(bool),
}


# =================================================================================================
def validate_npz(  # ------------------------------------------------------------------------------
    data: dict[str, np.ndarray],
) -> None:  # fmt: skip
    """Validate that *data* conforms to the canonical channel contract.

    Accepts both single-maze arrays ``(H, W)`` and stacked arrays ``(N, H, W)``.

    Args:
        data: Dict of channel name → numpy array.

    Raises:
        ValueError: If any mandatory channel is missing, any known channel has
            the wrong dtype, or channels have inconsistent spatial shapes.
    """
    missing = MANDATORY_CHANNELS - data.keys()
    if missing:
        raise ValueError(f"Missing mandatory NPZ channels: {sorted(missing)}")

    shapes: dict[str, tuple[int, ...]] = {}
    for name, arr in data.items():
        if name in CHANNEL_DTYPES and arr.dtype != CHANNEL_DTYPES[name]:
            raise ValueError(f"Channel '{name}' has dtype {arr.dtype}, expected {CHANNEL_DTYPES[name]}.")
        if arr.ndim not in (2, 3):
            raise ValueError(f"Channel '{name}' must be 2D (H, W) or 3D (N, H, W), got shape {arr.shape}.")
        # Compare only spatial dimensions (last two).
        shapes[name] = arr.shape[-2:]

    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        detail = ", ".join(f"'{k}': {v}" for k, v in shapes.items())
        raise ValueError(f"All channels must have the same (H, W) shape. Got: {detail}.")


# =================================================================================================
__all__ = [
    "CHANNEL_TOPOLOGY",
    "CHANNEL_OBSERVATIONS",
    "CHANNEL_START",
    "CHANNEL_GOALS",
    "CHANNEL_SOLUTION",
    "CHANNEL_LANDMARKS",
    "CHANNEL_REGIONS",
    "CHANNEL_MASK_VALID",
    "O_ID",
    "MANDATORY_CHANNELS",
    "CHANNEL_DTYPES",
    "validate_npz",
]
