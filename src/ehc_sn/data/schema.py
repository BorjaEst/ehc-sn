"""Canonical on-disk format contracts for shared-substrate maze datasets.

Each processed dataset is stored as a directory of named per-channel ``.npy``
files with spatial shape ``(H, W)`` (or ``(N, H, W)`` for stacked splits).
This module defines:

- Channel name constants for **shared-substrate** channels (``CHANNEL_*``).
- Expected numpy dtypes per channel (``CHANNEL_DTYPES``).
- The mandatory channel set (``MANDATORY_CHANNELS``).
- A ``validate_processed`` function that enforces the shared contract.

Trajectory and replay channel constants belong to the owning task packages
(``ehc_sn.tasks.dungeon`` and ``ehc_sn.tasks.arena``), not here.

See ``spec/spec-data-contracts.md`` §6 for the full channel classification.
"""

from __future__ import annotations

import numpy as np

# =================================================================================================
# Shared spatial channel name constants
# =================================================================================================

CHANNEL_TOPOLOGY: str = "topology"
"""Passable cells (``True``) vs walls (``False``). Mandatory."""

CHANNEL_OBSERVATIONS: str = "observations"
"""Unique observation ID per passable cell. Optional shared channel."""

CHANNEL_MASK_VALID: str = "mask_valid"
"""Explicit reachability mask (largest passable component). Optional shared channel."""

CHANNEL_REGIONS: str = "regions"
"""Room/region ID. Optional shared channel."""

CHANNEL_LANDMARKS: str = "landmarks"
"""Structural landmark IDs. Optional shared channel."""

# =================================================================================================
MANDATORY_CHANNELS: frozenset[str] = frozenset({CHANNEL_TOPOLOGY})

CHANNEL_DTYPES: dict[str, np.dtype] = {
    CHANNEL_TOPOLOGY: np.dtype(bool),
    CHANNEL_OBSERVATIONS: np.dtype(np.int32),
    CHANNEL_MASK_VALID: np.dtype(bool),
    CHANNEL_REGIONS: np.dtype(np.int32),
    CHANNEL_LANDMARKS: np.dtype(np.int32),
}

_SPATIAL_CHANNELS: frozenset[str] = frozenset(CHANNEL_DTYPES.keys())
"""All channels whose trailing two dimensions must be the same ``(H, W)`` grid."""


# =================================================================================================
def validate_processed(  # ------------------------------------------------------------------------
    data: dict[str, np.ndarray],
) -> None:  # fmt: skip
    """Validate that *data* conforms to the canonical processed dataset contract.

    Accepts both single-sample arrays and stacked arrays (leading batch dim).
    Validates only channels listed in ``CHANNEL_DTYPES`` (shared spatial
    channels). Trajectory and replay channels are validated by task-owned
    validators.

    Checks:
    - Mandatory channel ``topology`` is present.
    - All known channels have the expected dtype.
    - All 2-D spatial channels share the same ``(H, W)`` shape.

    Args:
        data: Dict of channel name → numpy array.

    Raises:
        ValueError: On any contract violation.
    """
    missing = MANDATORY_CHANNELS - data.keys()
    if missing:
        raise ValueError(f"Missing mandatory channels: {sorted(missing)}")

    shapes: dict[str, tuple[int, ...]] = {}
    for name, arr in data.items():
        if name in CHANNEL_DTYPES and arr.dtype != CHANNEL_DTYPES[name]:
            raise ValueError(f"Channel '{name}' has dtype {arr.dtype}, expected {CHANNEL_DTYPES[name]}.")
        if name in _SPATIAL_CHANNELS:
            if arr.ndim not in (2, 3):
                raise ValueError(
                    f"Channel '{name}' has invalid rank {arr.ndim}; expected 2 (H, W) or 3 (N, H, W)."
                )
            shapes[name] = arr.shape[-2:]

    unique = set(shapes.values())
    if len(unique) > 1:
        detail = ", ".join(f"'{k}': {v}" for k, v in shapes.items())
        raise ValueError(f"All spatial channels must share the same (H, W) shape. Got: {detail}.")


# =================================================================================================
__all__ = [
    "CHANNEL_TOPOLOGY",
    "CHANNEL_OBSERVATIONS",
    "CHANNEL_MASK_VALID",
    "CHANNEL_REGIONS",
    "CHANNEL_LANDMARKS",
    "MANDATORY_CHANNELS",
    "CHANNEL_DTYPES",
    "validate_processed",
]
