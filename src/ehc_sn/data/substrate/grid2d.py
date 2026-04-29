"""Grid2D shared-substrate channel constants and sample validator.

Owns channel names, expected dtypes, mandatory channel set, and the per-sample
validator for 2-D grid substrates (dungeongen, maze-nd).

This module is the canonical home for grid2d-specific contracts.  Generic
topology-kind identifiers live in :mod:`ehc_sn.data.schema`.  Generic
build mechanics live in :mod:`ehc_sn.data.build`.

See ``spec/spec-data-contracts.md`` §6 for the full channel classification.
"""

from __future__ import annotations

import numpy as np

# =================================================================================================
TOPOLOGY_KIND: str = "grid2d"
"""Canonical topology kind string for 2-D grid substrates."""

# Shared spatial channel name constants
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

MANDATORY_CHANNELS: frozenset[str] = frozenset({CHANNEL_TOPOLOGY})
"""Channels that must be present in every grid2d shared substrate sample."""

CHANNEL_DTYPES: dict[str, np.dtype] = {
    CHANNEL_TOPOLOGY: np.dtype(bool),
    CHANNEL_OBSERVATIONS: np.dtype(np.int32),
    CHANNEL_MASK_VALID: np.dtype(bool),
    CHANNEL_REGIONS: np.dtype(np.int32),
    CHANNEL_LANDMARKS: np.dtype(np.int32),
}
"""Expected numpy dtype per grid2d channel."""

_SPATIAL_CHANNELS: frozenset[str] = frozenset(CHANNEL_DTYPES.keys())


# =================================================================================================
def validate_grid2d_sample(data: dict[str, np.ndarray],) -> None:  # fmt: skip  # -------------------------------------------------------------------
    """Validate that *data* conforms to the grid2d shared-substrate sample contract.

    Accepts both single-sample arrays (H, W) and stacked arrays (N, H, W).

    Checks:
    - Mandatory channel ``topology`` is present.
    - All known channels have the expected dtype.
    - All 2-D spatial channels share the same ``(H, W)`` trailing shape.

    Args:
        data: Dict of channel name → numpy array.

    Raises:
        ValueError: On any contract violation.
    """
    missing = MANDATORY_CHANNELS - data.keys()
    if missing:
        raise ValueError(f"grid2d sample missing mandatory channels: {sorted(missing)}")

    shapes: dict[str, tuple[int, ...]] = {}
    for name, arr in data.items():
        if name in CHANNEL_DTYPES and arr.dtype != CHANNEL_DTYPES[name]:
            raise ValueError(f"Channel '{name}' has dtype {arr.dtype}, expected {CHANNEL_DTYPES[name]}.")
        if name in _SPATIAL_CHANNELS:
            if arr.ndim not in (2, 3):
                raise ValueError(f"Channel '{name}' has invalid rank {arr.ndim}; expected 2 (H, W) or 3 (N, H, W).")
            shapes[name] = arr.shape[-2:]

    unique = set(shapes.values())
    if len(unique) > 1:
        detail = ", ".join(f"'{k}': {v}" for k, v in shapes.items())
        raise ValueError(f"All grid2d spatial channels must share the same (H, W) shape. Got: {detail}.")


# =================================================================================================
__all__ = [
    "TOPOLOGY_KIND",
    "CHANNEL_TOPOLOGY",
    "CHANNEL_OBSERVATIONS",
    "CHANNEL_MASK_VALID",
    "CHANNEL_REGIONS",
    "CHANNEL_LANDMARKS",
    "MANDATORY_CHANNELS",
    "CHANNEL_DTYPES",
    "validate_grid2d_sample",
]
