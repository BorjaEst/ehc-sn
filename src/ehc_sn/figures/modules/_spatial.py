"""Figure-level helpers for spatial geometry policy."""

from __future__ import annotations

from collections.abc import Mapping

OPEN_FIELD_SPATIAL_GEOMETRY = "open_field"
OPEN_FIELD_RATE_SMOOTH_SIGMA = 1.0
DEFAULT_RATE_SMOOTH_SIGMA = 0.0


def world_spatial_geometry(world: object) -> str:
    """Return the declared spatial geometry for a world-like object."""
    if isinstance(world, Mapping):
        value = world.get("spatial_geometry", "unknown")
    else:
        value = getattr(world, "spatial_geometry", "unknown")
    return value if isinstance(value, str) and value else "unknown"


def spatial_rate_smooth_sigma(world: object) -> float:
    """Return the rate-map smoothing policy for the world's geometry."""
    if world_spatial_geometry(world) == OPEN_FIELD_SPATIAL_GEOMETRY:
        return OPEN_FIELD_RATE_SMOOTH_SIGMA
    return DEFAULT_RATE_SMOOTH_SIGMA
