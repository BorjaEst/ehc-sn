"""Selectors: extract and prepare trace data for figure templates."""

from ehc_sn.figures.selectors.spatial import (
    DEFAULT_RATE_MAP_MIN_BIN_OCCUPANCY,
    OPEN_FIELD_RATE_SMOOTH_SIGMA,
    OPEN_FIELD_SPATIAL_GEOMETRY,
    compute_cell_location_responses,
    compute_location_responses,
    prepare_rate_map,
    prepare_rate_map_from_location_responses,
    prepare_rate_maps,
    rectify_response,
    spatial_rate_smooth_sigma,
    world_spatial_geometry,
)

__all__ = [
    "DEFAULT_RATE_MAP_MIN_BIN_OCCUPANCY",
    "OPEN_FIELD_RATE_SMOOTH_SIGMA",
    "OPEN_FIELD_SPATIAL_GEOMETRY",
    "compute_cell_location_responses",
    "compute_location_responses",
    "prepare_rate_map",
    "prepare_rate_map_from_location_responses",
    "prepare_rate_maps",
    "rectify_response",
    "spatial_rate_smooth_sigma",
    "world_spatial_geometry",
]
