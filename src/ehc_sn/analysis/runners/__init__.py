"""Post-evaluation CPU analysis runners.

Each runner consumes aggregate artifacts (``SpatialPopulationStatisticsPayload``
persisted as Zarr) and produces analysis artifacts (rate maps, grid scores,
spatial information, etc.) that figure renderers consume.

All runners are pure CPU functions — no PyTorch, no model loading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# =============================================================================
def load_aggregate_artifact(path: Path) -> dict[str, np.ndarray]:
    """Load a Zarr aggregate directory into a flat dict of NumPy arrays.

    Iterates over immediate subdirectories under *path* (each subdirectory is
    one Zarr array group) and loads each into a NumPy array.

    Args:
        path: Path to a Zarr aggregate directory (e.g.
            ``aggregates/mec_spatial_population.zarr``).

    Returns:
        Dict mapping array name → loaded NumPy array.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If *path* contains no Zarr array subdirectories.
    """
    import zarr  # lazy import

    if not path.exists():
        raise FileNotFoundError(f"Aggregate artifact not found: {path}")

    result: dict[str, np.ndarray] = {}
    for child in sorted(path.iterdir()):
        if child.is_dir() and not child.name.startswith("_"):
            try:
                z = zarr.open_array(str(child), mode="r")
                result[child.name] = np.asarray(z[:])
            except Exception:
                pass  # skip non-Zarr subdirectories

    if not result:
        raise ValueError(f"No Zarr arrays found in aggregate artifact {path}")

    return result


# =============================================================================
def load_aggregate_metadata(path: Path) -> dict[str, object]:
    """Load Zarr ``.zattrs`` metadata from each array in an aggregate artifact.

    Returns a dict keyed by array name whose values are the parsed metadata
    dicts (or empty dict if none).
    """
    import zarr  # lazy import

    result: dict[str, object] = {}
    for child in sorted(path.iterdir()):
        if child.is_dir() and not child.name.startswith("_"):
            try:
                z = zarr.open_array(str(child), mode="r")
                result[child.name] = dict(z.attrs)
            except Exception:
                result[child.name] = {}
    return result


from ehc_sn.analysis.runners.hpc import compute_hpc_place_analysis  # noqa: E402

# =============================================================================
from ehc_sn.analysis.runners.mec import compute_mec_grid_analysis  # noqa: E402

__all__ = [
    "compute_mec_grid_analysis",
    "compute_hpc_place_analysis",
    "load_aggregate_artifact",
    "load_aggregate_metadata",
]
