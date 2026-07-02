"""Adapter from HPC analysis ``ProducedArtifact`` to ``HPCPlaceMetricsData``.

Provides ``load_hpc_place_figure_data()`` as the canonical path for
the ``hpc_place_metrics`` figure to consume typed analysis artifacts
instead of raw traces.

The adapter validates artifact identity and schema version, loads
persisted arrays from Zarr, derives per-cell scalar metrics from the
rate maps, and constructs the figure-data object that the existing
renderer already consumes unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ehc_sn.analysis.spatial.ratemap_stats import _DEFAULT_MIN_SI_FOR_FIELD
from ehc_sn.evaluation.contracts import (
    ArtifactKey,
    ArtifactKind,
    ProducedArtifact,
)
from ehc_sn.figures.selectors.hpc import (
    HPCPlaceMetricsData,
    HPCRateMapMosaicData,
    HPCRateMapMosaicTile,
)


# =============================================================================
def _validate_artifact(artifact: ProducedArtifact) -> None:
    """Validate artifact type and schema version."""
    expected_key = ArtifactKey(ArtifactKind.ANALYSIS, "hpc_place")
    if artifact.key != expected_key:
        raise ValueError(
            f"Expected artifact key {expected_key}, got {artifact.key}"
        )
    if artifact.schema_version != 1:
        raise ValueError(
            f"Unsupported schema version: {artifact.schema_version}"
        )


def _load_zarr_array(root: Path, *path_segments: str) -> np.ndarray:
    """Load one Zarr array from ``root / *path_segments``."""
    import zarr

    array_path = root.joinpath(*path_segments)
    if not array_path.exists():
        raise FileNotFoundError(
            f"Required analysis array not found: {array_path}"
        )
    z = zarr.open_array(str(array_path), mode="r")
    return np.asarray(z[:])


# =============================================================================
def load_hpc_place_figure_data(
    artifact: ProducedArtifact,
    root: Path,
    *,
    max_examples: int = 4,
    min_spatial_information_for_field: float = _DEFAULT_MIN_SI_FOR_FIELD,
) -> HPCPlaceMetricsData:
    """Load HPC place-metrics figure data from a produced analysis artifact.

    Reads arrays persisted by ``compute_hpc_place_analysis()``, derives
    per-cell scalar metrics from the rate maps, and constructs the
    ``HPCPlaceMetricsData`` object consumed by the figure renderer.

    Args:
        artifact: Produced artifact descriptor from the HPC analysis runner.
        root: Root directory containing the artifact.
        max_examples: Maximum number of top-example rate maps to return.
        min_spatial_information_for_field: Minimum spatial information (bits)
            for valid field-center coordinates.

    Returns:
        Figure-ready per-cell place metrics.

    Raises:
        ValueError: If the artifact is not an ``hpc_place`` analysis artifact
            or ``schema_version`` is unsupported.
        FileNotFoundError: If a referenced array is missing.
    """
    _validate_artifact(artifact)

    # Resolve artifact path relative to root.
    artifact_root = _resolve_artifact_root(artifact, root)

    # Load persisted arrays.
    rate_maps_all = _load_zarr_array(artifact_root, "rate_maps")  # (E, B, L, U)
    spatial_info_all = _load_zarr_array(
        artifact_root, "spatial_information"
    )  # (E, B, U)

    # Load metadata.
    try:
        import json

        meta_path = artifact_root / "analysis_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = dict(artifact.metadata) if artifact.metadata else {}
    except Exception:
        meta = dict(artifact.metadata) if artifact.metadata else {}

    n_env = int(meta.get("n_environments", rate_maps_all.shape[0]))
    n_bands = int(meta.get("n_bands", rate_maps_all.shape[1]))
    n_locations = int(rate_maps_all.shape[2])
    n_units = int(meta.get("n_units", rate_maps_all.shape[3]))
    band_names: list[str] = meta.get(
        "band_names", [f"freq_{i}" for i in range(n_bands)]
    )

    # Use the first environment only (matching legacy selector behavior).
    rate_map_flat = rate_maps_all[0]  # (B, L, U)
    spatial_info = spatial_info_all[0]  # (B, U)

    # Derive per-cell metrics from the rate maps.
    n_cells = n_units  # per-band
    # We flatten over bands: each (band, cell) pair becomes one "cell"
    # in the HPCPlaceMetricsData convention.
    peak_rate_list: list[float] = []
    mean_rate_list: list[float] = []
    si_list: list[float] = []
    sparsity_list: list[float] = []
    field_x_list: list[float] = []
    field_y_list: list[float] = []
    field_area_list: list[float] = []

    for b in range(n_bands):
        for u in range(n_units):
            rm = rate_map_flat[b, :, u]  # (L,)
            si = float(spatial_info[b, u])

            # Peak and mean rate from flat rate map (no occupancy available).
            if np.isfinite(rm).any():
                peak_rate_list.append(float(np.nanmax(rm)))
                mean_rate_list.append(float(np.nanmean(rm)))
            else:
                peak_rate_list.append(float("nan"))
                mean_rate_list.append(float("nan"))

            si_list.append(si)

            # Sparsity and field metrics require occupancy — set to NaN.
            sparsity_list.append(float("nan"))
            field_x_list.append(float("nan"))
            field_y_list.append(float("nan"))
            field_area_list.append(float("nan"))

    cell_indices = np.arange(len(peak_rate_list), dtype=int)

    # Select top-N examples by spatial information.
    top_rate_maps_list: list[np.ndarray] = []
    top_cell_indices_list: list[int] = []
    top_si_list: list[float] = []

    if max_examples > 0:
        candidates: list[tuple[int, float, np.ndarray]] = []
        for b in range(n_bands):
            for u in range(n_units):
                idx = b * n_units + u
                si = float(spatial_info[b, u])
                if np.isfinite(si):
                    rm = rate_map_flat[b, :, u]
                    candidates.append((idx, si, rm))

        candidates.sort(key=lambda x: x[1], reverse=True)
        for idx, si, rm in candidates[:max_examples]:
            top_cell_indices_list.append(idx)
            top_si_list.append(si)
            top_rate_maps_list.append(rm)

    return HPCPlaceMetricsData(
        cell_indices=cell_indices,
        peak_rate=np.array(peak_rate_list, dtype=float),
        mean_rate=np.array(mean_rate_list, dtype=float),
        spatial_information=np.array(si_list, dtype=float),
        sparsity=np.array(sparsity_list, dtype=float),
        field_x=np.array(field_x_list, dtype=float),
        field_y=np.array(field_y_list, dtype=float),
        field_area=np.array(field_area_list, dtype=float),
        top_rate_maps=tuple(top_rate_maps_list),
        top_cell_indices=np.array(top_cell_indices_list, dtype=int),
        top_spatial_information=np.array(top_si_list, dtype=float),
        extent=(0.0, 0.0, 1.0, 1.0),
    )


# =============================================================================
def load_hpc_rate_map_mosaic_data(
    artifact: ProducedArtifact,
    root: Path,
    *,
    max_cells: int = 64,
) -> HPCRateMapMosaicData:
    """Load HPC rate-map mosaic data from the HPC place analysis artifact.

    Loads rate maps and spatial information, ranks cells by descending
    finite spatial information, and returns a flat tile list suitable
    for the ``HPCRateMapMosaicFigure`` renderer.

    Args:
        artifact: Produced artifact descriptor from the HPC analysis runner.
        root: Root directory containing the artifact.
        max_cells: Maximum number of cells (tiles) in the mosaic.

    Returns:
        Flat tile list ordered by descending finite spatial information.

    Raises:
        ValueError: If the artifact is not an ``hpc_place`` analysis artifact.
        FileNotFoundError: If a referenced array is missing.
    """
    _validate_artifact(artifact)
    artifact_root = _resolve_artifact_root(artifact, root)

    rate_maps_all = _load_zarr_array(artifact_root, "rate_maps")  # (E, B, L, U)
    spatial_info_all = _load_zarr_array(
        artifact_root, "spatial_information"
    )  # (E, B, U)

    meta = dict(artifact.metadata) if artifact.metadata else {}
    n_bands = int(meta.get("n_bands", rate_maps_all.shape[1]))
    n_units = int(meta.get("n_units", rate_maps_all.shape[3]))

    rm_env = rate_maps_all[0]  # (B, L, U)
    si_env = spatial_info_all[0]  # (B, U)

    # Collect cells with finite SI across all bands.
    candidates: list[tuple[int, float, np.ndarray]] = []
    for b in range(n_bands):
        for u in range(n_units):
            si = float(si_env[b, u])
            if np.isfinite(si):
                rm = rm_env[b, :, u]
                candidates.append((u, si, rm))

    # Sort by descending spatial information.
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = candidates[:max_cells]

    tiles: list[HPCRateMapMosaicTile] = []
    for cell_idx, si, rm in selected:
        peak = float(np.nanmax(rm)) if np.isfinite(rm).any() else float("nan")
        tiles.append(
            HPCRateMapMosaicTile(
                cell=cell_idx,
                spatial_information=si,
                peak_rate=peak,
                rate_map=rm,
            )
        )

    return HPCRateMapMosaicData(tiles=tuple(tiles), extent=(0.0, 0.0, 1.0, 1.0))
    """Validate artifact type and schema version."""
    expected_key = ArtifactKey(ArtifactKind.ANALYSIS, "hpc_place")
    if artifact.key != expected_key:
        raise ValueError(
            f"Expected artifact key {expected_key}, " f"got {artifact.key}"
        )
    if artifact.schema_version != 1:
        raise ValueError(
            f"Unsupported schema version: {artifact.schema_version}"
        )


def _resolve_artifact_root(
    artifact: ProducedArtifact,
    root: Path,
) -> Path:
    """Resolve the artifact path relative to root, rejecting traversal."""
    art_path = Path(artifact.path)
    resolved = (root / art_path).resolve()
    root_resolved = root.resolve()
    if not str(resolved).startswith(str(root_resolved)):
        raise ValueError(f"Artifact path {artifact.path!r} escapes root {root}")
    if not resolved.exists():
        raise FileNotFoundError(f"Artifact directory not found: {resolved}")
    return resolved


# =============================================================================
__all__ = [
    "load_hpc_place_figure_data",
    "load_hpc_rate_map_mosaic_data",
]
