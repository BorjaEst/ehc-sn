"""HPC place analysis runner — consumes aggregate, produces place metrics.

Computes rate maps, spatial information, field segmentation, and coverage
for every cell across all frequency bands in a
``SpatialPopulationArtifact``.

All computation is pure NumPy/SciPy — no PyTorch or model loading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ehc_sn.analysis.runners import load_aggregate_artifact
from ehc_sn.analysis.spatial.ratemap_stats import compute_rate_map_stats
from ehc_sn.analysis.spatial.ratemaps import (
    RateMapConfig,
    SpatialPopulationStatisticsPayload,
    compute_rate_maps,
)
from ehc_sn.evaluation.artifacts import ZarrArtifactWriter
from ehc_sn.evaluation.contracts import (
    ArtifactKey,
    ArtifactKind,
    ProducedArtifact,
)
from ehc_sn.types import ScaleMetadata, SpatialRateMapPayload


# =============================================================================
class AnalysisExecutionError(RuntimeError):
    """Raised when an analysis runner produces no valid results."""


# =============================================================================
def compute_hpc_place_analysis(
    aggregate_artifact_path: Path,
    *,
    min_peak_distance: int = 3,
    minimum_occupancy: int = 5,
    output_dir: Path | None = None,
) -> tuple[ProducedArtifact, ...]:
    """Compute HPC place metrics from a spatial population aggregate artifact.

    Args:
        aggregate_artifact_path:
            Path to a Zarr aggregate artifact directory.
        min_peak_distance:
            Minimum distance between field peaks (in bins).
        minimum_occupancy:
            Minimum visits to a location bin for it to be considered valid.
        output_dir:
            Output directory for the analysis Zarr artifact.  Defaults to
            ``<aggregate_parent>/../analyses/hpc_place.zarr``.

    Returns:
        Tuple with one ``ProducedArtifact`` for the HPC place analysis.
    """
    from ehc_sn.analysis.spatial.geometry import SpatialBinGeometry

    # Resolve output directory.
    if output_dir is None:
        output_dir = (
            aggregate_artifact_path.parent.parent
            / "analyses"
            / "hpc_place.zarr"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load aggregate data.
    arrays = load_aggregate_artifact(aggregate_artifact_path)

    occupancy = arrays.get("occupancy")
    if occupancy is None:
        raise FileNotFoundError(
            f"Missing 'occupancy' array in {aggregate_artifact_path}"
        )

    # Reconstruct payload.
    activation_sum: dict[str, np.ndarray] = {}
    scale_metadata: dict[str, ScaleMetadata] = {}
    for key, arr in arrays.items():
        if key.startswith("activation_sum_"):
            band_name = key[len("activation_sum_") :]
            activation_sum[band_name] = arr
        elif key.startswith("scale_metadata_"):
            band_name = key[len("scale_metadata_") :]
            scale_metadata[band_name] = ScaleMetadata(
                band_name=band_name,
                feature_dim=arr.shape[-1] if arr.ndim > 0 else 0,
                index=len(scale_metadata),
            )

    n_environments = occupancy.shape[0]
    n_locations = occupancy.shape[1]
    grid_size = int(np.sqrt(n_locations))
    geometry = SpatialBinGeometry(bin_size_x=1.0, bin_size_y=1.0)

    environment_ids = tuple(
        arrays.get(
            "environment_ids", [f"env_{i}" for i in range(n_environments)]
        )
    )

    payload = SpatialPopulationStatisticsPayload(
        occupancy=occupancy,
        activation_sum=activation_sum,
        activation_sq_sum=None,
        geometry=geometry,
        environment_ids=environment_ids,
        scale_metadata=scale_metadata
        or {
            "freq_0": ScaleMetadata(band_name="freq_0", feature_dim=0, index=0),
        },
        valid_step_count=int(arrays.get("valid_step_count", np.array([0]))[0]),
        accumulation_dtype=str(
            arrays.get("accumulation_dtype", [b"float32"])[0]
        ),
    )

    # Compute rate maps.
    config = RateMapConfig(minimum_occupancy=minimum_occupancy)
    rate_map_payload: SpatialRateMapPayload = compute_rate_maps(payload, config)

    # ---- Per-cell place analysis ----
    band_names = list(rate_map_payload.rate_maps.keys())
    n_bands = len(band_names)

    n_units = (
        max(rate_map_payload.rate_maps[b].shape[-1] for b in band_names)
        if rate_map_payload.rate_maps
        else 0
    )

    all_rate_maps = np.zeros(
        (n_environments, n_bands, n_locations, n_units),
        dtype=np.float64,
    )
    all_spatial_info = np.full(
        (n_environments, n_bands, n_units), np.nan, dtype=np.float64
    )
    all_field_masks = np.zeros(
        (n_environments, n_bands, n_locations, n_units),
        dtype=bool,
    )
    all_coverage = np.full(
        (n_environments, n_bands, n_units), np.nan, dtype=np.float64
    )

    # Derive world extent from geometry once (same for all environments).
    extent = (
        0.0,
        grid_size * geometry.bin_size_x,
        0.0,
        grid_size * geometry.bin_size_y,
    )

    total_analyzed = 0
    total_valid = 0

    for env_i in range(n_environments):
        occ_2d = (
            rate_map_payload.occupancy[env_i]
            .reshape(grid_size, grid_size)
            .astype(np.float64)
        )
        for band_i, band_name in enumerate(band_names):
            rate_maps = rate_map_payload.rate_maps[band_name][env_i]
            for unit_i in range(rate_maps.shape[-1]):
                rate_map_1d = rate_maps[:, unit_i]
                rate_map_2d = rate_map_1d.reshape(grid_size, grid_size)
                all_rate_maps[env_i, band_i, :, unit_i] = rate_map_1d
                total_analyzed += 1

                stats = compute_rate_map_stats(
                    rate_map_2d,
                    occupancy=occ_2d,
                    extent=extent,
                    geometry=geometry,
                )
                all_spatial_info[env_i, band_i, unit_i] = (
                    stats.spatial_information
                )
                all_field_masks[env_i, band_i, :, unit_i] = (
                    stats.field_mask.flatten()
                )
                all_coverage[env_i, band_i, unit_i] = stats.coverage
                if np.isfinite(stats.spatial_information):
                    total_valid += 1

    # Guard: at least one valid unit across all environments and bands.
    if total_analyzed > 0 and total_valid == 0:
        raise AnalysisExecutionError(
            "HPC place analysis produced zero valid unit metrics; "
            f"checked {total_analyzed} units across "
            f"{n_environments} env(s) × {n_bands} band(s). "
            "Verify occupancy, geometry, and rate-map inputs are finite "
            "and contain visited bins."
        )

    # Persist analysis arrays.
    analysis_data: dict[str, np.ndarray] = {
        "rate_maps": all_rate_maps,
        "spatial_information": all_spatial_info,
        "field_masks": all_field_masks,
        "coverage": all_coverage,
    }
    ZarrArtifactWriter.write("hpc_place", analysis_data, output_dir)

    prod_digest = (
        f"hpc_place:v1:"
        f"min_peak_dist={min_peak_distance}:"
        f"min_occ={minimum_occupancy}"
    )

    return (
        ProducedArtifact(
            key=ArtifactKey(ArtifactKind.ANALYSIS, "hpc_place"),
            schema_version=1,
            path=(
                (output_dir / "hpc_place").relative_to(output_dir.parent.parent)
                if output_dir.parent.parent
                else output_dir
            ),
            media_type="application/vnd+zarr",
            producer_digest=prod_digest,
            content_digest=prod_digest,
            metadata={
                "n_environments": n_environments,
                "n_bands": n_bands,
                "n_units": n_units,
                "min_peak_distance": min_peak_distance,
                "minimum_occupancy": minimum_occupancy,
                "band_names": list(band_names),
            },
        ),
    )


__all__ = ["compute_hpc_place_analysis"]
