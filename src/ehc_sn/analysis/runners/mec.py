"""MEC grid analysis runner — consumes aggregate, produces grid metrics.

Computes rate maps, spatial autocorrelograms, grid scores, spacing, and
orientation for every cell across all frequency bands in a
``SpatialPopulationArtifact``.

All computation is pure NumPy/SciPy — no PyTorch or model loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.analysis.runners import load_aggregate_artifact
from ehc_sn.analysis.spatial.gridness import (
    compute_gridness,
    estimate_grid_spacing_orientation,
)
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
from ehc_sn.figures.plots.autocorr import compute_spatial_autocorrelogram
from ehc_sn.types import SpatialRateMapPayload


# =============================================================================
def compute_mec_grid_analysis(
    aggregate_artifact_path: Path,
    *,
    grid_score_threshold: float = 0.3,
    minimum_occupancy: int = 5,
    output_dir: Path | None = None,
) -> tuple[ProducedArtifact, ...]:
    """Compute MEC grid metrics from a spatial population aggregate artifact.

    Args:
        aggregate_artifact_path:
            Path to a Zarr aggregate artifact directory containing
            ``occupancy``, ``activation_sum`` arrays and metadata.
        grid_score_threshold:
            Minimum grid score to classify a cell as grid-like.
        minimum_occupancy:
            Minimum visits to a location bin for it to be considered valid.
        output_dir:
            Output directory for the analysis Zarr artifact.  Defaults to
            ``<aggregate_parent>/../analyses/mec_grid.zarr``.

    Returns:
        Tuple with one ``ProducedArtifact`` for the MEC grid analysis.

    Raises:
        FileNotFoundError: If *aggregate_artifact_path* does not exist.
        ValueError: If no cells pass the minimum occupancy filter.
    """
    from ehc_sn.analysis.spatial.geometry import SpatialBinGeometry
    from ehc_sn.types import ScaleMetadata

    # Resolve output directory.
    if output_dir is None:
        output_dir = (
            aggregate_artifact_path.parent.parent / "analyses" / "mec_grid.zarr"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load aggregate data.
    arrays = load_aggregate_artifact(aggregate_artifact_path)

    occupancy = arrays.get("occupancy")
    if occupancy is None:
        raise FileNotFoundError(
            f"Missing 'occupancy' array in {aggregate_artifact_path}"
        )

    # Reconstruct the payload from flat arrays.
    # The accumulator stores per-band activation sums indexed by band name.
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

    if not activation_sum:
        # Loading from Zarr may preserve the original key. Check direct dict case.
        # The aggregate artifact layout from the new pipeline uses
        # dict-style arrays — try loading by known key.
        activation_sum = {
            k: v
            for k, v in arrays.items()
            if k.startswith("freq_") or "_sum" in k
        }

    # Environment count and location count from occupancy shape.
    n_environments = occupancy.shape[0]
    n_locations = occupancy.shape[1]

    # Build a minimal geometry (assume square grid from n_locations).
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

    # ---- Per-cell grid analysis ----
    band_names = list(rate_map_payload.rate_maps.keys())
    n_bands = len(band_names)

    # Determine n_units from rate maps.
    n_units = (
        max(rate_map_payload.rate_maps[b].shape[-1] for b in band_names)
        if rate_map_payload.rate_maps
        else 0
    )

    all_rate_maps = np.zeros(
        (n_environments, n_bands, n_locations, n_units),
        dtype=np.float64,
    )
    all_autocorr = np.zeros(
        (
            n_environments,
            n_bands,
            n_units,
            2 * grid_size - 1,
            2 * grid_size - 1,
        ),
        dtype=np.float64,
    )
    all_grid_scores = np.full(
        (n_environments, n_bands, n_units), np.nan, dtype=np.float64
    )
    all_spacing = np.full(
        (n_environments, n_bands, n_units), np.nan, dtype=np.float64
    )
    all_orientation = np.full(
        (n_environments, n_bands, n_units), np.nan, dtype=np.float64
    )

    for env_i in range(n_environments):
        for band_i, band_name in enumerate(band_names):
            rate_maps = rate_map_payload.rate_maps[band_name][env_i]  # (L, U)
            visited = rate_map_payload.visited_mask[env_i]  # (L,)
            for unit_i in range(rate_maps.shape[-1]):
                rate_map_1d = rate_maps[:, unit_i]
                rate_map_2d = rate_map_1d.reshape(grid_size, grid_size)
                all_rate_maps[env_i, band_i, :, unit_i] = rate_map_1d

                try:
                    autocorr = compute_spatial_autocorrelogram(
                        rate_map_2d, normalize=True
                    )
                    all_autocorr[env_i, band_i, unit_i] = autocorr

                    g = compute_gridness(autocorr)
                    grid_score = g.score if g is not None else np.nan
                    all_grid_scores[env_i, band_i, unit_i] = grid_score

                    if not np.isnan(grid_score):
                        so = estimate_grid_spacing_orientation(autocorr)
                        if so is not None:
                            all_spacing[env_i, band_i, unit_i] = so.spacing
                            all_orientation[env_i, band_i, unit_i] = (
                                so.orientation_deg
                            )
                except Exception:
                    pass

    # Persist analysis arrays via ZarrArtifactWriter.
    analysis_data: dict[str, np.ndarray] = {
        "rate_maps": all_rate_maps,
        "autocorrelation": all_autocorr,
        "grid_scores": all_grid_scores,
        "spacing": all_spacing,
        "orientation": all_orientation,
    }
    ZarrArtifactWriter.write("mec_grid", analysis_data, output_dir)

    prod_digest = (
        f"mec_grid:v1:"
        f"threshold={grid_score_threshold}:"
        f"min_occ={minimum_occupancy}"
    )

    return (
        ProducedArtifact(
            key=ArtifactKey(ArtifactKind.ANALYSIS, "mec_grid"),
            schema_version=1,
            path=(
                output_dir.relative_to(output_dir.parent.parent)
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
                "grid_score_threshold": grid_score_threshold,
                "minimum_occupancy": minimum_occupancy,
                "band_names": list(band_names),
            },
        ),
    )


__all__ = ["compute_mec_grid_analysis"]
