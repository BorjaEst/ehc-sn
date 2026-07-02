"""Adapter from MEC analysis ``ProducedArtifact`` to ``MECGridMetricsData``.

Provides ``load_mec_grid_figure_data()`` as the canonical path for
the ``mec_grid_metrics`` figure to consume typed analysis artifacts
instead of raw traces or dict-based artifact data.

The adapter validates artifact identity and schema version, loads
persisted arrays from Zarr, constructs the ``MECGridMetricsData``
object that the existing renderer already consumes unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ehc_sn.evaluation.contracts import (
    ArtifactKey,
    ArtifactKind,
    ProducedArtifact,
)
from ehc_sn.figures.selectors.mec import (
    MECAutocorrMosaicData,
    MECGridMetricsData,
)


# =============================================================================
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
def load_mec_grid_figure_data(
    artifact: ProducedArtifact,
    root: Path,
    *,
    max_examples: int = 4,
) -> MECGridMetricsData:
    """Load MEC grid metrics figure data from a produced analysis artifact.

    Reads arrays persisted by ``compute_mec_grid_analysis()`` and
    constructs the ``MECGridMetricsData`` object consumed by the
    ``mec_grid_metrics`` figure renderer.

    Args:
        artifact: Produced artifact descriptor from the MEC analysis runner.
        root: Root directory containing the artifact.
        max_examples: Maximum number of top-example autocorrelograms.

    Returns:
        Figure-ready per-frequency, per-cell grid metrics.

    Raises:
        ValueError: If the artifact is not a ``mec_grid`` analysis artifact
            or ``schema_version`` is unsupported.
        FileNotFoundError: If a referenced array is missing.
    """
    _validate_artifact(artifact)

    # Resolve artifact path relative to root.
    artifact_root = _resolve_artifact_root(artifact, root)

    # Load persisted arrays.
    grid_scores_all = _load_zarr_array(
        artifact_root, "grid_scores"
    )  # (E, B, U)
    spacing_all = _load_zarr_array(artifact_root, "spacing")  # (E, B, U)
    orientation_all = _load_zarr_array(
        artifact_root, "orientation"
    )  # (E, B, U)
    autocorr_all = _load_zarr_array(
        artifact_root, "autocorrelation"
    )  # (E, B, U, H, W)

    # Load metadata.
    meta = dict(artifact.metadata) if artifact.metadata else {}
    n_env = int(meta.get("n_environments", grid_scores_all.shape[0]))
    n_bands = int(meta.get("n_bands", grid_scores_all.shape[1]))
    n_units = int(meta.get("n_units", grid_scores_all.shape[2]))
    band_names: list[str] = meta.get(
        "band_names", [f"freq_{i}" for i in range(n_bands)]
    )

    # Validate band count against loaded arrays.
    if grid_scores_all.shape[1] != n_bands:
        raise ValueError(
            f"Metadata n_bands={n_bands} does not match grid_scores "
            f"dim 1 = {grid_scores_all.shape[1]}"
        )

    # Use the first environment only (matching legacy selector behaviour).
    env_i = 0
    gs = np.asarray(grid_scores_all[env_i], dtype=np.float64)  # (B, U)
    sp = np.asarray(spacing_all[env_i], dtype=np.float64)  # (B, U)
    ori = np.asarray(orientation_all[env_i], dtype=np.float64)  # (B, U)

    n_cells = n_units
    freq_indices = np.arange(n_bands, dtype=int)
    cell_indices = np.arange(n_cells, dtype=int)

    # Ensure NaN is preserved (already float64, nan stays nan).
    valid_pixel_count = np.full(
        (n_bands, n_cells), autocorr_all.shape[-2], dtype=np.int64
    )

    # Top-N examples by grid score across all bands.
    top_autocorrs_list: list[np.ndarray] = []
    top_freq_list: list[int] = []
    top_cell_list: list[int] = []
    top_gs_list: list[float] = []

    if max_examples > 0:
        candidates: list[tuple[float, int, int, np.ndarray]] = []
        for b in range(n_bands):
            for u in range(n_cells):
                score = float(gs[b, u])
                if np.isfinite(score):
                    ac = autocorr_all[env_i, b, u]
                    candidates.append((score, b, u, ac))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for score, b, u, ac in candidates[:max_examples]:
            top_gs_list.append(score)
            top_freq_list.append(b)
            top_cell_list.append(u)
            top_autocorrs_list.append(ac)

    return MECGridMetricsData(
        freq_indices=freq_indices,
        cell_indices=cell_indices,
        gridness=gs,
        spacing=sp,
        orientation_deg=ori,
        valid_pixel_count=valid_pixel_count,
        inner_radius=0.0,
        outer_radius=0.0,
        mask_strategy="auto",
        top_autocorrs=tuple(top_autocorrs_list),
        top_freq_indices=np.array(top_freq_list, dtype=int),
        top_cell_indices=np.array(top_cell_list, dtype=int),
        top_gridness=np.array(top_gs_list, dtype=float),
    )


# =============================================================================
def load_mec_autocorr_mosaic_data(
    artifact: ProducedArtifact,
    root: Path,
    *,
    max_cells_per_freq: int = 36,
) -> MECAutocorrMosaicData:
    """Load MEC autocorrelation mosaic data from the MEC grid analysis artifact.

    Loads autocorrelograms and grid scores, groups by frequency band,
    ranks by descending finite grid scores within each band, and returns
    the data structure consumed by the ``MECAutocorrMosaicFigure`` renderer.

    Args:
        artifact: Produced artifact descriptor from the MEC analysis runner.
        root: Root directory containing the artifact.
        max_cells_per_freq: Maximum cells per frequency band.

    Returns:
        Per-frequency autocorrelograms and grid scores.

    Raises:
        ValueError: If the artifact is not a ``mec_grid`` analysis artifact.
        FileNotFoundError: If a referenced array is missing.
    """
    _validate_artifact(artifact)
    artifact_root = _resolve_artifact_root(artifact, root)

    autocorr_all = _load_zarr_array(
        artifact_root, "autocorrelation"
    )  # (E, B, U, H, W)
    grid_scores_all = _load_zarr_array(
        artifact_root, "grid_scores"
    )  # (E, B, U)

    meta = dict(artifact.metadata) if artifact.metadata else {}
    n_bands = int(meta.get("n_bands", grid_scores_all.shape[1]))
    band_names: list[str] = meta.get(
        "band_names", [f"freq_{i}" for i in range(n_bands)]
    )

    env_i = 0
    ac_env = autocorr_all[env_i]  # (B, U, H, W)
    gs_env = grid_scores_all[env_i]  # (B, U)

    freq_indices = np.arange(n_bands, dtype=int)
    gridness_by_freq: list[np.ndarray] = []
    autocorrs_by_freq: list[tuple[np.ndarray, ...]] = []
    cell_indices_by_freq: list[np.ndarray] = []

    for b in range(n_bands):
        # Collect cells with finite grid score, sorted descending.
        scored: list[tuple[float, int, np.ndarray]] = []
        for u in range(gs_env.shape[1]):
            gs = float(gs_env[b, u])
            if np.isfinite(gs):
                ac = ac_env[b, u]
                scored.append((gs, u, ac))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_cells_per_freq]

        if not top:
            gridness_by_freq.append(np.empty(0, dtype=float))
            autocorrs_by_freq.append(())
            cell_indices_by_freq.append(np.empty(0, dtype=int))
        else:
            gridness_by_freq.append(np.array([s[0] for s in top], dtype=float))
            autocorrs_by_freq.append(tuple(s[2] for s in top))
            cell_indices_by_freq.append(
                np.array([s[1] for s in top], dtype=int)
            )

    return MECAutocorrMosaicData(
        freq_indices=freq_indices,
        gridness_by_freq=gridness_by_freq,
        autocorrs_by_freq=autocorrs_by_freq,
        cell_indices_by_freq=cell_indices_by_freq,
        extent=(0.0, 0.0, 1.0, 1.0),
    )


# =============================================================================
def _validate_artifact(artifact: ProducedArtifact) -> None:
    """Validate artifact type and schema version."""
    expected_key = ArtifactKey(ArtifactKind.ANALYSIS, "mec_grid")
    if artifact.key != expected_key:
        raise ValueError(
            f"Expected artifact key {expected_key}, got {artifact.key}"
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
    "load_mec_grid_figure_data",
    "load_mec_autocorr_mosaic_data",
]
