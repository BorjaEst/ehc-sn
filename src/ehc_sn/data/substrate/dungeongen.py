"""Shared-substrate builder for the dungeongen dataset family.

Orchestrates the shared-substrate pipeline for the dungeongen source family:

1. :func:`ensure_raw` — create/validate canonical tar-sharded raw snapshot.
2. :func:`prepare_interim` — normalize raw topologies into per-split NPZ files.
3. :func:`build_shared_substrate` — build the versioned immutable substrate root.

Public surface: SHARED_FAMILY, SHARED_CHANNELS, ensure_raw, prepare_interim,
build_shared_substrate.

Shared-substrate channels (topology, observations, mask_valid, regions,
landmarks) are task-neutral.  Trajectory and replay channels belong in task
corpora; see ``ehc_sn.tasks.dungeon`` and ``ehc_sn.tasks.arena``.

Interim layer: ``data/interim/dungeongen/`` — one NPZ file per split.
Shared substrate: ``data/processed/dungeongen/v<version>/``
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterator

import numpy as np

from ehc_sn.data.layout import (
    DEFAULT_GRID_ACTION_SPACE,
    SpatialLayout,
    validate_spatial_layout,
    write_layout_dataset,
)
from ehc_sn.data.lifecycle import (
    extract_version,
    staging_root,
    write_index_at_root,
    write_split,
)
from ehc_sn.data.manifest import write_manifest
from ehc_sn.data.substrate._common import (
    binary_structural_landmarks,
    largest_component_mask,
    sample_observations,
)
from ehc_sn.data.substrate._dungeongen_raw import (
    ensure_raw_snapshot,
    iter_raw_topologies,
)
from ehc_sn.data.substrate.grid2d import TOPOLOGY_KIND as _GRID2D_KIND
from ehc_sn.data.substrate.grid2d import validate_grid2d_sample

# ---------------------------------------------------------------------------
SHARED_FAMILY: Final[str] = "dungeongen"
"""Shared-substrate family name for the dungeongen source."""

SHARED_CHANNELS: Final[list[str]] = [
    "topology",
    "observations",
    "mask_valid",
    "regions",
    "landmarks",
]
"""Shared-substrate channels (task-neutral layout data)."""

_SPLITS: tuple[str, ...] = ("train", "val", "test")
_SPLIT_SEED_OFFSET: dict[str, int] = {
    "train": 0,
    "val": 100_000,
    "test": 200_000,
}
_SOURCE_ID: Final[str] = "dungeongen"


# ---------------------------------------------------------------------------
def ensure_raw(
    raw_root: Path,
    base_seed: int,
    split_counts: dict[str, int],
) -> None:
    """Create or validate the canonical tar-sharded raw snapshot.

    Args:
        raw_root: Canonical raw root (e.g. ``data/raw/dungeongen``).
        base_seed: Root base seed used across splits.
        split_counts: Mapping of split name to number of samples.
    """
    ensure_raw_snapshot(raw_root, base_seed, split_counts)


def prepare_interim(
    raw_root: Path,
    interim_root: Path,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
) -> None:
    """Normalize the raw snapshot into per-split NPZ files in the interim layer.

    Reads topology data from the canonical raw snapshot and writes one NPZ
    file per split to ``data/interim/dungeongen/<split>.npz``.  Each split
    NPZ stores deterministically normalized arrays padded to the split's max
    shape.

    Per-split NPZ arrays:

    - ``sample_id``: (N,) int64
    - ``seed``:      (N,) int64
    - ``height``:    (N,) int32 — native generator height per sample
    - ``width``:     (N,) int32 — native generator width per sample
    - ``topology``:  (N, H_max, W_max) bool, padded with ``False``
    - ``regions``:   (N, H_max, W_max) int32, padded with ``-1``

    Args:
        raw_root: Canonical raw root (e.g. ``data/raw/dungeongen``).
        interim_root: Interim destination (e.g. ``data/interim/dungeongen``).
        n_train: Number of training samples to read.
        n_val: Number of validation samples to read.
        n_test: Number of test samples to read.

    Raises:
        RuntimeError: When a split has fewer raw samples than expected.
    """
    interim_root.mkdir(parents=True, exist_ok=True)
    split_counts = {"train": n_train, "val": n_val, "test": n_test}

    for split, n in split_counts.items():
        sample_ids: list[int] = []
        seeds: list[int] = []
        heights: list[int] = []
        widths: list[int] = []
        topologies: list[np.ndarray] = []
        regions_list: list[np.ndarray] = []

        for idx, (topology, regions, seed) in enumerate(
            iter_raw_topologies(raw_root, split)
        ):
            if idx >= n:
                break
            sample_ids.append(idx)
            seeds.append(seed)
            heights.append(topology.shape[0])
            widths.append(topology.shape[1])
            topologies.append(topology)
            regions_list.append(regions)

        count = len(sample_ids)
        if count < n:
            raise RuntimeError(
                f"Interim: only {count} raw topologies found for '{split}', need {n}."
            )

        h_max = int(max(h for h in heights))
        w_max = int(max(w for w in widths))
        topo_arr = np.zeros((count, h_max, w_max), dtype=bool)
        reg_arr = np.full((count, h_max, w_max), -1, dtype=np.int32)
        for i, (topo, reg, h, w) in enumerate(
            zip(topologies, regions_list, heights, widths)
        ):
            topo_arr[i, :h, :w] = topo
            reg_arr[i, :h, :w] = reg

        out_path = interim_root / f"{split}.npz"
        np.savez_compressed(
            out_path,
            sample_id=np.array(sample_ids, dtype=np.int64),
            seed=np.array(seeds, dtype=np.int64),
            height=np.array(heights, dtype=np.int32),
            width=np.array(widths, dtype=np.int32),
            topology=topo_arr,
            regions=reg_arr,
        )


def _iter_interim_topologies(
    interim_root: Path, split: str
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield ``(topology, regions, seed)`` from the dungeongen interim split file.

    Args:
        interim_root: Interim root (e.g. ``data/interim/dungeongen``).
        split: Split name.

    Yields:
        ``(topology, regions, seed)`` tuples in deterministic index order.

    Raises:
        FileNotFoundError: When the interim split NPZ does not exist.
    """
    split_path = interim_root / f"{split}.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Interim split file not found: {split_path}.  Run prepare-interim first."
        )
    data = np.load(split_path)
    heights = data["height"]
    widths = data["width"]
    seeds = data["seed"]
    topologies = data["topology"]
    regions_arr = data["regions"]
    for i in range(len(heights)):
        h, w = int(heights[i]), int(widths[i])
        yield topologies[i, :h, :w], regions_arr[i, :h, :w], int(seeds[i])


# ---------------------------------------------------------------------------
def _pad_to_shape(
    arr: np.ndarray, target_h: int, target_w: int, *, fill: int | bool
) -> np.ndarray:
    h, w = arr.shape
    if h > target_h or w > target_w:
        raise ValueError(
            f"Source shape ({h}, {w}) exceeds target ({target_h}, {target_w}). "
            "Increase --height/--width."
        )
    if h == target_h and w == target_w:
        return arr
    out = np.full((target_h, target_w), fill, dtype=arr.dtype)
    out[:h, :w] = arr
    return out


def _build_substrate_sample(
    topology: np.ndarray,
    dungeongen_regions: np.ndarray,
    *,
    target_h: int,
    target_w: int,
    s_size: int,
    topology_seed: int,
) -> dict[str, np.ndarray]:
    """Normalize a raw dungeongen topology into shared-substrate channels."""
    rng = np.random.default_rng(topology_seed)

    topology = _pad_to_shape(topology, target_h, target_w, fill=False)
    regions = _pad_to_shape(
        dungeongen_regions, target_h, target_w, fill=-1
    ).astype(np.int32)

    mask_valid = largest_component_mask(topology)
    observations = sample_observations(
        mask_valid, s_size, seed=int(rng.integers(2**31))
    )
    landmarks = binary_structural_landmarks(mask_valid)

    return {
        "topology": topology,
        "observations": observations,
        "mask_valid": mask_valid,
        "regions": regions,
        "landmarks": landmarks,
    }


def _sample_seed(base_seed: int, split: str, idx: int) -> int:
    return base_seed + _SPLIT_SEED_OFFSET[split] + idx


def _infer_shape(
    interim_root: Path,
    split_counts: dict[str, int],
) -> tuple[int, int]:
    """Return (max_height, max_width) across all selected interim samples."""
    max_h = 0
    max_w = 0
    for split, n in split_counts.items():
        for idx, (topology, _, _) in enumerate(
            _iter_interim_topologies(interim_root, split)
        ):
            if idx >= n:
                break
            h, w = topology.shape
            if h > max_h:
                max_h = h
            if w > max_w:
                max_w = w
    return max_h, max_w


# ---------------------------------------------------------------------------
def build_shared_substrate(
    version_root: Path,
    *,
    interim_root: Path,
    n_train: int = 1000,
    n_val: int = 40,
    n_test: int = 40,
    height: int | None = None,
    width: int | None = None,
    n_observations: int = 45,
    seed: int = 42,
) -> None:
    """Build the dungeongen shared substrate at *version_root*.

    Reads topology files from *interim_root* (produced by
    :func:`prepare_interim`), pads each topology, and assigns shared-substrate
    channels (no trajectory data).

    When *height* and/or *width* are omitted (``None``), the required
    processed grid dimensions are inferred from the selected interim slice.

    The version integer is derived from the ``v<N>`` leaf of *version_root*;
    there is no separate ``version`` parameter.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/dungeongen/v1``).  Must not already exist.
        interim_root: Interim leaf (e.g. ``data/interim/dungeongen``).
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        height: Target grid height after padding.  When ``None``, inferred.
        width: Target grid width after padding.  When ``None``, inferred.
        n_observations: Number of distinct observation ids to assign.
        seed: Base RNG seed.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        ValueError: When explicit height/width is smaller than the inferred max.
        RuntimeError: When augmentation encounters an empty valid mask.
    """
    version = extract_version(version_root)
    split_counts = {"train": n_train, "val": n_val, "test": n_test}

    inferred_h, inferred_w = _infer_shape(interim_root, split_counts)
    if height is None:
        resolved_h = inferred_h
    else:
        if height < inferred_h:
            raise ValueError(
                f"Explicit height={height} is smaller than the required maximum shape "
                f"({inferred_h}, {inferred_w}) from the selected interim slice. "
                "Increase --height or omit it to use the inferred value."
            )
        resolved_h = height
    if width is None:
        resolved_w = inferred_w
    else:
        if width < inferred_w:
            raise ValueError(
                f"Explicit width={width} is smaller than the required maximum shape "
                f"({inferred_h}, {inferred_w}) from the selected interim slice. "
                "Increase --width or omit it to use the inferred value."
            )
        resolved_w = width

    shape: tuple[int, int] = (resolved_h, resolved_w)
    n_states = resolved_h * resolved_w
    stage_params = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "height": resolved_h,
        "width": resolved_w,
        "n_observations": n_observations,
        "seed": seed,
    }

    with staging_root(version_root) as tmp:
        all_entries = []
        for split in _SPLITS:
            n = split_counts[split]
            interim_topologies = list(
                _iter_interim_topologies(interim_root, split)
            )[:n]
            samples = [
                _build_substrate_sample(
                    topology,
                    dungeon_regions,
                    target_h=resolved_h,
                    target_w=resolved_w,
                    n_observations=n_observations,
                    seed=_sample_seed(seed, split, idx),
                )
                for idx, (topology, dungeon_regions, _) in enumerate(
                    interim_topologies
                )
            ]
            entries = write_split(
                tmp,
                split,
                samples,
                source=SHARED_FAMILY,
                channels=SHARED_CHANNELS,
                topology_kind=_GRID2D_KIND,
                n_states=n_states,
                extent=[resolved_h, resolved_w],
                index_kwargs={},
                sample_validator=validate_grid2d_sample,
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="shared_substrate",
            family=SHARED_FAMILY,
            version=version,
            channels=SHARED_CHANNELS,
            topology_kind=_GRID2D_KIND,
            n_states=n_states,
            extent=[resolved_h, resolved_w],
            n_samples=split_counts,
            source_id=_SOURCE_ID,
            builder="ehc_sn.data.substrate.dungeongen.build_shared_substrate",
            seed=seed,
            stage_params=stage_params,
        )

    n_total = n_train + n_val + n_test
    print(
        f"dungeongen shared substrate written to {version_root}  ({n_total} samples.)"
    )


def validate_dungeongen_shared_root(root: Path) -> dict:
    """Validate a dungeongen shared substrate root against generic and family-owned rules.

    Calls the generic structural validator first, then checks dungeongen-specific
    family semantics (topology_kind, required channels).

    Args:
        root: Resolved versioned dungeongen shared substrate root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any contract violation.
        FileNotFoundError: When a required file is absent.
    """
    from ehc_sn.data.lifecycle import validate_version_root

    manifest = validate_version_root(root)
    if manifest.get("family") != SHARED_FAMILY:
        raise ValueError(
            f"Root family is {manifest.get('family')!r}, expected {SHARED_FAMILY!r}."
        )
    if manifest.get("topology_kind") != _GRID2D_KIND:
        raise ValueError(
            f"Root topology_kind is {manifest.get('topology_kind')!r}, expected {_GRID2D_KIND!r}."
        )
    missing_ch = set(SHARED_CHANNELS) - set(manifest.get("channels", []))
    if missing_ch:
        raise ValueError(
            f"Manifest missing required dungeongen channels: {sorted(missing_ch)}"
        )
    return manifest


# =============================================================================
# Layout dataset builder (converts shared-substrate records to SpatialLayout)
# =============================================================================


def build_dungeongen_layouts(
    version_root: Path,
    *,
    interim_root: Path,
    n_train: int = 1000,
    n_val: int = 40,
    n_test: int = 40,
    height: int | None = None,
    width: int | None = None,
    s_size: int = 45,
    topology_seed: int = 42,
    n_sensory_instances: int = 1,
    preset: str = "default",
) -> None:
    """Build a dungeongen layout dataset at *version_root*.

    Reads topology files from *interim_root*, pads each topology, assigns
    shared-substrate channels, and converts each sample into one or more
    :class:`SpatialLayout` records written via :func:`write_layout_dataset`.

    This is the dungeongen-specific producer that feeds into the common
    layout dataset contract consumed by ``build-arena.py``.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/interim/dungeongen/v1``).  Must not already exist.
        interim_root: Interim leaf (e.g. ``data/interim/dungeongen``).
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        height: Target grid height after padding.  When ``None``, inferred.
        width: Target grid width after padding.  When ``None``, inferred.
        s_size: Number of distinct observation ids to assign (sensory vocab size).
        topology_seed: Base RNG seed for topology and sensory assignment.
        n_sensory_instances: Number of sensory realizations per topology sample.
            Must be >= 1.

    Raises:
        ValueError: When explicit height/width is smaller than the inferred max, or
            ``n_sensory_instances < 1``.
    """
    split_counts = {"train": n_train, "val": n_val, "test": n_test}
    inferred_h, inferred_w = _infer_shape(interim_root, split_counts)

    if height is None:
        resolved_h = inferred_h
    else:
        if height < inferred_h:
            raise ValueError(
                f"Explicit height={height} is smaller than the required maximum shape "
                f"({inferred_h}, {inferred_w}) from the selected interim slice."
            )
        resolved_h = height
    if width is None:
        resolved_w = inferred_w
    else:
        if width < inferred_w:
            raise ValueError(
                f"Explicit width={width} is smaller than the required maximum shape "
                f"({inferred_h}, {inferred_w}) from the selected interim slice."
            )
        resolved_w = width

    if n_sensory_instances < 1:
        raise ValueError(
            f"n_sensory_instances must be >= 1, got {n_sensory_instances}."
        )

    layouts: list[SpatialLayout] = []
    topology_type = "grid2d"

    for split in _SPLITS:
        n = split_counts[split]
        interim_topologies = list(
            _iter_interim_topologies(interim_root, split)
        )[:n]
        for idx, (topology, dungeon_regions, _) in enumerate(
            interim_topologies
        ):
            sample = _build_substrate_sample(
                topology,
                dungeon_regions,
                target_h=resolved_h,
                target_w=resolved_w,
                s_size=s_size,
                topology_seed=_sample_seed(topology_seed, split, idx),
            )

            mask_valid = sample["mask_valid"]
            observations = sample["observations"]
            landmarks = sample["landmarks"]
            regions = sample["regions"]

            # Build a valid-state index: flatten the 2D mask.
            coords = np.argwhere(mask_valid)
            n_states = len(coords)

            state_to_row_col = np.zeros((n_states, 2), dtype=np.int32)
            valid_state_mask = np.ones(n_states, dtype=bool)
            region_id = np.full(n_states, -1, dtype=np.int32)
            landmark_id = np.full(n_states, -1, dtype=np.int32)

            for i, (r, c) in enumerate(coords):
                state_to_row_col[i, 0] = int(r)
                state_to_row_col[i, 1] = int(c)
                region_id[i] = int(regions[r, c])
                landmark_id[i] = int(landmarks[r, c])

            # Build adjacency over the valid-state index (4-neighbor).
            adj = np.zeros((n_states, n_states), dtype=bool)
            for i, (r, c) in enumerate(coords):
                adj[i, i] = True  # stay-still
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < resolved_h and 0 <= nc < resolved_w:
                        if mask_valid[nr, nc]:
                            j = np.where(
                                (coords[:, 0] == nr) & (coords[:, 1] == nc)
                            )[0][0]
                            adj[i, j] = True

            # Transition matrix from adjacency.
            tm = adj.astype(np.float64)
            row_sums = tm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            tm = tm / row_sums

            for inst_idx in range(n_sensory_instances):
                sensory_seed = (
                    _sample_seed(topology_seed, split, idx) + inst_idx
                )
                sensory_rng = np.random.default_rng(np.uint64(sensory_seed))
                obs_ids = sensory_rng.integers(0, s_size, size=n_states).astype(
                    np.int32
                )

                layout_id = (
                    f"dungeongen-{split}-{idx:06d}" f"-sens{inst_idx:02d}"
                )

                layout: SpatialLayout = {
                    "layout_id": layout_id,
                    "layout_family": "dungeongen",
                    "topology_type": topology_type,
                    "graph_state_count": n_states,
                    "valid_state_mask": valid_state_mask,
                    "state_to_row_col": state_to_row_col,
                    "observation_id": obs_ids,
                    "adjacency": adj,
                    "action_space": dict(DEFAULT_GRID_ACTION_SPACE),
                    "transition_matrix": tm,
                    "topology_seed": topology_seed,
                    "sensory_seed": sensory_seed,
                    "sensory_vocab_size": s_size,
                    "split": split,
                }
                # Attach optional dungeon-specific fields.
                layout["region_id"] = region_id
                layout["landmark_id"] = landmark_id

                validate_spatial_layout(layout)
                layouts.append(layout)

    # Write through the common writer.
    write_layout_dataset(
        layouts,
        version_root,
        topology_type=topology_type,
        s_size=s_size,
        topology_seed=topology_seed,
        version=extract_version(version_root),
        layout_family="dungeongen",
        preset=preset,
    )


__all__ = [
    "SHARED_FAMILY",
    "SHARED_CHANNELS",
    "build_dungeongen_layouts",
    "ensure_raw",
    "prepare_interim",
    "build_shared_substrate",
    "validate_dungeongen_shared_root",
]
