"""Shared-substrate builder for the dungeongen dataset family.

Orchestrates the shared-substrate pipeline for Dungeon:

1. ``dungeon_raw.ensure_raw_snapshot`` — create/validate canonical raw snapshot.
2. ``prepare_dungeongen_interim`` — normalize raw topologies into per-split NPZ files.
3. ``iter_interim_topologies`` — stream topology arrays from the interim split files.
4. Pad and augment each topology into shared-substrate channels.
5. ``_writer.create_version_root`` — create the immutable version leaf.
6. ``_writer.write_split`` / ``write_index_at_root`` — write processed output.
7. ``manifest.write_manifest`` — write the authoritative root manifest.

Shared-substrate channels (topology, observations, mask_valid, regions,
landmarks) are task-neutral.  Trajectory and replay channels belong in the
Dungeon task corpus; see ``ehc_sn.tasks.dungeon.data``.

Interim layer: ``data/interim/dungeongen/`` — one NPZ file per split.
Shared substrate: ``data/processed/dungeongen/v<version>/``
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterator

import numpy as np

from ehc_sn.data._canonical import (
    binary_structural_landmarks,
    largest_component_mask,
    sample_observations,
)
from ehc_sn.data._writer import _extract_version, _staging_root, write_index_at_root, write_split
from ehc_sn.data.dungeon_raw import iter_raw_topologies
from ehc_sn.data.manifest import write_manifest

# ---------------------------------------------------------------------------
SHARED_FAMILY: Final[str] = "dungeongen"
"""Shared-substrate family name for the dungeongen source."""

DUNGEON_SUBSTRATE_CHANNELS: Final[list[str]] = [
    "topology",
    "observations",
    "mask_valid",
    "regions",
    "landmarks",
]
"""Shared-substrate channels (task-neutral layout data)."""

_SPLITS: tuple[str, ...] = ("train", "val", "test")
_SPLIT_SEED_OFFSET: dict[str, int] = {"train": 0, "val": 100_000, "test": 200_000}
_SOURCE_ID: Final[str] = "dungeongen"
"""Stable upstream source identifier."""


# ---------------------------------------------------------------------------
def prepare_dungeongen_interim(
    raw_root: Path,
    interim_root: Path,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
) -> None:
    """Normalize the raw snapshot into per-split NPZ files in the interim layer.

    Reads topology data from the canonical raw snapshot via
    :func:`~ehc_sn.data.dungeon_raw.iter_raw_topologies` and writes one NPZ
    file per split to ``data/interim/dungeongen/<split>.npz``.  Each split
    NPZ stores deterministically normalized arrays padded to the split's max
    shape, suitable for downstream substrate building without re-reading raw
    shards.

    This is a real normalization boundary: the interim format is materially
    different from the raw tar-sharded snapshot.  It does not preserve tar
    packaging or per-sample filesystem fan-out.

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

        for idx, (topology, regions, seed) in enumerate(iter_raw_topologies(raw_root, split)):
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

        # Pad to max shape within split.
        h_max = int(max(h for h in heights))
        w_max = int(max(w for w in widths))
        topo_arr = np.zeros((count, h_max, w_max), dtype=bool)
        reg_arr = np.full((count, h_max, w_max), -1, dtype=np.int32)
        for i, (topo, reg, h, w) in enumerate(zip(topologies, regions_list, heights, widths)):
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


def iter_interim_topologies(
    interim_root: Path, split: str
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield ``(topology, regions, seed)`` from the dungeongen interim split file.

    Reconstructs native-shape arrays by slicing with the stored ``height`` and
    ``width`` arrays — no padding is exposed to callers.

    Args:
        interim_root: Interim root (e.g. ``data/interim/dungeongen``).
        split: Split name (``"train"``, ``"val"``, or ``"test"``).

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
def _pad_to_shape(arr: np.ndarray, target_h: int, target_w: int, *, fill: int | bool) -> np.ndarray:
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
    n_observations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Normalize a raw dungeongen topology into shared-substrate channels."""
    rng = np.random.default_rng(seed)

    topology = _pad_to_shape(topology, target_h, target_w, fill=False)
    regions = _pad_to_shape(dungeongen_regions, target_h, target_w, fill=-1).astype(np.int32)

    mask_valid = largest_component_mask(topology)
    observations = sample_observations(mask_valid, n_observations, seed=int(rng.integers(2**31)))
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


# ---------------------------------------------------------------------------
def _infer_shape(
    interim_root: Path,
    split_counts: dict[str, int],
) -> tuple[int, int]:
    """Return (max_height, max_width) across all selected interim samples.

    Scans the first ``n`` samples for each split as given by *split_counts*
    and returns the per-dimension maximum.  This is the minimum target shape
    that can accommodate every selected sample without discarding cells.

    Args:
        interim_root: Interim root (e.g. ``data/interim/dungeongen``).
        split_counts: Mapping of split name → number of samples to consider.

    Returns:
        ``(max_height, max_width)`` as a two-int tuple.
    """
    max_h = 0
    max_w = 0
    for split, n in split_counts.items():
        for idx, (topology, _, _) in enumerate(iter_interim_topologies(interim_root, split)):
            if idx >= n:
                break
            h, w = topology.shape
            if h > max_h:
                max_h = h
            if w > max_w:
                max_w = w
    return max_h, max_w


# ---------------------------------------------------------------------------
def build_dungeongen_substrate(
    version_root: Path,
    *,
    interim_root: Path,
    n_train: int = 200,
    n_val: int = 40,
    n_test: int = 40,
    height: int | None = None,
    width: int | None = None,
    n_observations: int = 6,
    seed: int = 42,
) -> None:
    """Build the dungeongen shared substrate at *version_root*.

    Reads topology files from *interim_root* (produced by
    :func:`prepare_dungeongen_interim`), pads each topology, and assigns
    shared-substrate channels (no trajectory data).

    When *height* and/or *width* are omitted (``None``), the required
    processed grid dimensions are inferred from the selected interim slice:
    the maximum height and maximum width across all samples in the chosen
    ``n_train``/``n_val``/``n_test`` slice.  Passing an explicit value
    smaller than the inferred maximum raises ``ValueError`` immediately.

    The version integer is derived from the ``v<N>`` leaf of *version_root*;
    there is no separate ``version`` parameter.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/dungeongen/v1``).  Must not already exist.
        interim_root: Interim leaf (e.g. ``data/interim/dungeongen``).
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        height: Target grid height after padding.  When ``None`` (default),
            inferred as the maximum height across the selected interim slice.
        width: Target grid width after padding.  When ``None`` (default),
            inferred as the maximum width across the selected interim slice.
        n_observations: Number of distinct observation ids to assign.
        seed: Base RNG seed; per-sample seeds are derived deterministically.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        ValueError: When an explicit *height* or *width* is smaller than the
            maximum shape required by the selected interim slice.
        RuntimeError: When augmentation encounters an empty valid mask.
    """
    version = _extract_version(version_root)
    split_counts = {"train": n_train, "val": n_val, "test": n_test}

    # Resolve grid dimensions — infer from interim slice when not explicit.
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
    stage_params = {
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "height": resolved_h, "width": resolved_w,
        "n_observations": n_observations, "seed": seed,
    }

    with _staging_root(version_root) as tmp:
        all_entries = []
        for split in _SPLITS:
            n = split_counts[split]
            interim_topologies = list(iter_interim_topologies(interim_root, split))[:n]
            samples = [
                _build_substrate_sample(
                    topology,
                    dungeon_regions,
                    target_h=resolved_h,
                    target_w=resolved_w,
                    n_observations=n_observations,
                    seed=_sample_seed(seed, split, idx),
                )
                for idx, (topology, dungeon_regions, _) in enumerate(interim_topologies)
            ]
            entries = write_split(
                tmp,
                split,
                samples,
                source=SHARED_FAMILY,
                shape=shape,
                channels=DUNGEON_SUBSTRATE_CHANNELS,
                spatial_channels=DUNGEON_SUBSTRATE_CHANNELS,
                index_kwargs={"n_observations": n_observations, "n_goals": 0, "difficulty": "medium"},
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="shared_substrate",
            family=SHARED_FAMILY,
            version=version,
            channels=DUNGEON_SUBSTRATE_CHANNELS,
            shape=shape,
            n_samples=split_counts,
            source_id=_SOURCE_ID,
            builder="ehc_sn.data.dungeon_builder.build_dungeongen_substrate",
            seed=seed,
            stage_params=stage_params,
        )

    n_total = n_train + n_val + n_test
    print(f"dungeongen shared substrate written to {version_root}  ({n_total} samples.)")


__all__ = [
    "SHARED_FAMILY",
    "DUNGEON_SUBSTRATE_CHANNELS",
    "_RAW_SEED_OFFSET",
    "prepare_dungeongen_interim",
    "iter_interim_topologies",
    "build_dungeongen_substrate",
]
