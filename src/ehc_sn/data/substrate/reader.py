"""Helpers for reading versioned shared-substrate roots.

Each split directory in a shared-substrate root stores one ``.npy`` file per
channel.  These helpers load those arrays and yield per-sample dicts without
holding all splits in memory at once.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ehc_sn.data.index import DatasetIndexEntry, read_index
from ehc_sn.data.manifest import read_manifest


def iter_substrate_samples(
    substrate_root: Path,
    split: str,
    channels: list[str],
) -> Iterator[dict[str, np.ndarray]]:
    """Yield per-sample dicts from a shared substrate versioned root.

    Args:
        substrate_root: Versioned root of the shared substrate
            (e.g. ``data/processed/dungeongen/v1``).
        split: One of ``"train"``, ``"val"``, ``"test"``.
        channels: Channel names to load.

    Yields:
        Dict mapping each channel name to a single-sample numpy array.

    Raises:
        FileNotFoundError: When a channel file is missing.
    """
    split_dir = substrate_root / split
    arrays = {ch: np.load(split_dir / f"{ch}.npy") for ch in channels}
    n = next(iter(arrays.values())).shape[0]
    for i in range(n):
        yield {ch: arrays[ch][i] for ch in channels}


def iter_substrate_entries_and_samples(
    substrate_root: Path,
    split: str,
    channels: list[str],
) -> Iterator[tuple[DatasetIndexEntry, dict[str, np.ndarray]]]:
    """Yield ``(entry, sample)`` pairs from a shared substrate versioned root.

    Pairs each ``index.jsonl`` entry with its corresponding per-sample channel
    arrays.  Use this when stable source identity (``entry.source_record_id``)
    is needed for joining with raw-source records.

    Args:
        substrate_root: Versioned root of the shared substrate.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        channels: Channel names to load.

    Yields:
        ``(DatasetIndexEntry, dict[str, ndarray])`` pairs in index order.

    Raises:
        FileNotFoundError: When the index or a channel file is missing.
        ValueError: When the index entry count does not match array length.
    """
    entries = [e for e in read_index(substrate_root / "index.jsonl") if e.split == split]
    split_dir = substrate_root / split
    arrays = {ch: np.load(split_dir / f"{ch}.npy") for ch in channels}
    n_arrays = next(iter(arrays.values())).shape[0]
    if len(entries) != n_arrays:
        raise ValueError(f"Index has {len(entries)} {split!r} entries but arrays have {n_arrays} samples " f"in {substrate_root}.")
    for i, entry in enumerate(entries):
        yield entry, {ch: arrays[ch][i] for ch in channels}


def load_substrate_manifest(substrate_root: Path) -> dict:
    """Return the manifest dict for a shared substrate root.

    Args:
        substrate_root: Versioned root directory.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: When no manifest exists at *substrate_root*.
        ValueError: When the root is not a ``shared_substrate``.
    """
    manifest = read_manifest(substrate_root)
    if manifest.get("dataset_class") != "shared_substrate":
        raise ValueError(f"Expected a shared_substrate root, got dataset_class=" f"{manifest.get('dataset_class')!r} at {substrate_root}.")
    return manifest


__all__ = [
    "iter_substrate_samples",
    "iter_substrate_entries_and_samples",
    "load_substrate_manifest",
]
