"""SeqMaze corpus reader — load persisted arrays from a versioned root.

Pattern mirrors ``goaltrace/corpus.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ehc_sn.data.manifest import read_manifest
from ehc_sn.tasks.seqmaze.builder import SEQMAZE_TASK_CHANNELS

_SEQMAZE_CHANNELS: tuple[str, ...] = tuple(SEQMAZE_TASK_CHANNELS)


def load_split_arrays(root: Path, split: str) -> dict[str, np.ndarray] | None:
    """Load all channel arrays for one split, mmap'd.

    Args:
        root: Versioned corpus root
            (e.g. ``data/processed/seqmaze/default/v1``).
        split: Split name (e.g. ``"train"``).

    Returns:
        Dict mapping channel names to ``(N, ...)`` arrays, or ``None`` if
        the split directory does not exist.
    """
    split_dir = root / split
    if not split_dir.is_dir():
        return None
    arrays: dict[str, np.ndarray] = {}
    for ch in _SEQMAZE_CHANNELS:
        fpath = split_dir / f"{ch}.npy"
        if fpath.exists():
            arrays[ch] = np.load(fpath, mmap_mode="r")
    return arrays if arrays else None


def load_sample(
    root: Path,
    split: str,
    index: int,
    channels: tuple[str, ...] = _SEQMAZE_CHANNELS,
) -> dict[str, np.ndarray]:
    """Load a single sample from the corpus.

    Args:
        root: Versioned corpus root.
        split: Split name.
        index: Sample index within the split.
        channels: Channel names to load (default: all seqmaze channels).

    Returns:
        Dict mapping channel names to 1-D arrays for one sample.

    Raises:
        FileNotFoundError: When the split directory does not exist.
        IndexError: When *index* is out of range.
    """
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    sample: dict[str, np.ndarray] = {}
    for ch in channels:
        fpath = split_dir / f"{ch}.npy"
        if not fpath.exists():
            raise FileNotFoundError(f"Channel file not found: {fpath}")
        arr = np.load(fpath, mmap_mode="r")
        if index >= arr.shape[0]:
            raise IndexError(
                f"Index {index} out of range for '{ch}' "
                f"(size {arr.shape[0]})"
            )
        sample[ch] = np.asarray(arr[index])
    return sample


def load_split_manifest(root: Path) -> dict:
    """Read and return the manifest for a versioned corpus root."""
    return read_manifest(root)


__all__ = [
    "load_sample",
    "load_split_arrays",
    "load_split_manifest",
]
