"""I/O helpers for writing canonical versioned dataset roots.

Owns the I/O concerns shared across builder scripts: creating immutable
version-leaf directories, stacking and writing per-channel ``.npy`` files,
writing ``dataset.json``, and collecting ``index.jsonl`` entries.

Dataset roots are immutable.  Builders must use :func:`_staging_root` for
transactional materialization: all output goes to a temp sibling dir, is
structurally validated, then atomically renamed to the final version leaf.

Source-specific processing logic belongs in source-specific modules
(e.g. ``dungeon_builder.py``, ``mazehard_builder.py``).
"""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ehc_sn.data.index import MazeIndexEntry, write_index
from ehc_sn.data.schema import validate_processed


# ---------------------------------------------------------------------------
# Version-path helpers
# ---------------------------------------------------------------------------


def _extract_version(version_root: Path) -> int:
    """Extract the version integer from a canonical version leaf name.

    Args:
        version_root: Path whose leaf must be ``v<integer>`` (e.g. ``…/v1``).

    Returns:
        The version integer.

    Raises:
        ValueError: When the leaf name is not ``v<integer>``.
    """
    name = version_root.name
    if not (name.startswith("v") and name[1:].isdigit()):
        raise ValueError(
            f"Version root leaf must be 'v<integer>', got: {name!r}.  "
            "Use a path like 'data/processed/maze-nd/v1'."
        )
    return int(name[1:])


# ---------------------------------------------------------------------------
# Transactional materialization
# ---------------------------------------------------------------------------


@contextmanager
def _staging_root(version_root: Path) -> Iterator[Path]:
    """Context manager for transactional dataset materialization.

    Materialises into a temporary sibling directory, runs a structural
    validation, then atomically renames to the final version root.  On any
    failure the temporary directory is removed and the final root is never
    created.

    Args:
        version_root: The intended final versioned root (e.g.
            ``data/processed/maze-nd/v1``).  Must not already exist.

    Yields:
        A temporary sibling directory to write all output into.

    Raises:
        FileExistsError: When *version_root* already exists.
        ValueError: When the version leaf name is not ``v<integer>``.
    """
    if version_root.exists():
        raise FileExistsError(
            f"Version root already exists (dataset roots are immutable): {version_root}\n"
            "Bump the version integer to create a new version."
        )
    _extract_version(version_root)  # validate leaf name before starting work
    tmp = version_root.parent / f".building-{version_root.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        yield tmp
        from ehc_sn.data._validator import _validate_structure  # local to avoid circular at module level

        _validate_structure(tmp)
        tmp.rename(version_root)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Direct root creation (for tests and non-transactional fixture setup only)
# ---------------------------------------------------------------------------


def create_version_root(version_root: Path) -> None:
    """Create an immutable version-leaf directory.

    Args:
        version_root: Target directory (e.g. ``data/processed/maze-nd/v1``).

    Raises:
        FileExistsError: When *version_root* already exists.  Dataset roots
            are immutable; bump the version integer to create a new version.
    """
    if version_root.exists():
        raise FileExistsError(
            f"Version root already exists (dataset roots are immutable): {version_root}\n"
            "Bump the version integer to create a new version."
        )
    version_root.mkdir(parents=True)


def write_split(
    output_root: Path,
    split: str,
    samples: list[dict[str, np.ndarray]],
    *,
    source: str,
    shape: tuple[int, int],
    channels: list[str],
    spatial_channels: list[str],
    index_kwargs: dict[str, Any],
    per_sample_extra: list[dict] | None = None,
    sample_validator: Any | None = None,
) -> list[MazeIndexEntry]:
    """Stack, validate, and write one split to disk; return index entries.

    Args:
        output_root: Dataset root (already created via
            :func:`create_version_root`).
        split: Split name, e.g. ``"train"``.
        samples: List of per-sample channel dicts.
        source: Dataset source identifier, e.g. ``"maze-nd"``.
        shape: Spatial grid shape ``(H, W)``.
        channels: Ordered list of all channel names to write.
        spatial_channels: Subset of *channels* that are ``(H, W)`` arrays
            to pass to :func:`~ehc_sn.data.schema.validate_processed`.
        index_kwargs: Extra keyword arguments forwarded to
            :class:`~ehc_sn.data.index.MazeIndexEntry` (all samples share these).
        per_sample_extra: Optional per-sample extra fields for index entries.
            If provided, must have the same length as *samples*.
        sample_validator: Optional callable ``(dict) -> None`` called on
            every sample after stacking.  Raises ``ValueError`` on violations.

    Returns:
        List of :class:`~ehc_sn.data.index.MazeIndexEntry` for the split.
    """
    n = len(samples)
    split_dir = output_root / split
    split_dir.mkdir()

    stacked: dict[str, np.ndarray] = {ch: np.stack([s[ch] for s in samples], axis=0) for ch in channels}

    for i in range(n):
        validate_processed({ch: stacked[ch][i] for ch in spatial_channels})
        if sample_validator is not None:
            sample_validator({ch: stacked[ch][i] for ch in channels})

    for ch, arr in stacked.items():
        np.save(split_dir / f"{ch}.npy", arr)

    (split_dir / "dataset.json").write_text(
        json.dumps(
            {
                "source": source,
                "split": split,
                "n_samples": n,
                "shape": list(shape),
                "channels": channels,
            },
            indent=2,
        )
    )

    return [
        MazeIndexEntry(
            id=f"{source}-{split}-{idx + 1:06d}",
            source=source,
            split=split,
            shape=shape,
            channels=channels,
            **index_kwargs,
            **(per_sample_extra[idx] if per_sample_extra else {}),
        )
        for idx in range(n)
    ]


def write_index_at_root(
    entries: list[MazeIndexEntry],
    output_root: Path,
) -> None:
    """Write the canonical ``index.jsonl`` at the dataset root.

    Args:
        entries: All index entries across all splits.
        output_root: Version-leaf root directory.
    """
    write_index(entries, output_root / "index.jsonl")


__all__ = [
    "create_version_root",
    "write_split",
    "write_index_at_root",
]
