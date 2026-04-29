"""I/O helpers for writing canonical versioned dataset roots.

Owns the I/O concerns shared across builder scripts: creating immutable
version-leaf directories, stacking and writing per-channel ``.npy`` files,
writing ``dataset.json``, and collecting ``index.jsonl`` entries.

Dataset roots are immutable.  Builders must use :func:`staging_root` for
transactional materialization: all output goes to a temp sibling dir, is
structurally validated, then atomically renamed to the final version leaf.
"""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from ehc_sn.data.index import DatasetIndexEntry, write_index

# ---------------------------------------------------------------------------
# Version-path helpers
# ---------------------------------------------------------------------------


def extract_version(version_root: Path) -> int:
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
        raise ValueError(f"Version root leaf must be 'v<integer>', got: {name!r}.  " "Use a path like 'data/processed/numberline/v1'.")
    return int(name[1:])


# ---------------------------------------------------------------------------
# Transactional materialization
# ---------------------------------------------------------------------------


@contextmanager
def staging_root(version_root: Path) -> Iterator[Path]:
    """Context manager for transactional dataset materialization.

    Materialises into a temporary sibling directory, runs a structural
    validation, then atomically renames to the final version root.  On any
    failure the temporary directory is removed and the final root is never
    created.

    Args:
        version_root: The intended final versioned root (e.g.
            ``data/processed/numberline/v1``).  Must not already exist.

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
    extract_version(version_root)  # validate leaf name before starting work
    tmp = version_root.parent / f".building-{version_root.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        yield tmp
        from ehc_sn.data.lifecycle._validate import _validate_structure

        _validate_structure(tmp)
        tmp.rename(version_root)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Direct root creation
# ---------------------------------------------------------------------------


def create_version_root(version_root: Path) -> None:
    """Create an immutable version-leaf directory.

    Args:
        version_root: Target directory (e.g. ``data/processed/numberline/v1``).

    Raises:
        FileExistsError: When *version_root* already exists.
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
    channels: list[str],
    topology_kind: str,
    n_states: int,
    extent: list[int],
    index_kwargs: dict[str, Any],
    per_sample_extra: list[dict] | None = None,
    sample_validator: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> list[DatasetIndexEntry]:
    """Stack and write one split to disk; return index entries.

    Args:
        output_root: Dataset root (already created via :func:`create_version_root`).
        split: Split name, e.g. ``"train"``.
        samples: List of per-sample channel dicts.
        source: Dataset source identifier.
        channels: Ordered list of all channel names to write.
        topology_kind: Canonical topology kind (e.g. ``"grid2d"``, ``"line1d"``).
        n_states: Total number of states in the topology.
        extent: Topology extent list (e.g. ``[H, W]`` or ``[N]``).
        index_kwargs: Extra keyword arguments forwarded to :class:`DatasetIndexEntry`.
        per_sample_extra: Optional per-sample extra fields for index entries.
        sample_validator: Optional callable called on every sample after stacking.

    Returns:
        List of :class:`DatasetIndexEntry` for the split.
    """
    n = len(samples)
    split_dir = output_root / split
    split_dir.mkdir()

    stacked: dict[str, np.ndarray] = {ch: np.stack([s[ch] for s in samples], axis=0) for ch in channels}

    if sample_validator is not None:
        for i in range(n):
            sample_validator({ch: stacked[ch][i] for ch in channels})

    for ch, arr in stacked.items():
        np.save(split_dir / f"{ch}.npy", arr)

    (split_dir / "dataset.json").write_text(
        json.dumps(
            {
                "source": source,
                "split": split,
                "n_samples": n,
                "topology_kind": topology_kind,
                "n_states": n_states,
                "extent": extent,
                "channels": channels,
            },
            indent=2,
        )
    )

    return [
        DatasetIndexEntry(
            id=f"{source}-{split}-{idx + 1:06d}",
            source=source,
            split=split,
            channels=channels,
            **index_kwargs,
            **(per_sample_extra[idx] if per_sample_extra else {}),
        )
        for idx in range(n)
    ]


def write_index_at_root(
    entries: list[DatasetIndexEntry],
    output_root: Path,
) -> None:
    """Write the canonical ``index.jsonl`` at the dataset root.

    Args:
        entries: All index entries across all splits.
        output_root: Version-leaf root directory.
    """
    write_index(entries, output_root / "index.jsonl")


__all__ = [
    "extract_version",
    "staging_root",
    "create_version_root",
    "write_split",
    "write_index_at_root",
]
