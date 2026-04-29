"""Dataset index types and JSONL I/O.

Public surface: :class:`MazeIndexEntry`, :func:`read_index`,
:func:`write_index`, :func:`filter_index`.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


# =================================================================================================
class MazeIndexEntry(BaseModel, extra="forbid"):
    """One sample entry in a shared-substrate or task-corpus index.

    Index entries are stored as JSONL and describe how to interpret a sample in
    a versioned processed root: spatial shape, materialized channels, split,
    and source provenance.
    """

    id: str = Field(..., description="Unique sample identifier within the versioned processed root")
    source: str = Field(..., description="Source family or task corpus that materialized the sample")
    split: str = Field(..., description="Canonical split name recorded for the sample")
    source_record_id: str | None = Field(
        default=None,
        description="Stable raw-source record identity (e.g. 'train:12345' for MazeHard puzzle_index). "
        "Used by task builders to recover task-owned channels by source identity, "
        "not by split-local position.",
    )

    shape: tuple[int, int] = Field(..., description="Spatial grid shape as (height, width)")

    @property
    def height(self) -> int:
        """Maze grid height (rows)."""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Maze grid width (columns)."""
        return self.shape[1]

    channels: list[str] = Field(..., description="Canonical processed channel names materialized for the sample")

    n_observations: int = Field(
        default=0,
        ge=0,
        description="Legacy count of task-visible observation channels when the corpus defines them",
    )
    n_goals: int = Field(
        default=0,
        ge=0,
        description="Legacy count of goal-like task channels when the corpus defines them",
    )
    difficulty: str = Field(
        default="unknown",
        description="Optional task-corpus difficulty label for the sample",
    )


# =================================================================================================
def read_index(  # --------------------------------------------------------------------------------
    path: Path,
) -> list[MazeIndexEntry]:  # fmt: skip
    """Read all entries from a JSONL index file.

    Args:
        path: Path to the ``index.jsonl`` file.

    Returns:
        List of :class:`MazeIndexEntry` objects, one per line.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    with path.open() as f:
        return [MazeIndexEntry.model_validate_json(line) for line in f if line.strip()]


# =================================================================================================
def write_index(  # -------------------------------------------------------------------------------
    entries: list[MazeIndexEntry], path: Path,
) -> None:  # fmt: skip
    """Write entries to a JSONL index file.

    Args:
        entries: Entries to write.
        path: Destination path (created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for entry in entries:
            f.write(entry.model_dump_json() + "\n")


# =================================================================================================
def filter_index(  # ------------------------------------------------------------------------------
    entries: list[MazeIndexEntry], *,
    split: str | None = None, source: str | None = None, min_size: int | None = None,
) -> list[MazeIndexEntry]:  # fmt: skip
    """Filter index entries by metadata predicates.

    Args:
        entries: Source list to filter.
        split: Keep only entries whose ``split`` matches (e.g. ``"train"``).
        source: Keep only entries whose ``source`` matches (e.g. ``"huggingface"``).
        min_size: Keep only entries where both ``height >= min_size`` and
            ``width >= min_size``.

    Returns:
        Filtered list (original list is not modified).
    """
    result = entries
    if split is not None:
        result = [e for e in result if e.split == split]
    if source is not None:
        result = [e for e in result if e.source == source]
    if min_size is not None:
        result = [e for e in result if e.height >= min_size and e.width >= min_size]
    return result


# =================================================================================================
__all__ = ["MazeIndexEntry", "read_index", "write_index", "filter_index"]
