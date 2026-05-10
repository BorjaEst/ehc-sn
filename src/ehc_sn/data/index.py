"""Dataset index types and JSONL I/O.

Public surface: :class:`DatasetIndexEntry`, :func:`read_index`,
:func:`write_index`, :func:`filter_index`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# =================================================================================================
class DatasetIndexEntry(BaseModel, extra="forbid"):
    """One sample entry in a shared-substrate or task-corpus index.

    Index entries are stored as JSONL and describe how to find a sample in a
    versioned processed root: materialized channels, split, and source provenance.
    Topology and spatial geometry are owned by the root manifest and family validators.
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
    channels: list[str] = Field(..., description="Canonical processed channel names materialized for the sample")
    task_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional task-owned per-sample provenance object. Structure is task-defined; "
        "tasks document the schema in their corpus spec section. Null unless the task builder supplies it.",
    )


# =================================================================================================
def read_index(path: Path,) -> list[DatasetIndexEntry]:  # fmt: skip  # --------------------------------------------------------------------------------
    """Read all entries from a JSONL index file.

    Args:
        path: Path to the ``index.jsonl`` file.

    Returns:
        List of :class:`DatasetIndexEntry` objects, one per line.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    with path.open() as f:
        return [DatasetIndexEntry.model_validate_json(line) for line in f if line.strip()]


# =================================================================================================
def write_index(entries: list[DatasetIndexEntry], path: Path,) -> None:  # fmt: skip  # -------------------------------------------------------------------------------
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
def filter_index(entries: list[DatasetIndexEntry], *, split: str | None = None, source: (
    str | None
) = None,) -> list[DatasetIndexEntry]:  # fmt: skip  # ------------------------------------------------------------------------------
    """Filter index entries by metadata predicates.

    Args:
        entries: Source list to filter.
        split: Keep only entries whose ``split`` matches (e.g. ``"train"``).
        source: Keep only entries whose ``source`` matches (e.g. ``"huggingface"``).

    Returns:
        Filtered list (original list is not modified).
    """
    result = entries
    if split is not None:
        result = [e for e in result if e.split == split]
    if source is not None:
        result = [e for e in result if e.source == source]
    return result


# =================================================================================================
__all__ = ["DatasetIndexEntry", "read_index", "write_index", "filter_index"]
