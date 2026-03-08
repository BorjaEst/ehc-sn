from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


# =================================================================================================
class MazeIndexEntry(BaseModel, extra="forbid"):
    """One entry in a maze dataset index.

    Index entries are stored as JSONL and describe where and how to interpret a
    sample dataset (shape, channels, split/source metadata).
    """

    id: str = Field(..., description="Unique identifier for the maze (e.g., 'maze_00001')")
    source: str = Field(..., description="Source dataset name (e.g., 'huggingface')")
    split: str = Field(..., description="Split name (e.g., 'train', 'val', 'test')")

    shape: tuple[int, int] = Field(..., description="Maze shape as (height, width)")

    @property
    def height(self) -> int:
        """Maze grid height (rows)."""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Maze grid width (columns)."""
        return self.shape[1]

    channels: list[str] = Field(..., description="List of channel names (e.g., ['observation', 'goal'])")

    n_observations: int = Field(
        default=0,
        ge=0,
        description="Number of observation channels (e.g., 1 for top-down view, >1 for multi-view)",
    )
    n_goals: int = Field(
        default=0,
        ge=0,
        description="Number of goal channels (e.g., 1 for single-goal mazes, >1 for multi-goal mazes)",
    )
    difficulty: str = Field(
        default="unknown",
        description="Difficulty level (e.g., 'easy', 'medium', 'hard')",
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
    entries: list[MazeIndexEntry], path: Path, *, append: bool = False,
) -> None:  # fmt: skip
    """Write entries to a JSONL index file.

    Args:
        entries: Entries to write.
        path: Destination path (created if needed).
        append: If ``True``, append to an existing file; otherwise overwrite.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode) as f:
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
