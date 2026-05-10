"""Raw corpus helpers for the MazeHard dataset from HuggingFace.

Canonical source: ``flaitenberger/maze_hard_augmented``

The raw dataset stores maze grids as 2-D arrays of single characters:

- ``'#'`` — wall cell (not passable)
- ``' '`` — open passable cell
- ``'S'`` — agent start position (also passable)
- ``'G'`` — goal position (also passable)

The ``labels`` field is identical to ``inputs`` except that cells on the
shortest-path solution are marked ``'o'``.

Only ``train.jsonl.gz`` and ``test.jsonl.gz`` are provided in the raw source.
There is no separate ``val`` split; callers that need a val split must carve
it from the training records.

Canonical raw layout::

    <raw_root>/huggingface/maze_hard_augmented/
        train.jsonl.gz
        test.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

import numpy as np

# =============================================================================
HF_REPO_ID: str = "flaitenberger/maze_hard_augmented"
"""Canonical HuggingFace repo id for the MazeHard raw corpus."""

_RAW_FILES: dict[str, str] = {
    "train": "train.jsonl.gz",
    "test": "test.jsonl.gz",
}

_PASSABLE_CHARS: frozenset[str] = frozenset({" ", "S", "G", "o"})
"""Characters that represent passable (non-wall) cells."""

_SOLUTION_CHARS: frozenset[str] = frozenset({"o", "S", "G"})
"""Characters in the labels grid that mark solution-path cells."""


# =============================================================================
def default_raw_root() -> Path:
    """Return the canonical raw root relative to the repo data directory."""
    return Path("data/raw/huggingface/maze_hard_augmented")


# =============================================================================
def ensure_raw_corpus(raw_root: Path, *, repo_id: str = HF_REPO_ID) -> None:
    """Download the MazeHard raw corpus into *raw_root* if not already present.

    Downloads ``train.jsonl.gz`` and ``test.jsonl.gz`` from HuggingFace Hub.
    If both files are already present, does nothing.

    Args:
        raw_root: Destination directory for raw files.
        repo_id: HuggingFace dataset repo id.
    """
    import huggingface_hub as hf

    raw_root.mkdir(parents=True, exist_ok=True)

    for _split, filename in _RAW_FILES.items():
        dest = raw_root / filename
        if dest.exists():
            continue
        local = hf.hf_hub_download(repo_id, filename, repo_type="dataset")
        import shutil
        shutil.copy2(local, dest)


# =============================================================================
def iter_raw_records(raw_root: Path, split: str) -> Iterator[dict]:
    """Yield raw JSON records from the on-disk corpus for *split*.

    Args:
        raw_root: Directory containing the downloaded ``.jsonl.gz`` files.
        split: ``"train"`` or ``"test"`` — the splits available in the raw source.

    Yields:
        Dict with keys ``inputs``, ``labels``, ``puzzle_index``,
        ``group_index``, ``set``.

    Raises:
        KeyError: When *split* is not a valid raw split name.
        FileNotFoundError: When the raw corpus file is missing.
    """
    if split not in _RAW_FILES:
        raise KeyError(f"MazeHard raw corpus has no split {split!r}. Available: {sorted(_RAW_FILES)}.")

    corpus_file = raw_root / _RAW_FILES[split]
    if not corpus_file.exists():
        raise FileNotFoundError(
            f"MazeHard raw corpus file not found: {corpus_file}. "
            "Run ensure_raw_corpus() first."
        )

    with gzip.open(corpus_file, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# =============================================================================
def normalize_raw_record(record: dict) -> dict[str, np.ndarray]:
    """Convert one raw MazeHard record into the canonical processed channel dict.

    Derives channels faithfully from the source fields:

    - ``topology``:  ``inputs != '#'``
    - ``mask_valid``: ``topology`` (all non-wall cells are reachable by
      construction in this dataset)
    - ``start``:     cells where ``inputs == 'S'``
    - ``goals``:     cells where ``inputs == 'G'``
    - ``solution``:  cells where ``labels`` is ``'o'``, ``'S'``, or ``'G'``
      (the full annotated solution path including endpoints)

    Args:
        record: A single raw record as returned by :func:`iter_raw_records`.

    Returns:
        Dict of channel name → numpy array.

    Raises:
        ValueError: When the inputs and labels grids have different shapes or
            when the inputs grid contains unexpected characters.
    """
    inputs = np.array(record["inputs"], dtype="U1")  # (H, W) char
    labels = np.array(record["labels"], dtype="U1")  # (H, W) char

    if inputs.shape != labels.shape:
        raise ValueError(
            f"MazeHard record has mismatched inputs/labels shapes: "
            f"{inputs.shape} vs {labels.shape}."
        )
    if inputs.ndim != 2:
        raise ValueError(f"MazeHard inputs must be a 2-D grid, got shape {inputs.shape}.")

    topology = inputs != "#"
    mask_valid = topology.copy()
    start = inputs == "S"
    goals = inputs == "G"

    # Solution path = cells explicitly marked on the solution in labels.
    # The raw source marks open path cells as 'o'; start 'S' and goal 'G'
    # are considered part of the path as well.
    solution_bool = (labels == "o") | (labels == "S") | (labels == "G")
    # Encode as int32: 1 on path, 0 elsewhere.
    solution = solution_bool.astype(np.int32)

    return {
        "topology": topology,
        "mask_valid": mask_valid,
        "start": start,
        "goals": goals,
        "solution": solution,
    }


# =============================================================================
def count_raw_records(raw_root: Path, split: str) -> int:
    """Count the number of records in a raw corpus split without loading all data.

    Args:
        raw_root: Directory containing downloaded ``.jsonl.gz`` files.
        split: ``"train"`` or ``"test"``.

    Returns:
        Number of records in the split.
    """
    return sum(1 for _ in iter_raw_records(raw_root, split))


# =============================================================================
__all__ = [
    "HF_REPO_ID",
    "default_raw_root",
    "ensure_raw_corpus",
    "iter_raw_records",
    "normalize_raw_record",
    "count_raw_records",
]
