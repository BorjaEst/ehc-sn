"""Benchmark-manifest schemas and I/O helpers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class MazeHardSubsetManifest(BaseModel, extra="forbid"):
    """Persisted B0 hard-subset selection for one MazeHard split."""

    dataset_root: str = Field(..., description="Processed MazeHard dataset root.")
    source: str = Field(..., description="Source dataset identifier.")
    split: str = Field(..., description="Dataset split used to derive the subset.")
    selection_rule: str = Field(..., description="Human-readable subset selection rule.")
    sample_ids: list[str] = Field(..., description="Canonical maze ids in the hard subset.")
    difficulty_values: list[int] = Field(..., description="Difficulty values aligned with sample_ids.")
    n_selected: int = Field(..., ge=0, description="Number of selected samples.")


def b0_hard_subset_path(dataset_root: Path) -> Path:
    """Return the canonical B0 hard-subset manifest path for ``dataset_root``."""
    return dataset_root / "b0-hard-subset.json"


def write_mazehard_subset_manifest(manifest: MazeHardSubsetManifest, path: Path) -> None:
    """Serialize a MazeHard hard-subset manifest to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def read_mazehard_subset_manifest(path: Path) -> MazeHardSubsetManifest:
    """Load a MazeHard hard-subset manifest from JSON."""
    return MazeHardSubsetManifest.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "MazeHardSubsetManifest",
    "b0_hard_subset_path",
    "read_mazehard_subset_manifest",
    "write_mazehard_subset_manifest",
]
