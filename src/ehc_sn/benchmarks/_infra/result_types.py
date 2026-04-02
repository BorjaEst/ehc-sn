"""Lightweight benchmark result structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactRecord:
    """One persisted artifact emitted by a benchmark run."""

    path: Path
    kind: str


@dataclass(frozen=True)
class RunMetadata:
    """Common metadata written alongside benchmark artifacts."""

    benchmark_id: str
    output_root: Path
    seed: int | None = None


__all__ = ["ArtifactRecord", "RunMetadata"]
