"""B0-specific artifact manifest structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class B0Manifest:
    """Summary of persisted B0 artifacts."""

    output_root: Path
    artifact_paths: tuple[Path, ...] = ()


__all__ = ["B0Manifest"]
