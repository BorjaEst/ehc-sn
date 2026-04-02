"""Small filesystem helpers shared by benchmark internals."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create ``path`` and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["ensure_directory"]
