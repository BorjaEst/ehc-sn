"""Artifact-writing helpers shared by benchmark evaluators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ehc_sn.benchmarks._infra.io import ensure_directory


def write_artifact_json(path: Path, payload: Any) -> Path:
    """Write one JSON artifact and return its path."""
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = ["write_artifact_json"]
