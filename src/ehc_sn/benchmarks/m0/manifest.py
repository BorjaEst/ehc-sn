"""M0-specific artifact manifest structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ehc_sn.benchmarks.m0.metrics import MetricMap


@dataclass(frozen=True)
class M0Manifest:
    """Summary of persisted M0 artifacts."""

    output_root: Path
    artifact_paths: tuple[Path, ...] = ()
    metrics: MetricMap | None = None


__all__ = ["M0Manifest"]
