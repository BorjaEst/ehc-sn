"""B0-specific artifact manifest structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ehc_sn.benchmarks.b0.metrics import MetricMap


@dataclass(frozen=True)
class B0Manifest:
    """Summary of persisted B0 artifacts."""

    output_root: Path
    artifact_paths: tuple[Path, ...] = ()
    full_metrics: MetricMap | None = None
    hard_subset_metrics: MetricMap | None = None


__all__ = ["B0Manifest"]
