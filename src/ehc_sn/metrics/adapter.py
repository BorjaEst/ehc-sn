"""Mapping helpers for TorchMetrics collections."""

from __future__ import annotations

from torchmetrics import MetricCollection

from ehc_sn.metrics.types import StepMetrics

__all__ = ["update_metrics_from_step"]


def update_metrics_from_step(  # ------------------------------------------------------------------
    collection: MetricCollection, step: StepMetrics,
) -> None:  # fmt: skip
    """Update a metrics collection from aggregated step metrics."""
    # FIXME: This is a bit hacky
