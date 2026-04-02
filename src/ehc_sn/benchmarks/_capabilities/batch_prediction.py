"""Capability contracts for batch prediction benchmarks."""

from __future__ import annotations

from typing import Any, Protocol


class BatchPredicts(Protocol):
    """Predict over a batch-shaped benchmark payload."""

    def predict_batch(self, batch: Any) -> Any:
        """Return model outputs for the provided batch."""


__all__ = ["BatchPredicts"]
