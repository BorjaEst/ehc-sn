"""Capability contracts for batch prediction benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor


@dataclass(frozen=True)
class BatchPrediction:
    """Generic batch-prediction result consumed by benchmark evaluators."""

    predictions: Tensor
    targets: Tensor | None = None
    logits: Tensor | None = None
    steps: int | None = None
    halted: bool | None = None


class BatchPredicts(Protocol):
    """Predict over a batch-shaped benchmark payload."""

    def predict_batch(self, batch: Any) -> BatchPrediction:
        """Return model outputs for the provided batch."""


__all__ = ["BatchPrediction", "BatchPredicts"]
