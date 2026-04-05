"""B0-specific metric helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from ehc_sn.benchmarks._capabilities import BatchPrediction

MetricMap = dict[str, float]


@dataclass(frozen=True)
class B0SampleResult:
    """Per-sample result record for B0 evaluation."""

    sample_id: str
    token_accuracy: float
    exact_match: float
    steps: int
    halted: bool
    is_hard_subset: bool

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the result."""
        return asdict(self)


def build_sample_result(*, sample_id: str, prediction: BatchPrediction, is_hard_subset: bool) -> B0SampleResult:
    """Build one per-sample B0 result from a batch prediction."""
    if prediction.targets is None:
        raise ValueError("B0 metrics require prediction.targets to be populated.")

    predictions = prediction.predictions.detach().to(dtype=torch.int64)
    targets = prediction.targets.detach().to(dtype=torch.int64)
    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction and target shapes must match, got {predictions.shape} vs {targets.shape}.")

    return B0SampleResult(
        sample_id=sample_id,
        token_accuracy=float((predictions == targets).float().mean().item()),
        exact_match=float(torch.equal(predictions, targets)),
        steps=int(prediction.steps or 0),
        halted=bool(prediction.halted),
        is_hard_subset=is_hard_subset,
    )


def summarize_results(results: list[B0SampleResult]) -> MetricMap:
    """Aggregate B0 sample results into benchmark metrics."""
    if not results:
        return {
            "n_samples": 0.0,
            "exact_match_rate": 0.0,
            "mean_token_accuracy": 0.0,
            "mean_steps": 0.0,
            "halt_rate": 0.0,
        }

    count = float(len(results))
    return {
        "n_samples": count,
        "exact_match_rate": sum(result.exact_match for result in results) / count,
        "mean_token_accuracy": sum(result.token_accuracy for result in results) / count,
        "mean_steps": sum(float(result.steps) for result in results) / count,
        "halt_rate": sum(float(result.halted) for result in results) / count,
    }


__all__ = ["B0SampleResult", "MetricMap", "build_sample_result", "summarize_results"]
