"""M0-specific metric helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MetricMap = dict[str, float]


@dataclass(frozen=True)
class M0TargetResult:
    """Per-layout, per-target result record for M0 evaluation."""

    layout_id: str
    target_slot: int
    localization_accuracy: float
    exposure_recall: float
    probe_recall: float
    write_read_consistency: float
    interference_retention: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the result."""
        return asdict(self)


def summarize_results(results: list[M0TargetResult]) -> MetricMap:
    """Aggregate M0 target results into benchmark metrics."""
    if not results:
        return {
            "n_target_trials": 0.0,
            "mean_localization_accuracy": 0.0,
            "cue_location_recall_rate": 0.0,
            "exposure_recall_rate": 0.0,
            "probe_recall_rate": 0.0,
            "write_read_consistency_rate": 0.0,
            "interference_retention_rate": 0.0,
            "n_interference_trials": 0.0,
        }

    count = float(len(results))
    interference_values = [result.interference_retention for result in results if result.interference_retention is not None]
    cue_recall = [0.5 * (result.exposure_recall + result.probe_recall) for result in results]
    return {
        "n_target_trials": count,
        "mean_localization_accuracy": sum(result.localization_accuracy for result in results) / count,
        "cue_location_recall_rate": sum(cue_recall) / count,
        "exposure_recall_rate": sum(result.exposure_recall for result in results) / count,
        "probe_recall_rate": sum(result.probe_recall for result in results) / count,
        "write_read_consistency_rate": sum(result.write_read_consistency for result in results) / count,
        "interference_retention_rate": (sum(interference_values) / float(len(interference_values)) if interference_values else 0.0),
        "n_interference_trials": float(len(interference_values)),
    }


__all__ = ["M0TargetResult", "MetricMap", "summarize_results"]
