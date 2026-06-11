"""Metric specification types: catalog metadata and benchmark validation.

``MetricSpec`` and ``TaskScoringSpec`` are the canonical types for
describing which metrics a task owns, which one is the default score, and
which metrics are eligible as benchmark primary metrics.

This module has zero dependencies on other ``ehc_sn`` packages — it can be
imported freely without circular-import risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

MetricScope = Literal["task", "diagnostic"]


@dataclass(frozen=True)
class MetricSpec:
    """Immutable metadata for one task-level or diagnostic metric.

    Attributes:
        name: Canonical metric name (e.g. ``"accuracy_revisit"``).
        label: Short human-readable label (e.g. ``"Revisit accuracy"``).
        higher_is_better: Whether higher values are better.
        unit: Physical unit string (e.g. ``"proportion"``, ``"count"``).
        description: Optional longer description.
        scope: ``"task"`` (model-family-neutral) or ``"diagnostic"``
            (family-specific supplementary).
        benchmark_eligible: Whether this metric may be used as a benchmark
            track primary metric.  Defaults to ``True`` for task-level,
            ``False`` for diagnostic.
    """

    name: str
    label: str
    higher_is_better: bool
    unit: str | None = None
    description: str | None = None
    scope: MetricScope = "task"
    benchmark_eligible: bool | None = None

    def __post_init__(self) -> None:
        if self.benchmark_eligible is None:
            object.__setattr__(self, "benchmark_eligible", self.scope == "task")


@dataclass(frozen=True)
class TaskScoringSpec:
    """Metric catalog and default score for one task family.

    Attributes:
        task_name: Canonical task-family identifier (e.g. ``"arena"``).
        metrics: Mapping from metric name to ``MetricSpec``.
        default_score: One metric name from ``metrics`` used as the
            default diagnostic score for generic task evaluation.
    """

    task_name: str
    metrics: Mapping[str, MetricSpec]
    default_score: str

    def require_metric(self, name: str) -> MetricSpec:
        """Return the ``MetricSpec`` for *name*, or raise ``KeyError``."""
        try:
            return self.metrics[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.metrics))
            raise KeyError(
                f"Unknown metric {name!r} for task {self.task_name!r}. "
                f"Known metrics: {known}"
            ) from exc

    def require_benchmark_metric(self, name: str) -> MetricSpec:
        """Return the ``MetricSpec`` for *name* if benchmark-eligible.

        Raises ``KeyError`` if the metric is unknown, ``ValueError`` if
        it exists but ``benchmark_eligible`` is ``False``.
        """
        spec = self.require_metric(name)
        if not spec.benchmark_eligible:
            raise ValueError(
                f"Metric {name!r} for task {self.task_name!r} is not "
                "benchmark eligible. Use a task-level metric or introduce "
                "an explicit recipe escape hatch."
            )
        return spec


__all__ = [
    "MetricScope",
    "MetricSpec",
    "TaskScoringSpec",
]
