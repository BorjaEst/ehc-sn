"""Central registry of ``TaskScoringSpec`` instances for all known task families.

Task modules own their ``*_SCORING_SPEC`` constant in ``evaluation.py``.
This module aggregates them in one place for cross-module discovery.

There is no independent primary-metric mapping here — every metric name
comes from a canonical task-owned ``TaskScoringSpec``.
"""

from __future__ import annotations

from ehc_sn.metrics.spec import TaskScoringSpec
from ehc_sn.tasks.arena.evaluation import ARENA_SCORING_SPEC
from ehc_sn.tasks.goaltrace.evaluation import GOALTRACE_SCORING_SPEC
from ehc_sn.tasks.mazehard.evaluation import MAZEHARD_SCORING_SPEC
from ehc_sn.tasks.routebind.evaluation import ROUTEBIND_SCORING_SPEC
from ehc_sn.tasks.seqmaze.evaluation import SEQMAZE_V1_SCORING_SPEC

TASK_SCORING_SPECS: dict[str, TaskScoringSpec] = {
    spec.task_name: spec
    for spec in [
        ARENA_SCORING_SPEC,
        GOALTRACE_SCORING_SPEC,
        MAZEHARD_SCORING_SPEC,
        ROUTEBIND_SCORING_SPEC,
        SEQMAZE_V1_SCORING_SPEC,
    ]
}
"""Mapping from task-family name to its ``TaskScoringSpec``.

This is the single aggregation point.  Each task family's evaluation module
owns its spec; this dict collects them for discovery.
"""


def scoring_spec_for_task(task_name: str) -> TaskScoringSpec:
    """Return the ``TaskScoringSpec`` for *task_name*, or raise ``KeyError``.

    Args:
        task_name: Canonical task-family identifier (e.g. ``"mazehard"``).

    Returns:
        The task's ``TaskScoringSpec``.

    Raises:
        KeyError: If *task_name* is not a known task with a registered spec.
    """
    try:
        return TASK_SCORING_SPECS[task_name]
    except KeyError:
        known = ", ".join(sorted(TASK_SCORING_SPECS))
        raise KeyError(
            f"Unknown task {task_name!r}. Known tasks: {known}."
        ) from None


def primary_metric_name_for_task(task_name: str) -> str | None:
    """Return the default primary metric name for *task_name*, or ``None``.

    Delegates to the task's ``TaskScoringSpec.default_score``.  Returns
    ``None`` for unknown tasks rather than raising.
    """
    spec = TASK_SCORING_SPECS.get(task_name)
    if spec is None:
        return None
    return spec.default_score


__all__ = [
    "TASK_SCORING_SPECS",
    "primary_metric_name_for_task",
    "scoring_spec_for_task",
]
