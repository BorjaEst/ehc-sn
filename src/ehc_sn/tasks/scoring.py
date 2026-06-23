"""Central registry mapping task families to their canonical primary metric.

This module is the single access point for ``eval/`` and ``reporting/`` to
determine which metric is the headline benchmark score for a task.

Task modules define their own ``*_PRIMARY_METRIC_NAME`` constants in
``evaluation.py``.  This registry aggregates them in one place to avoid
fragmented lookups across the codebase.
"""

from __future__ import annotations

from ehc_sn.tasks.arena.evaluation import ARENA_PRIMARY_METRIC_NAME
from ehc_sn.tasks.mazehard.evaluation import MAZEHARD_PRIMARY_METRIC_NAME

PRIMARY_METRIC_BY_TASK: dict[str, str] = {
    "arena": ARENA_PRIMARY_METRIC_NAME,
    "mazehard": MAZEHARD_PRIMARY_METRIC_NAME,
}
"""Mapping from task-family name to its canonical primary benchmark metric.

Only tasks that have an implemented evaluation surface and a defined primary
metric appear here.  Unimplemented tasks (e.g. ``cue_recall``) are absent.
"""


def primary_metric_name_for_task(task_name: str) -> str | None:
    """Return the canonical primary metric name for *task_name*, or ``None``.

    Args:
        task_name: Canonical task-family identifier (e.g. ``"mazehard"``).

    Returns:
        The primary metric string (e.g. ``"token_accuracy"``) or ``None`` if
        the task is recognised but has no defined primary metric.

    Raises:
        ValueError: If *task_name* is not in ``KNOWN_TASKS``.
    """
    if task_name not in KNOWN_TASKS:
        known = ", ".join(sorted(KNOWN_TASKS))
        raise ValueError(f"Unknown task {task_name!r}. Known tasks: {known}.")
    return PRIMARY_METRIC_BY_TASK.get(task_name)


__all__ = [
    "PRIMARY_METRIC_BY_TASK",
    "primary_metric_name_for_task",
]
