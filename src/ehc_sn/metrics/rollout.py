"""Rollout metric folding helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from torchmetrics import MetricCollection

from ehc_sn.metrics.adapter import Route, update_metrics_from_step
from ehc_sn.rollouts.materialization import EvaluatedChunk, ObservedStep


# =============================================================================
def update_metric_collection_from_observed_step(  # ---------------------------
    collection: MetricCollection,
    observed: ObservedStep,
    routes: Sequence[Route],
) -> None:
    """Fold an observed-step metrics payload into a collection."""
    update_metrics_from_step(collection, observed.outputs.metrics, routes)


# =============================================================================
def update_metric_collection_from_evaluated_chunk(  # -------------------------
    collection: MetricCollection,
    evaluated: EvaluatedChunk,
    routes: Sequence[Route],
) -> None:
    """Fold all scored-step metrics from an evaluated chunk into a collection."""
    for step in evaluated.steps:
        update_metric_collection_from_observed_step(collection, step, routes)


# =============================================================================
def make_observed_step_metric_observer(  # ------------------------------------
    collection: MetricCollection,
    routes: Sequence[Route],
) -> Callable[[ObservedStep], None]:
    """Return an observer that folds observed-step metrics into a collection."""

    def _observer(step: ObservedStep) -> None:
        update_metric_collection_from_observed_step(collection, step, routes)

    return _observer


# =============================================================================
__all__ = [
    "make_observed_step_metric_observer",
    "update_metric_collection_from_evaluated_chunk",
    "update_metric_collection_from_observed_step",
]
