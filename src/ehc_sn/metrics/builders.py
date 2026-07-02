"""Metrics collection builders."""

from __future__ import annotations

from collections.abc import Sequence

from torchmetrics import MetricCollection

from ehc_sn.metrics.adapter import Route
from ehc_sn.metrics.torchmetrics import LastRatioMetric, RatioMetric


# =============================================================================
def build_train_metrics(  # ---------------------------------------------------
    routes: Sequence[Route],
) -> MetricCollection:
    """Build a MetricCollection for training metrics (non-accumulated, direct step ratios).

    Args:
        routes: Paradigm-specific routing table (e.g. :data:`~ehp_sn.metrics.routes.ACT_STEP_ROUTES`,
            :data:`~ehp_sn.metrics.routes.ACT_EPISODE_ROUTES`,
            :data:`~ehp_sn.metrics.routes.RL_STEP_ROUTES`, or
            :data:`~ehp_sn.metrics.routes.RL_EPISODE_ROUTES`). The metric keys are derived from
            ``route.key`` for each entry.

    Returns:
        A :class:`~torchmetrics.MetricCollection` of :class:`LastRatioMetric` instances,
        keyed by the route keys.  Pass the returned collection to
        :func:`update_metrics_from_step` with the same *routes*.
    """
    keys = [r.key for r in routes]
    return MetricCollection(
        {k: LastRatioMetric() for k in keys}, compute_groups=[keys]
    )


# =============================================================================
def build_val_metrics(  # -----------------------------------------------------
    routes: Sequence[Route],
) -> MetricCollection:
    """Build a MetricCollection for validation metrics (accumulated over the epoch).

    Args:
        routes: Paradigm-specific routing table (e.g. :data:`~ehp_sn.metrics.routes.ACT_STEP_ROUTES`,
            :data:`~ehp_sn.metrics.routes.ACT_EPISODE_ROUTES`,
            :data:`~ehp_sn.metrics.routes.RL_STEP_ROUTES`, or
            :data:`~ehp_sn.metrics.routes.RL_EPISODE_ROUTES`). The metric keys are derived from
            ``route.key`` for each entry.

    Returns:
        A :class:`~torchmetrics.MetricCollection` of :class:`RatioMetric` instances,
        keyed by the route keys.  Pass the returned collection to
        :func:`update_metrics_from_step` with the same *routes*.
    """
    keys = [r.key for r in routes]
    return MetricCollection(
        {k: RatioMetric() for k in keys}, compute_groups=[keys]
    )


# =============================================================================
__all__ = ["build_train_metrics", "build_val_metrics"]
