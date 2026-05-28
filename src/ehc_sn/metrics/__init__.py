"""Metrics and related utilities for EHC-SN."""

from ehc_sn.metrics.adapter import Route, update_metrics_from_step
from ehc_sn.metrics.builders import build_train_metrics, build_val_metrics
from ehc_sn.metrics.reducers import HiddenNormHistogram, OccupancyHistogram
from ehc_sn.metrics.renderers import (
    log_reducer_figure,
    render_hidden_norm_histogram,
    render_occupancy_histogram,
)
from ehc_sn.metrics.rollout import (
    make_observed_step_metric_observer,
    update_metric_collection_from_evaluated_chunk,
    update_metric_collection_from_observed_step,
)
from ehc_sn.metrics.routes import (
    EHC_EPISODE_ROUTES,
    EHC_PRIMARY_VAL_ROUTE_KEY,
    EHC_STEP_ROUTES,
    RL_EPISODE_ROUTES,
    RL_STEP_ROUTES,
)
from ehc_sn.metrics.torchmetrics import LastRatioMetric, RatioMetric

# =============================================================================
__all__ = [
    "HiddenNormHistogram",
    "OccupancyHistogram",
    "Route",
    "RatioMetric",
    "LastRatioMetric",
    "log_reducer_figure",
    "render_hidden_norm_histogram",
    "render_occupancy_histogram",
    "update_metrics_from_step",
    "make_observed_step_metric_observer",
    "update_metric_collection_from_evaluated_chunk",
    "update_metric_collection_from_observed_step",
    "build_train_metrics",
    "build_val_metrics",
    "EHC_EPISODE_ROUTES",
    "EHC_PRIMARY_VAL_ROUTE_KEY",
    "EHC_STEP_ROUTES",
    "RL_EPISODE_ROUTES",
    "RL_STEP_ROUTES",
]
