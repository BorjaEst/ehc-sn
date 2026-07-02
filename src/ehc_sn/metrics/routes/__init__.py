"""Paradigm-specific routing tables for :func:`~ehp_sn.metrics.adapter.update_metrics_from_step`.

Each module in this package defines a tuple of :class:`~ehp_sn.metrics.adapter.Route` entries
that map base metric keys to dotted attribute paths on the paradigm's step-metrics object.

Import the appropriate table and pass it to :func:`~ehp_sn.metrics.update_metrics_from_step`
and :func:`~ehp_sn.metrics.build_train_metrics` / :func:`~ehp_sn.metrics.build_val_metrics`.
"""

from ehc_sn.metrics.routes.act import ACT_EPISODE_ROUTES, ACT_STEP_ROUTES
from ehc_sn.metrics.routes.rl import RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.metrics.routes.tem import (
    TEM_EPISODE_ROUTES,
    TEM_PRIMARY_VAL_ROUTE_KEY,
    TEM_STEP_ROUTES,
)

# =============================================================================
__all__ = [
    "ACT_EPISODE_ROUTES",
    "ACT_STEP_ROUTES",
    "RL_EPISODE_ROUTES",
    "RL_STEP_ROUTES",
    "TEM_EPISODE_ROUTES",
    "TEM_PRIMARY_VAL_ROUTE_KEY",
    "TEM_STEP_ROUTES",
]
