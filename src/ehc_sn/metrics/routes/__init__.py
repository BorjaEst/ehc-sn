"""Paradigm-specific routing tables for :func:`~ehc_sn.metrics.adapter.update_metrics_from_step`.

Each module in this package defines a tuple of :class:`~ehc_sn.metrics.adapter.Route` entries
that map base metric keys to dotted attribute paths on the paradigm's step-metrics object.

Import the appropriate table and pass it to :func:`~ehc_sn.metrics.update_metrics_from_step`
and :func:`~ehc_sn.metrics.build_train_metrics` / :func:`~ehc_sn.metrics.build_val_metrics`.
"""

from ehc_sn.metrics.routes.act import ACT_ROUTES
from ehc_sn.metrics.routes.rl import RL_ROUTES

__all__ = ["ACT_ROUTES", "RL_ROUTES"]
