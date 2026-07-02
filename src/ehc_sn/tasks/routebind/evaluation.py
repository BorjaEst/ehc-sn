"""Routebind field evaluation — metrics, score report, and route extraction stubs.

Primary metric: ``balanced_trajectory_field_error`` (on-route vs off-route
separate normalization).
Behavioral metrics: ``valid_semantic_spatial_route_rate``,
``semantic_spatial_path_cost_ratio``.
Secondary and diagnostic metrics defined in metric specs.

Route extraction from the predicted trajectory field (canonical greedy decoder)
is a stub — raises ``NotImplementedError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ehc_sn.evaluation.contracts import EvaluationCaseResult
from ehc_sn.metrics.spec import MetricSpec, TaskScoringSpec
from ehc_sn.tasks.routebind.decoding import (
    extract_route_from_trajectory_field,
    extract_waypoints_from_field,
)
from ehc_sn.types import Batch

from .contracts import RoutebindTargets

# =============================================================================
# Metric specs
# =============================================================================

ROUTEBIND_METRIC_SPECS: list[MetricSpec] = [
    # Primary representational metric
    MetricSpec(
        name="balanced_trajectory_field_error",
        label="Balanced trajectory MSE",
        higher_is_better=False,
        unit="mse",
        scope="task",
        benchmark_eligible=True,
        description="Mean of on-route and off-route MSE over the "
        "trajectory field (separate normalization prevents trivial "
        "all-zero advantage).",
    ),
    # Primary behavioral metric
    MetricSpec(
        name="valid_semantic_spatial_route_rate",
        label="Valid semantic-spatial route rate",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=True,
        description="Fraction of samples for which the extracted route "
        "satisfies both spatial validity and semantic realizability checks.",
    ),
    # Primary optimality metric
    MetricSpec(
        name="semantic_spatial_path_cost_ratio",
        label="Path cost ratio",
        higher_is_better=False,
        unit="ratio",
        scope="task",
        benchmark_eligible=True,
        description="Ratio of extracted route cost to oracle optimal cost. "
        "1.0 = exact optimality.",
    ),
    # Calibration metrics
    MetricSpec(
        name="trajectory_field_mse",
        label="Trajectory MSE (full grid)",
        higher_is_better=False,
        unit="mse",
        scope="task",
        benchmark_eligible=True,
        description="Raw full-grid MSE over the trajectory field, retained "
        "for cross-task comparability.",
    ),
    MetricSpec(
        name="balanced_waypoint_field_error",
        label="Balanced waypoint MSE",
        higher_is_better=False,
        unit="mse",
        scope="task",
        benchmark_eligible=False,
        description="Balanced MSE over the waypoint field (waypoint vs "
        "non-waypoint separate normalization).",
    ),
    # Auxiliary metrics (multi-label)
    MetricSpec(
        name="next_direction_precision",
        label="Next direction precision",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=False,
        description="Multi-label precision over optimal direction mask.",
    ),
    MetricSpec(
        name="next_direction_recall",
        label="Next direction recall",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=False,
        description="Multi-label recall over optimal direction mask.",
    ),
    MetricSpec(
        name="next_direction_f1",
        label="Next direction F1",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=False,
        description="Multi-label F1 over optimal direction mask.",
    ),
    MetricSpec(
        name="next_observation_precision",
        label="Next observation precision",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=False,
        description="Multi-label precision over optimal observation mask.",
    ),
    MetricSpec(
        name="next_observation_recall",
        label="Next observation recall",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=False,
        description="Multi-label recall over optimal observation mask.",
    ),
    MetricSpec(
        name="next_observation_f1",
        label="Next observation F1",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=False,
        description="Multi-label F1 over optimal observation mask.",
    ),
]

ROUTEBIND_SCORING_SPEC: TaskScoringSpec = TaskScoringSpec(
    task_name="routebind",
    metrics={spec.name: spec for spec in ROUTEBIND_METRIC_SPECS},
    default_score="balanced_trajectory_field_error",
)

# =============================================================================
# Score report
# =============================================================================


@dataclass
class RoutebindStepScore:
    """Per-batch field prediction metrics for routebind.

    All metrics are mean-aggregated over samples.

    Attributes:
        balanced_trajectory_field_error: Balanced MSE over trajectory field.
        valid_semantic_spatial_route_rate: Fraction of valid routes.
        semantic_spatial_path_cost_ratio: Mean path cost ratio.
        trajectory_field_mse: Raw full-grid trajectory MSE.
        balanced_waypoint_field_error: Balanced MSE over waypoint field.
        next_direction_accuracy: Next-direction prediction accuracy (single-label).
        next_observation_accuracy: Next-observation prediction accuracy (single-label).
        next_direction_precision: Multi-label direction precision.
        next_direction_recall: Multi-label direction recall.
        next_direction_f1: Multi-label direction F1.
        next_observation_precision: Multi-label observation precision.
        next_observation_recall: Multi-label observation recall.
        next_observation_f1: Multi-label observation F1.
        n_samples: Number of samples in this batch.
    """

    balanced_trajectory_field_error: float = 0.0
    valid_semantic_spatial_route_rate: float = 0.0
    semantic_spatial_path_cost_ratio: float = 0.0
    trajectory_field_mse: float = 0.0
    balanced_waypoint_field_error: float = 0.0
    # Multi-label auxiliary metrics
    next_direction_precision: float = 0.0
    next_direction_recall: float = 0.0
    next_direction_f1: float = 0.0
    next_observation_precision: float = 0.0
    next_observation_recall: float = 0.0
    next_observation_f1: float = 0.0
    n_samples: int = 0

    def __str__(self) -> str:
        parts = [
            f"balanced_traj_field_err={self.balanced_trajectory_field_error:.6f}",
            f"valid_route_rate={self.valid_semantic_spatial_route_rate:.4f}",
            f"path_cost_ratio={self.semantic_spatial_path_cost_ratio:.4f}",
            f"traj_field_mse={self.trajectory_field_mse:.6f}",
            f"balanced_wp_field_err={self.balanced_waypoint_field_error:.6f}",
            f"next_dir_prec={self.next_direction_precision:.4f}",
            f"next_dir_rec={self.next_direction_recall:.4f}",
            f"next_dir_f1={self.next_direction_f1:.4f}",
            f"next_obs_prec={self.next_observation_precision:.4f}",
            f"next_obs_rec={self.next_observation_recall:.4f}",
            f"next_obs_f1={self.next_observation_f1:.4f}",
            f"samples={self.n_samples}",
        ]
        return "RoutebindStepScore(" + ", ".join(parts) + ")"

    def merge(self, other: RoutebindStepScore) -> RoutebindStepScore:
        """Merge (weighted-average) another report into this one.

        Used for aggregating per-batch scores across an epoch.
        """
        total = self.n_samples + other.n_samples
        if total == 0:
            return self
        w_self = self.n_samples / total
        w_other = other.n_samples / total
        return RoutebindStepScore(
            balanced_trajectory_field_error=self.balanced_trajectory_field_error
            * w_self
            + other.balanced_trajectory_field_error * w_other,
            valid_semantic_spatial_route_rate=self.valid_semantic_spatial_route_rate
            * w_self
            + other.valid_semantic_spatial_route_rate * w_other,
            semantic_spatial_path_cost_ratio=self.semantic_spatial_path_cost_ratio
            * w_self
            + other.semantic_spatial_path_cost_ratio * w_other,
            trajectory_field_mse=self.trajectory_field_mse * w_self
            + other.trajectory_field_mse * w_other,
            balanced_waypoint_field_error=self.balanced_waypoint_field_error
            * w_self
            + other.balanced_waypoint_field_error * w_other,
            next_direction_precision=self.next_direction_precision * w_self
            + other.next_direction_precision * w_other,
            next_direction_recall=self.next_direction_recall * w_self
            + other.next_direction_recall * w_other,
            next_direction_f1=self.next_direction_f1 * w_self
            + other.next_direction_f1 * w_other,
            next_observation_precision=self.next_observation_precision * w_self
            + other.next_observation_precision * w_other,
            next_observation_recall=self.next_observation_recall * w_self
            + other.next_observation_recall * w_other,
            next_observation_f1=self.next_observation_f1 * w_self
            + other.next_observation_f1 * w_other,
            n_samples=total,
        )


# =============================================================================
__all__ = [
    "ROUTEBIND_METRIC_SPECS",
    "ROUTEBIND_SCORING_SPEC",
    "RoutebindStepScore",
]
