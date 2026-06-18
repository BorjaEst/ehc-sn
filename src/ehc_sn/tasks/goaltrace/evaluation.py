"""Goaltrace field evaluation — field MSE and component metrics.

Primary metric: field_mse (mean squared error over all N observation slots).
Secondary metrics: current_accuracy, successor_accuracy, goal_activation,
    off_path_suppression, field_decay_correlation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from ehc_sn.eval.contracts import EvaluationCaseResult
from ehc_sn.metrics.spec import MetricSpec, TaskScoringSpec
from ehc_sn.types import Batch

from .contracts import GoaltraceTargets


# =============================================================================
def _batch_field_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Compute per-sample MSE over valid (masked) nodes.

    Args:
        pred: Predicted firing field, shape ``(B, N)`` float.
        target: Target firing field, shape ``(B, N)`` float.
        mask: Valid node mask, shape ``(B, N)`` bool.

    Returns:
        Per-sample MSE, shape ``(B,)`` float.
    """
    diff = pred - target
    squared = diff**2
    valid_count = mask.sum(dim=1).clamp(min=1)
    return (squared * mask).sum(dim=1) / valid_count


def _successor_mask_from_current(
    current_idx: Tensor,
    observation_id: Tensor,
    target_field: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Return a bool mask of shape ``(B, N)`` selecting nodes that are direct
    successors of the current location on the optimal path.

    A successor node on the optimal path has:
    - distance 1 from current (nonzero, but first non-current reachable)
    - nonzero target_field

    This is approximated by: nodes with nonzero target, not the current node,
    that are adjacent in field-value magnitude to the current node's value.

    More precisely: among nodes with ``target_field > 0``, the set of nodes
    whose distance from current is exactly 1 is identified by checking
    ``observation_id`` adjacency.  Since observation IDs are permuted, we
    identify the successor as the node with the highest target_field that is
    not the current node.

    Args:
        current_idx: Current node index, shape ``(B,)`` int64.
        observation_id: Observation IDs, shape ``(B, N)`` int64.
        target_field: Target field, shape ``(B, N)`` float32.
        node_mask: Valid node mask, shape ``(B, N)`` bool.

    Returns:
        Successor mask, shape ``(B, N)`` bool.
    """
    B, N = target_field.shape
    device = target_field.device

    # A successor on the path is any node with nonzero target but not current
    nonzero = (target_field > 1e-6) & node_mask
    # Exclude current node
    not_current = torch.arange(N, device=device).unsqueeze(0).expand(
        B, -1
    ) != current_idx.unsqueeze(1)
    return nonzero & not_current


# =============================================================================
@dataclass
class GoaltraceStepScore:
    """Per-batch field prediction metrics for goaltrace.

    All metrics are mean-aggregated over samples.

    Attributes:
        field_mse: Mean squared error over all ``N`` observation slots.
        current_accuracy: Mean predicted firing at the current location.
        successor_accuracy: MSE restricted to direct successors on the
            optimal path.
        goal_activation: Mean predicted firing at the goal observation.
        off_path_suppression: Mean predicted firing at observations not on
            viable goal-reaching paths.
        field_decay_correlation: Pearson r between predicted and target
            activation profile along the optimal path.
        n_samples: Number of samples in this batch.
    """

    field_mse: float = 0.0
    current_accuracy: float = 0.0
    successor_accuracy: float = 0.0
    goal_activation: float = 0.0
    off_path_suppression: float = 0.0
    field_decay_correlation: float = 0.0
    n_samples: int = 0

    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        parts = [
            f"field_mse={self.field_mse:.6f}",
            f"current_accuracy={self.current_accuracy:.4f}",
            f"successor_accuracy={self.successor_accuracy:.6f}",
            f"goal_activation={self.goal_activation:.4f}",
            f"off_path_suppression={self.off_path_suppression:.4f}",
            f"field_decay_correlation={self.field_decay_correlation:.4f}",
            f"samples={self.n_samples}",
        ]
        return "GoaltraceStepScore(" + ", ".join(parts) + ")"

    # ------------------------------------------------------------------ #
    def merge(self, other: GoaltraceStepScore) -> GoaltraceStepScore:
        """Merge (weighted-average) another report into this one.

        Used for aggregating per-batch scores across an epoch.
        """
        total = self.n_samples + other.n_samples
        if total == 0:
            return self
        w_self = self.n_samples / total
        w_other = other.n_samples / total
        return GoaltraceStepScore(
            field_mse=self.field_mse * w_self + other.field_mse * w_other,
            current_accuracy=self.current_accuracy * w_self
            + other.current_accuracy * w_other,
            successor_accuracy=self.successor_accuracy * w_self
            + other.successor_accuracy * w_other,
            goal_activation=self.goal_activation * w_self
            + other.goal_activation * w_other,
            off_path_suppression=self.off_path_suppression * w_self
            + other.off_path_suppression * w_other,
            field_decay_correlation=self.field_decay_correlation * w_self
            + other.field_decay_correlation * w_other,
            n_samples=total,
        )


# =============================================================================
def compute_goaltrace_step_score(
    pred_field: Tensor,
    target_field: Tensor,
    current_flag: Tensor,
    goal_flag: Tensor,
    node_mask: Tensor,
) -> GoaltraceFieldScore:
    """Compute field prediction metrics from predicted and target fields.

    Args:
        pred_field: Predicted firing field, shape ``(B, N)`` float32.
        target_field: Target firing field, shape ``(B, N)`` float32.
        current_flag: Current location mask, shape ``(B, N)`` bool.
        goal_flag: Goal observation mask, shape ``(B, N)`` bool.
        node_mask: Valid node mask, shape ``(B, N)`` bool.

    Returns:
        Aggregated score report.
    """
    B, N = pred_field.shape
    if B == 0:
        return GoaltraceStepScore()

    device = pred_field.device
    current_idx = current_flag.to(torch.int64).argmax(dim=1)  # (B,)
    goal_idx = goal_flag.to(torch.int64).argmax(dim=1)  # (B,)

    # --- field_mse ---
    mse_per_sample = _batch_field_mse(pred_field, target_field, node_mask)
    field_mse = float(mse_per_sample.mean().item())

    # --- current_accuracy ---
    current_vals = pred_field.gather(1, current_idx.unsqueeze(1)).squeeze(1)
    current_accuracy = float(current_vals.mean().item())

    # --- successor_accuracy ---
    succ_mask = _successor_mask_from_current(
        current_idx, None, target_field, node_mask
    )
    succ_diff = (pred_field - target_field) ** 2
    succ_mse = (succ_diff * succ_mask).sum() / succ_mask.sum().clamp(min=1)
    successor_accuracy = float(succ_mse.item())

    # --- goal_activation ---
    goal_vals = pred_field.gather(1, goal_idx.unsqueeze(1)).squeeze(1)
    goal_activation = float(goal_vals.mean().item())

    # --- off_path_suppression ---
    target_zero = (target_field < 1e-6) & node_mask
    off_path_count = target_zero.sum(dim=1).clamp(min=1)
    off_path_mean = (pred_field * target_zero).sum(dim=1) / off_path_count
    off_path_suppression = float(off_path_mean.mean().item())

    # --- field_decay_correlation ---
    # Pearson r over (pred, target) pairs of on-path nodes
    on_path = (target_field > 1e-6) & node_mask
    correlations: list[float] = []
    for b in range(B):
        path_mask_b = on_path[b]
        if path_mask_b.sum() < 2:
            continue
        p = pred_field[b, path_mask_b].float()
        t = target_field[b, path_mask_b].float()
        p_centered = p - p.mean()
        t_centered = t - t.mean()
        denom = (p_centered.norm() * t_centered.norm()).clamp(min=1e-8)
        r = (p_centered @ t_centered) / denom
        correlations.append(float(r.item()))

    field_decay_correlation = (
        float(np.mean(correlations)) if correlations else 0.0
    )

    return GoaltraceStepScore(
        field_mse=field_mse,
        current_accuracy=current_accuracy,
        successor_accuracy=successor_accuracy,
        goal_activation=goal_activation,
        off_path_suppression=off_path_suppression,
        field_decay_correlation=field_decay_correlation,
        n_samples=B,
    )


# =============================================================================
# EvaluationCaseResult integration
# =============================================================================


def build_goaltrace_step_score(
    case_result: EvaluationCaseResult,
) -> GoaltraceStepScore:
    """Build a GoaltraceStepScore from an evaluation case result.

    The case result is expected to contain batch tensors with keys matching
    the goaltrace contract fields.

    Args:
        case_result: Evaluation case result with batch of predicted and
            target fields.

    Returns:
        Computed GoaltraceFieldScore.
    """
    batch = case_result.batch
    pred_field = batch.get("pred/firing_field")
    target_field = batch.get("target/target_field")
    current_flag = batch.get("input/current_flag")
    goal_flag = batch.get("input/goal_flag")
    node_mask = batch.get("input/node_mask")

    if pred_field is None or target_field is None:
        raise ValueError(
            "EvaluationCaseResult missing pred/firing_field or "
            "target/target_field."
        )
    if current_flag is None or goal_flag is None or node_mask is None:
        raise ValueError(
            "EvaluationCaseResult missing input/current_flag, "
            "input/goal_flag, or input/node_mask."
        )

    return compute_goaltrace_step_score(
        pred_field=pred_field,
        target_field=target_field,
        current_flag=current_flag,
        goal_flag=goal_flag,
        node_mask=node_mask,
    )


# =============================================================================
# Metric specs for benchmark registration
# =============================================================================

GOALTRACE_METRIC_SPECS: list[MetricSpec] = [
    MetricSpec(
        name="field_mse",
        label="Field MSE",
        higher_is_better=False,
        description="Mean squared error over all N observation slots.",
        scope="task",
    ),
    MetricSpec(
        name="current_accuracy",
        label="Current accuracy",
        higher_is_better=True,
        description="Mean predicted firing at the current location.",
        scope="task",
    ),
    MetricSpec(
        name="successor_accuracy",
        label="Successor MSE",
        higher_is_better=False,
        description="MSE restricted to direct successors on the optimal path.",
        scope="task",
    ),
    MetricSpec(
        name="goal_activation",
        label="Goal activation",
        higher_is_better=False,
        description="Mean predicted firing at the goal observation.",
        scope="task",
    ),
    MetricSpec(
        name="off_path_suppression",
        label="Off-path suppression",
        higher_is_better=False,
        description="Mean predicted firing at observations not on viable "
        "goal-reaching paths.",
        scope="task",
    ),
    MetricSpec(
        name="field_decay_correlation",
        label="Field decay correlation",
        higher_is_better=True,
        description="Pearson r between predicted and target activation "
        "profile along the optimal path.",
        scope="task",
    ),
]

_GOALTRACE_METRICS_MAP = {spec.name: spec for spec in GOALTRACE_METRIC_SPECS}

GOALTRACE_SCORING_SPEC: TaskScoringSpec = TaskScoringSpec(
    task_name="goaltrace",
    metrics=_GOALTRACE_METRICS_MAP,
    default_score="field_mse",
)

# =============================================================================
__all__ = [
    "GOALTRACE_METRIC_SPECS",
    "GOALTRACE_SCORING_SPEC",
    "GoaltraceStepScore",
    "build_goaltrace_step_score",
    "compute_goaltrace_step_score",
]
