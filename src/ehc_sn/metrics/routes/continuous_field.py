"""Routing tables for Continuous Field objectives (Goaltrace, Prospect).

These routes map metric keys to dotted attribute paths on
:class:`~ehc_sn.metrics.step_metrics.StepMetrics`, which is the step-metrics
object produced by :class:`~ehc_sn.objectives.continuous_field.ContinuousFieldObjective`.
"""

from ehc_sn.metrics.adapter import Route
from ehc_sn.metrics.keys import extra_ratio_paths
from ehc_sn.metrics.routes.act import _with_namespace

# =============================================================================
CONTINUOUS_FIELD_STEP_ROUTES: tuple[Route, ...] = (
    Route(
        key="all/accuracy",
        num_path="step.accuracy_sum",
        den_path="step.evaluated_count",
    ),
    Route(
        key="rollout/completed_rate",
        num_path="episode.completed_count",
        den_path="step.evaluated_count",
    ),
    Route(
        key="rollout/avg_steps",
        num_path="step.steps_sum",
        den_path="step.evaluated_count",
    ),
    Route(
        "loss/field_mse",
        *extra_ratio_paths("field_mse"),
    ),
    Route(
        "accuracy/q_done",
        *extra_ratio_paths("q_done_accuracy"),
    ),
)


# =============================================================================
CONTINUOUS_FIELD_EPISODE_ROUTES: tuple[Route, ...] = _with_namespace(
    "episode",
    (
        Route(
            key="all/accuracy",
            num_path="episode.accuracy_sum",
            den_path="episode.completed_count",
        ),
        Route(
            key="rollout/completed_rate",
            num_path="episode.completed_count",
            den_path="episode.eligible_count",
        ),
        Route(
            key="rollout/avg_steps",
            num_path="episode.steps_sum",
            den_path="episode.completed_count",
        ),
        Route(
            "loss/field_mse",
            *extra_ratio_paths("field_mse"),
        ),
        Route(
            "accuracy/q_done",
            *extra_ratio_paths("q_done_accuracy"),
        ),
    ),
)
