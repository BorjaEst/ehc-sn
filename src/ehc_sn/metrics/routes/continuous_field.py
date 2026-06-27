"""Routing tables for Continuous Field objectives (Goaltrace, Prospect).

These routes map metric keys to dotted attribute paths on
:class:`~ehc_sn.metrics.step_metrics.StepMetrics`, which is the step-metrics
object produced by :class:`~ehc_sn.objectives.combine.ACTSupervisedScorer` (field modality).

Route categories
----------------
[deliberation]
    Populated from StepMetrics aggregate fields (step.accuracy_sum,
    episode.completed_count, etc.).  Zero in ``single_step`` mode;
    filled from ACT carry data when deliberation is enabled.

[extras]
    Populated from StepMetrics.extras dict, which the objective fills
    from its loss computation.  Active in all training modes.
"""

from ehc_sn.metrics.adapter import Route
from ehc_sn.metrics.keys import (
    ACT_LOSS_Q_CONTINUE,
    ACT_LOSS_Q_DONE,
    extra_ratio_paths,
)
from ehc_sn.metrics.routes.act import _with_namespace

# =============================================================================
# [deliberation] Fraction of eligible slots that completed the episode.
# Zero in single_step mode.
# [deliberation] Mean deliberation steps per slot.  Zero in single_step.
# [extras] MSE between predicted and target firing field.  Active always.
# [extras] Q(done) classifier accuracy.  Active always.
# =============================================================================
CONTINUOUS_FIELD_STEP_ROUTES: tuple[Route, ...] = (
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
    Route(
        "loss/q_done",
        *extra_ratio_paths(ACT_LOSS_Q_DONE),
    ),
    Route(
        "loss/q_continue",
        *extra_ratio_paths(ACT_LOSS_Q_CONTINUE),
    ),
)


# =============================================================================
CONTINUOUS_FIELD_EPISODE_ROUTES: tuple[Route, ...] = _with_namespace(
    "episode",
    (
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
