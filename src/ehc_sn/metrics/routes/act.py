"""Routing tables for HRM v1 ACT (Adaptive Computation Time).

These routes map metric keys to dotted attribute paths on
:class:`~ehp_sn.metrics.step_metrics.StepMetrics`, which is the step-metrics
object produced by :class:`~ehp_sn.objectives.composites.act.ACTSupervisedScorer`.
"""

from ehc_sn.metrics.adapter import Route
from ehc_sn.metrics.keys import (
    ACT_LOSS_Q_CONTINUE,
    ACT_LOSS_Q_DONE,
    LOSS_TOKEN,
    extra_ratio_paths,
)


# =============================================================================
def _with_namespace(  # -------------------------------------------------------
    namespace: str, routes: tuple[Route, ...]
) -> tuple[Route, ...]:
    """Prefix route keys with a metric namespace."""
    return tuple(
        Route(f"{namespace}/{route.key}", route.num_path, route.den_path)
        for route in routes
    )


# =============================================================================
ACT_STEP_ROUTES: tuple[Route, ...] = (
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
        key="tokens/accuracy",
        num_path="step_tokens.token_correct_sum",
        den_path="step_tokens.token_count_sum",
    ),
    Route(
        "loss/token",
        *extra_ratio_paths(LOSS_TOKEN),
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
ACT_EPISODE_ROUTES: tuple[Route, ...] = _with_namespace(
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
            key="tokens/accuracy",
            num_path="episode_tokens.token_correct_sum",
            den_path="episode_tokens.token_count_sum",
        ),
    ),
)

# =============================================================================
__all__ = ["ACT_EPISODE_ROUTES", "ACT_STEP_ROUTES"]
