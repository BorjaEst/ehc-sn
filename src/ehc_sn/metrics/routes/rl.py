"""Routing tables for HRM v2 value-control (Reinforcement Learning).

These routes map metric keys to dotted attribute paths on
:class:`~ehc_sn.metrics.step_metrics.StepMetrics`, which is the step-metrics
object produced by :class:`~ehc_sn.objectives.hybrid_rl.HybridRLObjective` via
:meth:`~ehc_sn.objectives.hybrid_rl.HybridRLObjective.compute_step`.

The hybrid RL path is a **learner-owned batch-loss path**, not a rollout-scoring
objective.  These routes are consumed by
:func:`~ehc_sn.metrics.rollout.update_metric_collection_from_evaluated_chunk`
for both training and validation (where the scorer is
:class:`~ehc_sn.training.actor_critic.ZeroBootstrapActorCriticValidationScorer`).
"""

from ehc_sn.metrics.adapter import Route
from ehc_sn.metrics.keys import (
    LOSS_TOKEN,
    RL_LOSS_Q_VALUE,
    RL_LOSS_STATE_VALUE,
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
RL_STEP_ROUTES: tuple[Route, ...] = (
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
        "loss/state_value",
        *extra_ratio_paths(RL_LOSS_STATE_VALUE),
    ),
    Route(
        "loss/q_value",
        *extra_ratio_paths(RL_LOSS_Q_VALUE),
    ),
)

# =============================================================================
RL_EPISODE_ROUTES: tuple[Route, ...] = _with_namespace(
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
__all__ = ["RL_EPISODE_ROUTES", "RL_STEP_ROUTES"]
