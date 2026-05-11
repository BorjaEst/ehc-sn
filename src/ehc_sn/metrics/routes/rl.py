"""Routing tables for HRM v2 RL (Reinforcement Learning).

These routes map metric keys to dotted attribute paths on
:class:`~ehc_sn.training.types.StepMetrics`, which is the step-metrics object
produced by :class:`~ehc_sn.objectives.hybrid_rl.HybridRLLossHead` via
:meth:`~ehc_sn.objectives.hybrid_rl.HybridRLLossHead.compute_step`.

The hybrid RL path is a **learner-owned batch-loss path**, not a rollout-scoring
objective.  These routes are consumed by
:func:`~ehc_sn.lightning._rollout.update_metric_collection_from_evaluated_chunk`
for both training and validation (where the scorer is
:class:`~ehc_sn.training.actor_critic.ZeroBootstrapActorCriticValidationScorer`).
"""

from ehc_sn.metrics.adapter import Route
from ehc_sn.metrics.keys import LOSS_LM, RL_LOSS_ACTOR, RL_LOSS_CRITIC, RL_LOSS_ENTROPY, RL_LOSS_Q_VALUE, extra_ratio_paths


def _with_namespace(namespace: str, routes: tuple[Route, ...]) -> tuple[Route, ...]:
    """Prefix route keys with a metric namespace."""
    return tuple(Route(f"{namespace}/{route.key}", route.num_path, route.den_path) for route in routes)


RL_STEP_ROUTES: tuple[Route, ...] = (
    # key                                  numerator path                           denominator path
    Route("all/accuracy",                "step.accuracy_sum",                    "step.evaluated_count"),         # fmt: skip
    Route("rollout/completed_rate",      "episode.completed_count",              "step.evaluated_count"),         # fmt: skip
    Route("rollout/avg_steps",           "step.steps_sum",                       "step.evaluated_count"),         # fmt: skip
    Route("tokens/accuracy",             "step_tokens.token_correct_sum",        "step_tokens.token_count_sum"),  # fmt: skip
    Route("loss/lm",                     *extra_ratio_paths(LOSS_LM)),                                            # fmt: skip
    Route("loss/actor",                  *extra_ratio_paths(RL_LOSS_ACTOR)),                                      # fmt: skip
    Route("loss/critic",                 *extra_ratio_paths(RL_LOSS_CRITIC)),                                     # fmt: skip
    Route("loss/entropy",                *extra_ratio_paths(RL_LOSS_ENTROPY)),                                    # fmt: skip
    Route("loss/q_value",                *extra_ratio_paths(RL_LOSS_Q_VALUE)),                                    # fmt: skip
)

RL_EPISODE_ROUTES: tuple[Route, ...] = _with_namespace(
    "episode",
    (
        Route("all/accuracy", "episode.accuracy_sum", "episode.completed_count"),
        Route("rollout/completed_rate", "episode.completed_count", "episode.eligible_count"),
        Route("rollout/avg_steps", "episode.steps_sum", "episode.completed_count"),
        Route("tokens/accuracy", "episode_tokens.token_correct_sum", "episode_tokens.token_count_sum"),
    ),
)

__all__ = ["RL_EPISODE_ROUTES", "RL_STEP_ROUTES"]
