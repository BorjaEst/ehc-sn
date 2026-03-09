"""Routing tables for HRM v1 ACT (Adaptive Computation Time).

These routes map metric keys to dotted attribute paths on
:class:`~ehc_sn.training.types.StepMetrics`, which is the step-metrics object
produced by :class:`~ehc_sn.heads.act.ACTLossHead`.
"""

from ehc_sn.metrics.adapter import Route
from ehc_sn.metrics.keys import ACT_LOSS_Q_CONTINUE, ACT_LOSS_Q_DONE, LOSS_LM, extra_ratio_paths


def _with_namespace(namespace: str, routes: tuple[Route, ...]) -> tuple[Route, ...]:
    """Prefix route keys with a metric namespace."""
    return tuple(Route(f"{namespace}/{route.key}", route.num_path, route.den_path) for route in routes)


ACT_STEP_ROUTES: tuple[Route, ...] = (
    # key                                  numerator path                           denominator path
    Route("all/accuracy",                "step.accuracy_sum",                    "step.evaluated_count"),         # fmt: skip
    Route("rollout/completed_rate",      "episode.completed_count",              "step.evaluated_count"),         # fmt: skip
    Route("rollout/avg_steps",           "step.steps_sum",                       "step.evaluated_count"),         # fmt: skip
    Route("tokens/accuracy",             "step_tokens.token_correct_sum",        "step_tokens.token_count_sum"),  # fmt: skip
    Route("loss/lm",                     *extra_ratio_paths(LOSS_LM)),                                            # fmt: skip
    Route("loss/q_done",                 *extra_ratio_paths(ACT_LOSS_Q_DONE)),                                    # fmt: skip
    Route("loss/q_continue",             *extra_ratio_paths(ACT_LOSS_Q_CONTINUE)),                                # fmt: skip
)

ACT_EPISODE_ROUTES: tuple[Route, ...] = _with_namespace(
    "episode",
    (
        Route("all/accuracy", "episode.accuracy_sum", "episode.completed_count"),
        Route("rollout/completed_rate", "episode.completed_count", "episode.eligible_count"),
        Route("rollout/avg_steps", "episode.steps_sum", "episode.completed_count"),
        Route("tokens/accuracy", "episode_tokens.token_correct_sum", "episode_tokens.token_count_sum"),
    ),
)

__all__ = ["ACT_EPISODE_ROUTES", "ACT_STEP_ROUTES"]
