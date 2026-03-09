"""Routing table for HRM v2 RL (Reinforcement Learning) training paradigm.

These routes map metric keys to dotted attribute paths on
:class:`~ehc_sn.training.types.StepMetrics`, which is the step-metrics object
produced by :class:`~ehc_sn.heads.rl.RLLossHead`.
"""

from ehc_sn.metrics.adapter import Route
from ehc_sn.metrics.keys import LOSS_LM, RL_LOSS_ACTOR, RL_LOSS_CRITIC, RL_LOSS_ENTROPY, RL_LOSS_Q_VALUE, extra_ratio_paths  # fmt: skip

RL_ROUTES: tuple[Route, ...] = (
    # key                              numerator path                          denominator path
    Route("all/accuracy",            "rollout.accuracy_sum",                 "rollout.completed_count"),  # fmt: skip
    Route("rollout/completed_rate",  "rollout.completed_count",              "rollout.eligible_count"),   # fmt: skip
    Route("rollout/avg_steps",       "rollout.steps_sum",                    "rollout.completed_count"),  # fmt: skip
    Route("tokens/accuracy",         "tokens.token_correct_sum",             "tokens.token_count_sum"),   # fmt: skip
    Route("loss/lm",                 *extra_ratio_paths(LOSS_LM)),                                        # fmt: skip
    Route("loss/actor",              *extra_ratio_paths(RL_LOSS_ACTOR)),                                  # fmt: skip
    Route("loss/critic",             *extra_ratio_paths(RL_LOSS_CRITIC)),                                 # fmt: skip
    Route("loss/entropy",            *extra_ratio_paths(RL_LOSS_ENTROPY)),                                # fmt: skip
    Route("loss/q_value",            *extra_ratio_paths(RL_LOSS_Q_VALUE)),                                # fmt: skip
)

__all__ = ["RL_ROUTES"]
