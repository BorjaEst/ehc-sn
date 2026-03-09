"""Routing table for HRM v2 RL (Reinforcement Learning) training paradigm.

These routes map metric keys to dotted attribute paths on
:class:`~ehc_sn.training.types.StepMetrics`, which is the step-metrics object
produced by :class:`~ehc_sn.heads.rl.RLLossHead`.
"""

from ehc_sn.metrics.adapter import Route

RL_ROUTES: tuple[Route, ...] = (
    # key                              numerator path                          denominator path
    Route("all/accuracy",            "rollout.accuracy_sum",                 "rollout.completed_count"),                 # fmt: skip
    Route("rollout/completed_rate",  "rollout.completed_count",              "rollout.eligible_count"),                  # fmt: skip
    Route("rollout/avg_steps",       "rollout.steps_sum",                    "rollout.completed_count"),                 # fmt: skip
    Route("tokens/accuracy",         "tokens.token_correct_sum",             "tokens.token_count_sum"),                  # fmt: skip
    Route("loss/lm",                 "extras.loss_lm.numerator_sum",         "extras.loss_lm.denominator_sum"),          # fmt: skip
    Route("loss/actor",              "extras.loss_actor.numerator_sum",      "extras.loss_actor.denominator_sum"),       # fmt: skip
    Route("loss/critic",             "extras.loss_critic.numerator_sum",     "extras.loss_critic.denominator_sum"),      # fmt: skip
    Route("loss/entropy",            "extras.loss_entropy.numerator_sum",    "extras.loss_entropy.denominator_sum"),     # fmt: skip
    Route("loss/q_value",            "extras.loss_q_value.numerator_sum",    "extras.loss_q_value.denominator_sum"),     # fmt: skip
)

__all__ = ["RL_ROUTES"]
