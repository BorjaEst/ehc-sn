"""Routing table for HRM v2 RL (Reinforcement Learning) training paradigm.

These routes map metric keys to dotted attribute paths on
:class:`~ehc_sn.training.types.StepMetrics`, which is the step-metrics object
produced by :class:`~ehc_sn.training.rl_head.RLLossHead`.
"""

from ehc_sn.metrics.adapter import Route

RL_ROUTES: tuple[Route, ...] = (
    # key                  numerator path                   denominator path
    Route("all/accuracy",     "halted.accuracy_sum",        "halted.halted_count"),   # fmt: skip
    Route("halted/rate",      "halted.halted_count",        "halted.eligible_count"), # fmt: skip
    Route("halted/avg_steps", "halted.steps_sum",           "halted.halted_count"),   # fmt: skip
    Route("tokens/accuracy",  "tokens.token_correct_sum",   "tokens.token_count_sum"),# fmt: skip
    Route("loss/lm",          "loss.lm_loss_sum",           "loss.batch_count"),      # fmt: skip
    Route("loss/q_halt",      "loss.q_halt_loss_sum",       "loss.batch_count"),      # fmt: skip
    Route("loss/actor",       "loss.actor_loss_sum",        "loss.batch_count"),      # fmt: skip
    Route("loss/critic",      "loss.critic_loss_sum",       "loss.batch_count"),      # fmt: skip
    Route("loss/entropy",     "loss.entropy_loss_sum",      "loss.batch_count"),      # fmt: skip
    Route("loss/q_value",     "loss.q_value_loss_sum",      "loss.batch_count"),      # fmt: skip
)

__all__ = ["RL_ROUTES"]
