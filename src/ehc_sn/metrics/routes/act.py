"""Routing table for HRM v1 ACT (Adaptive Computation Time) training paradigm.

These routes map metric keys to dotted attribute paths on
:class:`~ehc_sn.training.types.StepMetrics`, which is the step-metrics object
produced by :class:`~ehc_sn.training.act_head.ACTLossHead`.
"""

from ehc_sn.metrics.adapter import Route

ACT_ROUTES: tuple[Route, ...] = (
    # key                  numerator path                   denominator path
    Route("all/accuracy",     "halted.accuracy_sum",        "halted.halted_count"),   # fmt: skip
    Route("halted/rate",      "halted.halted_count",        "halted.eligible_count"), # fmt: skip
    Route("halted/avg_steps", "halted.steps_sum",           "halted.halted_count"),   # fmt: skip
    Route("tokens/accuracy",  "tokens.token_correct_sum",   "tokens.token_count_sum"),# fmt: skip
    Route("loss/lm",          "loss.lm_loss_sum",           "loss.batch_count"),      # fmt: skip
    Route("loss/q_halt",      "loss.q_halt_loss_sum",       "loss.batch_count"),      # fmt: skip
    Route("loss/q_continue",  "loss.q_continue_loss_sum",   "loss.batch_count"),      # fmt: skip
)

__all__ = ["ACT_ROUTES"]
