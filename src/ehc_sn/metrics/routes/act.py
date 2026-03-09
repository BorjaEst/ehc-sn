"""Routing table for HRM v1 ACT (Adaptive Computation Time) training paradigm.

These routes map metric keys to dotted attribute paths on
:class:`~ehc_sn.training.types.StepMetrics`, which is the step-metrics object
produced by :class:`~ehc_sn.heads.act.ACTLossHead`.
"""

from ehc_sn.metrics.adapter import Route

ACT_ROUTES: tuple[Route, ...] = (
    # key                             numerator path                           denominator path
    Route("all/accuracy",           "rollout.accuracy_sum",                  "rollout.completed_count"),                # fmt: skip
    Route("rollout/completed_rate", "rollout.completed_count",               "rollout.eligible_count"),                 # fmt: skip
    Route("rollout/avg_steps",      "rollout.steps_sum",                     "rollout.completed_count"),                # fmt: skip
    Route("tokens/accuracy",        "tokens.token_correct_sum",              "tokens.token_count_sum"),                 # fmt: skip
    Route("loss/lm",                "extras.loss_lm.numerator_sum",          "extras.loss_lm.denominator_sum"),         # fmt: skip
    Route("loss/q_done",            "extras.loss_q_done.numerator_sum",      "extras.loss_q_done.denominator_sum"),     # fmt: skip
    Route("loss/q_continue",        "extras.loss_q_continue.numerator_sum",  "extras.loss_q_continue.denominator_sum"), # fmt: skip
)

__all__ = ["ACT_ROUTES"]
