"""Trace field vocabulary and per-paradigm TraceSpec constructors.

Mirrors :mod:`ehc_sn.metrics.signals` (scalar diagnostic names) and
:mod:`ehc_sn.metrics.routes` (metric routing tables) for the temporal rollout
data path.  All :class:`~ehc_sn.rollouts.collect.TraceField` definitions live
here so model files contain *no* trace wiring.

Usage
-----
    from ehc_sn.metrics.traces import build_trace_spec
    self.trace_specs = build_trace_spec("act")   # or "rl" / "tem"

Naming convention
-----------------
Trace keys follow the same namespace hierarchy as diagnostic signals:

    act/*      — ACT halting/stepping signals
    pred/*     — model output predictions
    loss/*     — loss scalars
    value/*    — value / Q-function outputs
    reward/*   — environment reward signals
    policy/*   — policy / action outputs
    dopamine/* — reward prediction error signals
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from ehc_sn.data.schema import O_ID
from ehc_sn.rollouts.collect import TraceField, TraceSpec, TraceValue
from ehc_sn.training.step_loop import StepContext

# =================================================================================================
# Common — usable in any paradigm (ACT and RL share these accessor paths)
# =================================================================================================


def _get_loss(ctx: StepContext) -> TraceValue:
    return ctx.outputs.loss.detach()


def _get_halted(ctx: StepContext) -> TraceValue:
    return ctx.carry.halted.detach()


def _get_steps(ctx: StepContext) -> TraceValue:
    return ctx.carry.steps.detach()


def _get_solution_overlay(ctx: StepContext) -> TraceValue:
    """Binary mask: 1 where the model predicts the solution-path token."""
    logits_lm: Tensor = ctx.outputs.outputs.logits[0]  # (B, S, vocab)
    pred = torch.argmax(logits_lm.detach(), dim=-1)  # (B, S)
    return (pred == O_ID).to(torch.uint8)


TRACE_LOSS = TraceField(name="loss/total", get=_get_loss)
TRACE_HALTED = TraceField(name="act/halted", get=_get_halted)
TRACE_STEPS = TraceField(name="act/steps", get=_get_steps)
TRACE_SOLUTION_OVERLAY = TraceField(name="pred/solution_overlay", get=_get_solution_overlay)

COMMON_TRACE_FIELDS: tuple[TraceField, ...] = (
    TRACE_LOSS,
    TRACE_HALTED,
    TRACE_STEPS,
    TRACE_SOLUTION_OVERLAY,
)


# =================================================================================================
# ACT-specific — produced by ACTLossHead / ACTController
# =================================================================================================


def _get_q_logits_act(ctx: StepContext) -> TraceValue:
    """Q-logits over halt/continue actions from the ACT controller."""
    logits_q: Tensor = ctx.outputs.outputs.logits[1]  # (B, n_actions)
    return logits_q.detach()


TRACE_Q_LOGITS_ACT = TraceField(name="value/q_logits", get=_get_q_logits_act)

ACT_TRACE_FIELDS: tuple[TraceField, ...] = (TRACE_Q_LOGITS_ACT,)


# =================================================================================================
# RL-specific — produced by RLLossHead / RLController
# =================================================================================================


def _get_q_logits_rl(ctx: StepContext) -> TraceValue:
    """vmPFC Q-logits over actions from the RL controller."""
    logits_q: Tensor = ctx.outputs.outputs.logits[1]  # (B, n_actions)
    return logits_q.detach()


def _get_r_logits_rl(ctx: StepContext) -> TraceValue:
    """STR value estimates V(s) from the RL critic."""
    logits_r: Tensor = ctx.outputs.outputs.logits[2]  # (B, 1)
    return logits_r.detach()


def _get_reward_env(ctx: StepContext) -> TraceValue:
    """Scalar environment reward for each batch slot."""
    return ctx.outputs.outputs.reward.squeeze(-1).detach()


def _get_action(ctx: StepContext) -> TraceValue:
    """Selected action index for each batch slot."""
    return ctx.outputs.outputs.action.detach()


def _get_rpe(ctx: StepContext) -> TraceValue:
    """Reward prediction error: reward − V(s)."""
    reward: Tensor = ctx.outputs.outputs.reward.squeeze(-1)
    value: Tensor = ctx.outputs.outputs.logits[2].squeeze(-1)  # STR critic
    return (reward - value).detach()


TRACE_Q_LOGITS_RL = TraceField(name="value/q_logits", get=_get_q_logits_rl)
TRACE_R_LOGITS_RL = TraceField(name="value/r_logits", get=_get_r_logits_rl)
TRACE_REWARD_ENV = TraceField(name="reward/env", get=_get_reward_env)
TRACE_ACTION = TraceField(name="policy/action", get=_get_action)
TRACE_RPE = TraceField(name="dopamine/rpe", get=_get_rpe)

RL_TRACE_FIELDS: tuple[TraceField, ...] = (
    TRACE_Q_LOGITS_RL,
    TRACE_R_LOGITS_RL,
    TRACE_REWARD_ENV,
    TRACE_ACTION,
    TRACE_RPE,
)


# =================================================================================================
# TEM-specific — TEM uses only the rollout-safe baseline fields for now.
# =================================================================================================


TEM_TRACE_FIELDS: tuple[TraceField, ...] = (
    TRACE_LOSS,
    TRACE_HALTED,
    TRACE_STEPS,
)


# =================================================================================================
def build_trace_spec(paradigm: Literal["act", "rl", "tem"]) -> TraceSpec:  # --------------------
    """Build a :class:`~ehc_sn.rollouts.collect.TraceSpec` for a training paradigm.

    Returns common fields plus paradigm-specific fields.  Pass the returned
    spec to :class:`~ehc_sn.rollouts.collect.TraceCollector` instead of
    constructing :class:`~ehc_sn.rollouts.collect.TraceField` lists in model files.

    Args:
        paradigm: ``"act"`` for ACT-based models (hrm_v1) or
            ``"rl"`` for RL-based models (hrm_v2), or
            ``"tem"`` for TEM-based models (tem_v1).

    Returns:
        A :class:`~ehc_sn.rollouts.collect.TraceSpec` instance.

    Raises:
        ValueError: If *paradigm* is not ``"act"``, ``"rl"``, or ``"tem"``.
    """
    if paradigm == "act":
        fields = COMMON_TRACE_FIELDS + ACT_TRACE_FIELDS
    elif paradigm == "rl":
        fields = COMMON_TRACE_FIELDS + RL_TRACE_FIELDS
    elif paradigm == "tem":
        fields = TEM_TRACE_FIELDS
    else:
        raise ValueError(f"Unknown paradigm: {paradigm!r}. Expected 'act', 'rl', or 'tem'.")
    return TraceSpec(fields=list(fields))


# =================================================================================================
__all__ = [
    "COMMON_TRACE_FIELDS", "ACT_TRACE_FIELDS", "RL_TRACE_FIELDS", TEM_TRACE_FIELDS,
    "build_trace_spec"
]  # fmt: skip
