"""Trace field vocabulary and per-paradigm TraceSpec constructors.

Mirrors :mod:`ehc_sn.metrics.signals` (scalar diagnostic names) and
:mod:`ehc_sn.metrics.routes` (metric routing tables) for the temporal rollout
data path.  All :class:`~ehc_sn.traces.TraceField` definitions live
here so model files contain *no* trace wiring.

Usage
-----
    from ehc_sn.metrics.traces import build_trace_spec
    self.trace_specs = build_trace_spec("act")   # or "rl" / "tem" / "ehc"

Naming convention
-----------------
Trace keys follow the same namespace hierarchy as diagnostic signals:

    act/*      — ACT halting/stepping signals
    pred/*     — model output predictions
    value/*    — value / Q-function outputs
    reward/*   — environment reward signals
    policy/*   — policy / action outputs
    dopamine/* — reward prediction error signals
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, Protocol

import torch
from torch import Tensor

from ehc_sn.traces import TraceField, TraceSpec, TraceValue
from ehc_sn.types import MemoryEntry


# =================================================================================================
class _CommonTraceCarry(Protocol):
    """Minimal carry surface shared across executed-step trace fields."""

    halted: Tensor
    steps: Tensor
    data: Mapping[str, Any]


class _CommonTraceContext(Protocol):
    """Minimal execution context shared across all trace paradigms."""

    index: int
    carry: _CommonTraceCarry


class _ACTTraceOutputs(Protocol):
    """Raw ACTStepOutput surface accessed by ACT trace fields."""

    class _Control(Protocol):
        q_logits: Tensor

    class _Backbone(Protocol):
        control: "_ACTTraceOutputs._Control"

    backbone_output: _Backbone


class _ACTTraceContext(_CommonTraceContext, Protocol):
    """Execution context exposing raw ACT controller outputs."""

    outputs: _ACTTraceOutputs


class _RLTraceOutputs(Protocol):
    """Output surface required by RL trace fields (matches InteractionRecord)."""

    policy_logits: Tensor
    value_estimate: Tensor
    reward: Tensor
    sampled_action: Tensor


class _RLTraceContext(_CommonTraceContext, Protocol):
    """Execution context exposing RL controller outputs."""

    outputs: _RLTraceOutputs


class _TEMTraceCarry(_CommonTraceCarry, Protocol):
    """Carry surface required by TEM trace fields."""

    static_data: Mapping[str, Any]
    model_state: Any


class _TEMTraceContext(_CommonTraceContext, Protocol):
    """Execution context exposing TEM carry."""

    carry: _TEMTraceCarry


# =================================================================================================
class ReplayableEnvironments:
    """Non-pytree wrapper for replayable environment metadata."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._items[index]


# =================================================================================================
# Common — usable in any executed rollout paradigm
# =================================================================================================


def _get_halted(ctx: _CommonTraceContext) -> TraceValue:
    return ctx.carry.halted.detach()


def _get_steps(ctx: _CommonTraceContext) -> TraceValue:
    return ctx.carry.steps.detach()


def _get_input_ids_meta(ctx: _CommonTraceContext) -> TraceValue:
    """Static token-id batch captured as metadata when figures request it."""
    value = ctx.carry.data.get("input_ids")
    return None if value is None else value.detach()


def _get_labels_meta(ctx: _CommonTraceContext) -> TraceValue:
    """Static label batch captured as metadata when figures request it."""
    value = ctx.carry.data.get("labels")
    return None if value is None else value.detach()


TRACE_HALTED = TraceField(
    name="act/halted",
    get=_get_halted,
)
TRACE_STEPS = TraceField(
    name="act/steps",
    get=_get_steps,
)
TRACE_INPUT_IDS_META = TraceField(
    name="input_ids",
    get=_get_input_ids_meta,
    storage="meta",
)
TRACE_LABELS_META = TraceField(
    name="labels",
    get=_get_labels_meta,
    storage="meta",
)

COMMON_TRACE_FIELDS: tuple[TraceField, ...] = (
    TRACE_HALTED,
    TRACE_STEPS,
    TRACE_INPUT_IDS_META,
    TRACE_LABELS_META,
)


# =================================================================================================
# ACT-specific — produced by ACTLossHead / ACTController
# =================================================================================================


def _get_q_logits_act(ctx: _ACTTraceContext) -> TraceValue:
    """Q-logits over halt/continue actions from the raw ACT controller step."""
    logits_q: Tensor = ctx.outputs.backbone_output.control.q_logits  # (B, n_actions)
    return logits_q.detach()


TRACE_Q_LOGITS_ACT = TraceField(
    name="value/q_logits",
    get=_get_q_logits_act,
)

ACT_TRACE_FIELDS: tuple[TraceField, ...] = (TRACE_Q_LOGITS_ACT,)


# =================================================================================================
# RL-specific — produced by HybridRLLossHead.compute_step() via HRMV2ValidationScorer
# =================================================================================================


def _get_q_logits_rl(ctx: _RLTraceContext) -> TraceValue:
    """Actor policy logits over actions from the RL controller."""
    return ctx.outputs.policy_logits.detach()


def _require_state_value(ctx: _RLTraceContext) -> Tensor:
    """Return the critic state value from the interaction record."""
    return ctx.outputs.value_estimate


def _get_state_value_rl(ctx: _RLTraceContext) -> TraceValue:
    """Critic state-value estimates V(s) from the actor-critic head."""
    return ctx.outputs.value_estimate.detach()


def _get_reward_env(ctx: _RLTraceContext) -> TraceValue:
    """Scalar environment reward for each batch slot."""
    return ctx.outputs.reward.squeeze(-1).detach()


def _get_action(ctx: _RLTraceContext) -> TraceValue:
    """Selected action index for each batch slot."""
    return ctx.outputs.sampled_action.detach()


def _get_rpe(ctx: _RLTraceContext) -> TraceValue:
    """Reward prediction error: reward − V(s)."""
    reward: Tensor = ctx.outputs.reward.squeeze(-1)
    value: Tensor = ctx.outputs.value_estimate.squeeze(-1)
    return (reward - value).detach()


def _get_world_observation_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Current-step observation encoding aligned with this step's TEM outputs."""
    return ctx.carry.data["observation"].detach()


def _get_world_location_ids_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Current-step location ids aligned with this step's TEM outputs."""
    return ctx.carry.data["location_id"].squeeze(-1).detach()


def _get_diagnostic_lec_cells_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Replayable LEC activations by frequency for diagnostic figures."""
    return [cell.detach() for cell in ctx.carry.model_state.lec.cells]


def _get_diagnostic_lec_filtered_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Replayable LEC filtered by frequency for diagnostic figures."""
    return [cell.detach() for cell in ctx.carry.model_state.lec.filtered]


def _get_diagnostic_mec_location_mean_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Replayable MEC location codes by frequency for diagnostic figures."""
    return [cell.detach() for cell in ctx.carry.model_state.mec.cells]


def _get_diagnostic_hpc_location_mean_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Replayable HPC grounded-location codes by frequency for diagnostic figures."""
    return [cell.detach() for cell in ctx.carry.model_state.hpc.cells]


def _get_diagnostic_hpc_memory_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Replayable final-step-compatible HPC memory state for diagnostic figures."""
    memory = ctx.carry.model_state.hpc.memory
    return {
        "g_cued": _memory_entry_for_trace(memory.g_cued),
        "x_cued": _memory_entry_for_trace(memory.x_cued),
    }


def _memory_entry_for_trace(memory: MemoryEntry) -> Tensor:
    """Return the canonical dense memory operator for trace storage."""
    return memory.to_dense().detach()


def _get_lec_alpha_sigmoid_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Static sigmoid-transformed LEC filter alpha values captured in carry data."""
    value = ctx.carry.data.get("lec_alpha_sigmoid")
    if value is None:
        static_data = getattr(ctx.carry, "static_data", None)
        value = None if static_data is None else static_data.get("lec_alpha_sigmoid")
    return None if value is None else value.detach()


def _get_lec_w_f_sigmoid_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Static sigmoid-transformed LEC frequency weights captured in carry data."""
    value = ctx.carry.data.get("lec_w_f_sigmoid")
    if value is None:
        static_data = getattr(ctx.carry, "static_data", None)
        value = None if static_data is None else static_data.get("lec_w_f_sigmoid")
    return None if value is None else value.detach()


TRACE_Q_LOGITS_RL = TraceField(
    name="value/policy_logits",
    get=_get_q_logits_rl,
)
TRACE_STATE_VALUE_RL = TraceField(
    name="value/state_value",
    get=_get_state_value_rl,
)
TRACE_REWARD_ENV = TraceField(
    name="reward/env",
    get=_get_reward_env,
)
TRACE_ACTION = TraceField(
    name="policy/action",
    get=_get_action,
)
TRACE_RPE = TraceField(
    name="dopamine/rpe",
    get=_get_rpe,
)
TRACE_WORLD_OBSERVATION_TEM = TraceField(
    name="world_step/observation",
    get=_get_world_observation_tem,
)
TRACE_WORLD_LOCATION_IDS_TEM = TraceField(
    name="world_step/location_ids",
    get=_get_world_location_ids_tem,
)
TRACE_DIAGNOSTIC_LEC_CELLS_TEM = TraceField(
    name="diagnostic/lec/cells",
    get=_get_diagnostic_lec_cells_tem,
)
TRACE_DIAGNOSTIC_LEC_FILTERED_TEM = TraceField(
    name="diagnostic/lec/filtered",
    get=_get_diagnostic_lec_filtered_tem,
)
TRACE_DIAGNOSTIC_MEC_LOCATION_MEAN_TEM = TraceField(
    name="diagnostic/mec/location_mean",
    get=_get_diagnostic_mec_location_mean_tem,
)
TRACE_DIAGNOSTIC_HPC_LOCATION_MEAN_TEM = TraceField(
    name="diagnostic/hpc/location_mean",
    get=_get_diagnostic_hpc_location_mean_tem,
)
TRACE_DIAGNOSTIC_HPC_MEMORY_TEM = TraceField(
    name="diagnostic/hpc/memory",
    get=_get_diagnostic_hpc_memory_tem,
)
TRACE_LEC_FILTER_ALPHA_SIGMOID_TEM = TraceField(
    name="lec/filter/alpha_sigmoid",
    get=_get_lec_alpha_sigmoid_tem,
    storage="meta",
)
TRACE_LEC_W_F_SIGMOID_TEM = TraceField(
    name="lec/w_f_sigmoid",
    get=_get_lec_w_f_sigmoid_tem,
    storage="meta",
)

RL_TRACE_FIELDS: tuple[TraceField, ...] = (
    TRACE_Q_LOGITS_RL,
    TRACE_STATE_VALUE_RL,
    TRACE_REWARD_ENV,
    TRACE_ACTION,
    TRACE_RPE,
)


# =================================================================================================
# TEM-specific — TEM uses only the rollout-safe baseline fields for now.
# =================================================================================================


TEM_TRACE_FIELDS: tuple[TraceField, ...] = (
    TRACE_HALTED,
    TRACE_STEPS,
    TRACE_WORLD_OBSERVATION_TEM,
    TRACE_WORLD_LOCATION_IDS_TEM,
    TRACE_DIAGNOSTIC_LEC_CELLS_TEM,
    TRACE_DIAGNOSTIC_LEC_FILTERED_TEM,
    TRACE_DIAGNOSTIC_MEC_LOCATION_MEAN_TEM,
    TRACE_DIAGNOSTIC_HPC_LOCATION_MEAN_TEM,
    TRACE_DIAGNOSTIC_HPC_MEMORY_TEM,
    TRACE_LEC_FILTER_ALPHA_SIGMOID_TEM,
    TRACE_LEC_W_F_SIGMOID_TEM,
)


# =================================================================================================
def _select_trace_fields(  # ----------------------------------------------------------------------
    fields: tuple[TraceField, ...], include_keys: Iterable[str] | None,
) -> tuple[TraceField, ...]:  # fmt: skip
    """Return the selected trace fields for a requested public key set."""
    if include_keys is None:
        return fields
    requested = {key for key in include_keys}
    return tuple(field for field in fields if field.name in requested)


# =================================================================================================
def build_trace_spec(  # --------------------------------------------------------------------------
    paradigm: Literal["act", "rl", "tem", "ehc"],
    *,
    include_keys: Iterable[str] | None = None,
    extra_fields: Iterable[TraceField] | None = None,
) -> TraceSpec:  # fmt: skip
    """Build a :class:`~ehc_sn.traces.TraceSpec` for a training paradigm.

    Returns common fields plus paradigm-specific fields.  Pass the returned
    spec to :class:`~ehc_sn.traces.TraceObserver` instead of
    constructing :class:`~ehc_sn.traces.TraceField` lists in model files.

    Args:
        paradigm: ``"act"`` for ACT-based models (hrm_v1) or
            ``"rl"`` for RL-based models (hrm_v2), or
            ``"tem"`` for TEM-based models (tem_v1), or
            ``"ehc"`` for EHC-based models (ehc_v1).

    Returns:
        A :class:`~ehc_sn.traces.TraceSpec` instance.

    Raises:
        ValueError: If *paradigm* is not ``"act"``, ``"rl"``, ``"tem"``, or ``"ehc"``.
    """
    if paradigm == "act":
        fields = _select_trace_fields(COMMON_TRACE_FIELDS + ACT_TRACE_FIELDS, include_keys)
    elif paradigm == "rl":
        fields = _select_trace_fields(COMMON_TRACE_FIELDS + RL_TRACE_FIELDS, include_keys)
    elif paradigm == "tem":
        fields = _select_trace_fields(TEM_TRACE_FIELDS, include_keys)
    elif paradigm == "ehc":
        fields = _select_trace_fields(TEM_TRACE_FIELDS, include_keys)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm!r}. Expected 'act', 'rl', 'tem', or 'ehc'.")
    if extra_fields is not None:
        fields = fields + tuple(extra_fields)
    return TraceSpec(fields=list(fields))


# =================================================================================================
__all__ = [
    "ReplayableEnvironments",
    "COMMON_TRACE_FIELDS", "ACT_TRACE_FIELDS", "RL_TRACE_FIELDS", TEM_TRACE_FIELDS,
    "build_trace_spec",
]  # fmt: skip
