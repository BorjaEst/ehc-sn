"""Trace field vocabulary and per-paradigm TraceSpec constructors.

Mirrors :mod:`ehc_sn.metrics.signals` (scalar diagnostic names) and
:mod:`ehc_sn.metrics.routes` (metric routing tables) for the temporal rollout
data path.  All :class:`~ehc_sn.traces.TraceField` definitions live
here so model files contain *no* trace wiring.

Usage
-----
    from ehc_sn.traces import build_trace_spec
    self.trace_spec = build_trace_spec("act")   # or "rl" / "tem" / "ehp"

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

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Protocol

import torch
from torch import Tensor

from ehc_sn.traces.observer import TraceField, TraceSpec, TraceValue
from ehc_sn.types import MemoryEntry


# =============================================================================
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
    """Raw ACT controller-step output surface accessed by ACT trace fields."""

    q_logits: Tensor


class _ACTTraceContext(_CommonTraceContext, Protocol):
    """Execution context exposing raw ACT controller outputs."""

    outputs: _ACTTraceOutputs


class _RLTraceOutputs(Protocol):
    """Output surface required by RL trace fields (matches InteractionRecord)."""

    q_values: Tensor
    state_value: Tensor
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


class _HRMTraceCarry(_CommonTraceCarry, Protocol):
    """Carry surface required by HRM/PFC trace fields."""

    model_state: Any


class _HRMTraceContext(_CommonTraceContext, Protocol):
    """Execution context for HRM/PFC trace fields."""

    carry: _HRMTraceCarry


# =============================================================================
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


# =============================================================================
# Common — usable in any executed rollout paradigm
# =============================================================================


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


# =============================================================================
# ACT-specific — produced by ACTSupervisedScorer / ACTController
# =============================================================================


def _get_action_logits_act(ctx: _ACTTraceContext) -> TraceValue:
    """Action logits over halt/continue from the raw ACT controller step."""
    logits_q: Tensor = ctx.outputs.action_logits  # (B, n_actions)
    return logits_q.detach()


TRACE_ACTION_LOGITS_ACT = TraceField(
    name="value/action_logits",
    get=_get_action_logits_act,
)

ACT_TRACE_FIELDS: tuple[TraceField, ...] = (TRACE_ACTION_LOGITS_ACT,)


# =============================================================================
# RL-specific — produced by HybridRLObjective.compute_step() via validation scorer
# =============================================================================


def _get_q_logits_rl(ctx: _RLTraceContext) -> TraceValue:
    """Value-control Q-values over actions from the RL controller."""
    return ctx.outputs.q_values.detach().cpu()


def _get_state_value_rl(ctx: _RLTraceContext) -> TraceValue:
    """Critic state-value estimates V(s) from the value head."""
    return ctx.outputs.state_value.detach().cpu()


def _get_reward_env(ctx: _RLTraceContext) -> TraceValue:
    """Scalar environment reward for each batch slot."""
    return ctx.outputs.reward.squeeze(-1).detach().cpu()


def _get_action(ctx: _RLTraceContext) -> TraceValue:
    """Selected action index for each batch slot."""
    return ctx.outputs.sampled_action.detach().cpu()


def _get_rpe(ctx: _RLTraceContext) -> TraceValue:
    """Reward prediction error: reward − V(s)."""
    reward: Tensor = ctx.outputs.reward.squeeze(-1)
    value: Tensor = ctx.outputs.state_value.squeeze(-1)
    return (reward - value).detach().cpu()


def _get_world_observation_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Current-step observation encoding aligned with this step's TEM outputs."""
    v = ctx.carry.data.get("observation_id")
    return None if v is None else v.detach().cpu()


def _get_world_location_ids_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Current-step location ids aligned with this step's TEM outputs."""
    v = ctx.carry.data.get("location_id")
    return None if v is None else v.squeeze(-1).detach().cpu()


def _get_diagnostic_lec_cells_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Replayable LEC activations by frequency for diagnostic figures."""
    return [cell.detach().cpu() for cell in ctx.carry.model_state.lec.cells]


def _get_diagnostic_lec_filtered_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Replayable LEC filtered by frequency for diagnostic figures."""
    return [cell.detach().cpu() for cell in ctx.carry.model_state.lec.filtered]


def _get_diagnostic_lec_sensory_code_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Replayable raw sensory code entering LEC inference by frequency."""
    return [
        cell.detach().cpu() for cell in ctx.carry.model_state.lec.sensory_code
    ]


def _get_diagnostic_mec_location_mean_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Replayable MEC location codes by frequency for diagnostic figures."""
    return [cell.detach().cpu() for cell in ctx.carry.model_state.mec.cells]


def _get_diagnostic_hpc_location_mean_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Replayable HPC grounded-location codes by frequency for diagnostic figures."""
    return [cell.detach().cpu() for cell in ctx.carry.model_state.hpc.cells]


def _get_diagnostic_hpc_memory_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Replayable final-step-compatible HPC memory state for diagnostic figures."""
    memory = ctx.carry.model_state.hpc.memory
    return {
        "g_cued": _memory_entry_for_trace(memory.g_cued),
        "x_cued": _memory_entry_for_trace(memory.x_cued),
    }


def _memory_entry_for_trace(memory: MemoryEntry) -> Tensor:
    """Return the canonical dense memory operator for trace storage."""
    return memory.to_dense().detach().cpu()


def _get_lec_alpha_sigmoid_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Static sigmoid-transformed LEC filter alpha values captured in carry data."""
    value = ctx.carry.data.get("lec_alpha_sigmoid")
    if value is None:
        static_data = getattr(ctx.carry, "static_data", None)
        value = (
            None
            if static_data is None
            else static_data.get("lec_alpha_sigmoid")
        )
    return None if value is None else value.detach().cpu()


def _get_lec_w_f_sigmoid_tem(ctx: _TEMTraceContext) -> TraceValue:
    """Static sigmoid-transformed LEC frequency weights captured in carry data."""
    value = ctx.carry.data.get("lec_w_f_sigmoid")
    if value is None:
        static_data = getattr(ctx.carry, "static_data", None)
        value = (
            None if static_data is None else static_data.get("lec_w_f_sigmoid")
        )
    return None if value is None else value.detach().cpu()


TRACE_Q_LOGITS_RL = TraceField(
    name="value/q_values",
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
    requires_model_state=True,
)
TRACE_DIAGNOSTIC_LEC_FILTERED_TEM = TraceField(
    name="diagnostic/lec/filtered",
    get=_get_diagnostic_lec_filtered_tem,
    requires_model_state=True,
)
TRACE_DIAGNOSTIC_LEC_SENSORY_CODE_TEM = TraceField(
    name="diagnostic/lec/sensory_code",
    get=_get_diagnostic_lec_sensory_code_tem,
    requires_model_state=True,
)
TRACE_DIAGNOSTIC_MEC_LOCATION_MEAN_TEM = TraceField(
    name="diagnostic/mec/location_mean",
    get=_get_diagnostic_mec_location_mean_tem,
    requires_model_state=True,
)
TRACE_DIAGNOSTIC_HPC_LOCATION_MEAN_TEM = TraceField(
    name="diagnostic/hpc/location_mean",
    get=_get_diagnostic_hpc_location_mean_tem,
    requires_model_state=True,
)
TRACE_DIAGNOSTIC_HPC_MEMORY_TEM = TraceField(
    name="diagnostic/hpc/memory",
    get=_get_diagnostic_hpc_memory_tem,
    requires_model_state=True,
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


# =============================================================================
# HRM/PFC hidden-state — additive diagnostic fields for HRM-family models
# =============================================================================


def _get_hrm_pfc_memory(ctx: _HRMTraceContext) -> Any | None:
    """Return HRM/PFC working memory if present on the carry model state.

    Returns ``None`` when model state is missing or the expected PFC
    scratch memory path is not populated (e.g. a non-HRM model).
    """
    model_state = getattr(ctx.carry, "model_state", None)
    if model_state is None:
        return None
    pfc_state = getattr(model_state, "pfc", None)
    if pfc_state is None:
        return None
    scratch = getattr(pfc_state, "scratch", None)
    if scratch is None:
        return None
    return getattr(scratch, "memory", None)


def _get_hrm_z_H(ctx: _HRMTraceContext) -> TraceValue:
    """High-level HRM/PFC working-memory state.

    Expected shape ``(B, S+1, D)`` where *S+1* includes the controller
    slot.  Returns ``None`` when the memory path is unavailable.
    """
    memory = _get_hrm_pfc_memory(ctx)
    if memory is None:
        return None
    z_H = getattr(memory, "z_H", None)
    if z_H is None:
        return None
    return z_H.detach().cpu()


def _get_hrm_z_L(ctx: _HRMTraceContext) -> TraceValue:
    """Low-level HRM/PFC working-memory state.

    Expected shape ``(B, S+1, D)``.  Returns ``None`` when the memory
    path is unavailable.
    """
    memory = _get_hrm_pfc_memory(ctx)
    if memory is None:
        return None
    z_L = getattr(memory, "z_L", None)
    if z_L is None:
        return None
    return z_L.detach().cpu()


TRACE_HRM_Z_H = TraceField(
    name="pfc/z_H",
    get=_get_hrm_z_H,
    storage="dense",
    requires_model_state=True,
)

TRACE_HRM_Z_L = TraceField(
    name="pfc/z_L",
    get=_get_hrm_z_L,
    storage="dense",
    requires_model_state=True,
)

HRM_HIDDEN_STATE_FIELDS: tuple[TraceField, ...] = (
    TRACE_HRM_Z_H,
    TRACE_HRM_Z_L,
)

# WM task metadata — ground-truth labels for cue-dependent recall / working
# memory tasks.  These are placeholders; the full set of WM-specific fields
# will be populated when the cue-recall evaluation pipeline is implemented.
# For now, this tuple is deliberately empty to avoid NameError from the
# HRM_REASONING_TRACE_FIELDS reference below.
WM_TASK_METADATA_FIELDS: tuple[TraceField, ...] = ()

# Shared HRM reasoning fields — available to any HRM-family paradigm
# (act and rl are controller variants; the hidden-state and task-metadata
# surface is the same across ACT-style and RL-style HRM models.)
HRM_REASONING_TRACE_FIELDS: tuple[TraceField, ...] = (
    HRM_HIDDEN_STATE_FIELDS + WM_TASK_METADATA_FIELDS
)


# =============================================================================
# Goaltrace-specific — stored as metadata, consumed by figure selectors
# =============================================================================


def _get_goaltrace_firing_field(ctx: _CommonTraceContext) -> TraceValue:
    """Predicted firing field from goaltrace task metadata."""
    return ctx.carry.data.get("firing_field")


def _get_goaltrace_target_field(ctx: _CommonTraceContext) -> TraceValue:
    """Target firing field from goaltrace task metadata."""
    return ctx.carry.data.get("target_field")


def _get_goaltrace_node_mask(ctx: _CommonTraceContext) -> TraceValue:
    """Node validity mask from goaltrace task metadata."""
    return ctx.carry.data.get("node_mask")


def _get_goaltrace_observation_id(ctx: _CommonTraceContext) -> TraceValue:
    """Observation IDs from goaltrace task metadata."""
    return ctx.carry.data.get("observation_id")


GOALTRACE_TRACE_FIRING_FIELD = TraceField(
    name="goaltrace/firing_field",
    get=_get_goaltrace_firing_field,
    storage="meta",
)
GOALTRACE_TRACE_TARGET_FIELD = TraceField(
    name="goaltrace/target_field",
    get=_get_goaltrace_target_field,
    storage="meta",
)
GOALTRACE_TRACE_NODE_MASK = TraceField(
    name="goaltrace/node_mask",
    get=_get_goaltrace_node_mask,
    storage="meta",
)
GOALTRACE_TRACE_OBSERVATION_ID = TraceField(
    name="goaltrace/observation_id",
    get=_get_goaltrace_observation_id,
    storage="meta",
)

GOALTRACE_TRACE_FIELDS: tuple[TraceField, ...] = (
    GOALTRACE_TRACE_FIRING_FIELD,
    GOALTRACE_TRACE_TARGET_FIELD,
    GOALTRACE_TRACE_NODE_MASK,
    GOALTRACE_TRACE_OBSERVATION_ID,
)


# =============================================================================
# TEM-specific — TEM uses only the rollout-safe baseline fields for now.
# =============================================================================


TEM_TRACE_FIELDS: tuple[TraceField, ...] = (
    TRACE_HALTED,
    TRACE_STEPS,
    TRACE_WORLD_OBSERVATION_TEM,
    TRACE_WORLD_LOCATION_IDS_TEM,
    TRACE_DIAGNOSTIC_LEC_CELLS_TEM,
    TRACE_DIAGNOSTIC_LEC_FILTERED_TEM,
    TRACE_DIAGNOSTIC_LEC_SENSORY_CODE_TEM,
    TRACE_DIAGNOSTIC_MEC_LOCATION_MEAN_TEM,
    TRACE_DIAGNOSTIC_HPC_LOCATION_MEAN_TEM,
    TRACE_DIAGNOSTIC_HPC_MEMORY_TEM,
    TRACE_LEC_FILTER_ALPHA_SIGMOID_TEM,
    TRACE_LEC_W_F_SIGMOID_TEM,
)


# =============================================================================
def _select_trace_fields(  # --------------------------------------------------
    fields: tuple[TraceField, ...],
    include_keys: Iterable[str] | None,
) -> tuple[TraceField, ...]:
    """Return the selected trace fields for a requested public key set."""
    if include_keys is None:
        return fields
    requested = {key for key in include_keys}
    return tuple(field for field in fields if field.name in requested)


# =============================================================================
def build_trace_spec(  # ------------------------------------------------------
    paradigm: Literal["act", "rl", "tem", "ehp"],
    *,
    include_keys: Iterable[str] | None = None,
    extra_fields: Iterable[TraceField] | None = None,
) -> TraceSpec:
    """Build a :class:`~ehc_sn.traces.TraceSpec` for a training paradigm.

    Returns common fields plus paradigm-specific fields.  Pass the returned
    spec to :class:`~ehc_sn.traces.TraceObserver` instead of
    constructing :class:`~ehc_sn.traces.TraceField` lists in model files.

    Args:
        paradigm: ``"act"`` for ACT-based models (hrm_v1) or
            ``"rl"`` for RL-based models (hrm_v2), or
            ``"tem"`` for TEM-based models (tem_v1), or
            ``"ehp"`` for EHP-based models (ehc_v1).

    Returns:
        A :class:`~ehc_sn.traces.TraceSpec` instance.

    Raises:
        ValueError: If *paradigm* is not ``"act"``, ``"rl"``, ``"tem"``, or ``"ehp"``.
    """
    if paradigm == "act":
        fields = _select_trace_fields(
            COMMON_TRACE_FIELDS
            + ACT_TRACE_FIELDS
            + HRM_REASONING_TRACE_FIELDS
            + GOALTRACE_TRACE_FIELDS,
            include_keys,
        )
    elif paradigm == "rl":
        fields = _select_trace_fields(
            COMMON_TRACE_FIELDS
            + RL_TRACE_FIELDS
            + HRM_REASONING_TRACE_FIELDS
            + GOALTRACE_TRACE_FIELDS,
            include_keys,
        )
    elif paradigm == "tem":
        fields = _select_trace_fields(TEM_TRACE_FIELDS, include_keys)
    elif paradigm == "ehp":
        fields = _select_trace_fields(TEM_TRACE_FIELDS, include_keys)
    else:
        raise ValueError(
            f"Unknown paradigm: {paradigm!r}. Expected 'act', 'rl', 'tem', or 'ehp'."
        )
    if extra_fields is not None:
        fields = fields + tuple(extra_fields)
    return TraceSpec(fields=list(fields))


# =============================================================================
# Capture profiles — named sets of trace fields declared by analysis consumers
# ---------------------------------------------------------------------------
# Each profile maps paradigm → (required_keys, default_optional_keys).
# Required keys cannot be excluded.  Unsupported profile–paradigm combinations
# raise ValueError at experiment construction time.
# =============================================================================


@dataclass(frozen=True)
class CaptureProfileSpec:
    """A named capture profile with paradigm-specific field requirements.

    This is the consumer-facing contract: the fields a given analysis
    type (e.g. prediction-reasoning, full-diagnostic) needs to function.
    """

    name: str
    version: int
    description: str
    paradigm_fields: dict[str, CaptureParadigmBinding]


@dataclass(frozen=True)
class CaptureParadigmBinding:
    """Required and optional trace fields for one paradigm under a profile."""

    required: tuple[str, ...]
    optional: tuple[str, ...] = ()


# ── Profile: prediction_reasoning ────────────────────────────────────────────
# Requires controller halt/step state plus decoded prediction and target.
# Task-specific additions (e.g. solution_overlay for MazeHard) are injected
# by experiment builders via include=.

_PREDICTION_REASONING_BINDINGS: dict[str, CaptureParadigmBinding] = {
    "act": CaptureParadigmBinding(
        required=(
            "act/halted",
            "act/steps",
            "pred/solution_overlay",
            "target/solution_overlay",
        ),
        optional=("value/action_logits", "input_ids"),
    ),
    "rl": CaptureParadigmBinding(
        required=(
            "act/halted",
            "act/steps",
            "pred/solution_overlay",
            "target/solution_overlay",
        ),
        optional=(
            "value/q_values",
            "value/q_logits",
            "value/state_value",
            "input_ids",
        ),
    ),
    "tem": CaptureParadigmBinding(
        required=(
            "act/halted",
            "act/steps",
            "pred/observation_id/post",
            "target/observation_id",
        ),
        optional=(
            "pred/observation_id/recall",
            "pred/observation_id/path",
        ),
    ),
    "ehp": CaptureParadigmBinding(
        required=(
            "act/halted",
            "act/steps",
            "pred/observation_id/post",
            "target/observation_id",
        ),
        optional=(
            "pred/observation_id/recall",
            "pred/observation_id/path",
        ),
    ),
}

# ── Profile: full_diagnostic ─────────────────────────────────────────────────
# All available controller, prediction, and diagnostic fields for a paradigm.

_FULL_DIAGNOSTIC_BINDINGS: dict[str, CaptureParadigmBinding] = {
    "act": CaptureParadigmBinding(
        required=(
            "act/halted",
            "act/steps",
            "pred/solution_overlay",
            "target/solution_overlay",
            "value/action_logits",
        ),
        optional=("pfc/z_H", "pfc/z_L", "wm/input_ids", "wm/step"),
    ),
    "rl": CaptureParadigmBinding(
        required=(
            "act/halted",
            "act/steps",
            "pred/solution_overlay",
            "target/solution_overlay",
            "value/q_values",
            "value/q_logits",
            "value/state_value",
        ),
        optional=("pfc/z_H", "pfc/z_L", "wm/input_ids", "wm/step"),
    ),
    "tem": CaptureParadigmBinding(
        required=(
            "act/halted",
            "act/steps",
            "pred/observation_id/post",
            "pred/observation_id/recall",
            "target/observation_id",
        ),
        optional=(
            "pred/observation_id/path",
            "world_step/observation",
            "world_step/location_ids",
            "diagnostic/lec/cells",
            "diagnostic/lec/filtered",
            "diagnostic/mec/location_mean",
            "diagnostic/hpc/location_mean",
            "diagnostic/hpc/memory",
        ),
    ),
    "ehp": CaptureParadigmBinding(
        required=(
            "act/halted",
            "act/steps",
            "pred/observation_id/post",
            "pred/observation_id/recall",
            "target/observation_id",
        ),
        optional=(
            "pred/observation_id/path",
            "world_step/observation",
            "world_step/location_ids",
            "diagnostic/lec/cells",
            "diagnostic/lec/filtered",
            "diagnostic/mec/location_mean",
            "diagnostic/hpc/location_mean",
            "diagnostic/hpc/memory",
        ),
    ),
}

# ── Profile: metrics_only ────────────────────────────────────────────────────
# No trace fields; only scalar metrics are collected.

_METRICS_ONLY_BINDINGS: dict[str, CaptureParadigmBinding] = {
    paradigm: CaptureParadigmBinding(required=())
    for paradigm in ("act", "rl", "tem", "ehp")
}

# ── Registry ─────────────────────────────────────────────────────────────────

TRACE_PROFILES: dict[str, CaptureProfileSpec] = {
    "metrics_only": CaptureProfileSpec(
        name="metrics_only",
        version=1,
        description="No trace fields — only scalar metrics are collected.",
        paradigm_fields=_METRICS_ONLY_BINDINGS,
    ),
    "prediction_reasoning": CaptureProfileSpec(
        name="prediction_reasoning",
        version=1,
        description="Controller halt/step state plus decoded prediction and target.",
        paradigm_fields=_PREDICTION_REASONING_BINDINGS,
    ),
    "full_diagnostic": CaptureProfileSpec(
        name="full_diagnostic",
        version=1,
        description="All available controller, prediction, and diagnostic fields.",
        paradigm_fields=_FULL_DIAGNOSTIC_BINDINGS,
    ),
}


def _paradigm_all_field_names(paradigm: str) -> set[str]:
    """Return the set of all known trace-field names for *paradigm*."""
    if paradigm == "act":
        fields = (
            COMMON_TRACE_FIELDS
            + ACT_TRACE_FIELDS
            + HRM_REASONING_TRACE_FIELDS
            + GOALTRACE_TRACE_FIELDS
        )
    elif paradigm == "rl":
        fields = (
            COMMON_TRACE_FIELDS
            + RL_TRACE_FIELDS
            + HRM_REASONING_TRACE_FIELDS
            + GOALTRACE_TRACE_FIELDS
        )
    elif paradigm in ("tem", "ehp"):
        fields = TEM_TRACE_FIELDS
    else:
        raise ValueError(
            f"Unknown paradigm: {paradigm!r}. Expected 'act', 'rl', 'tem', or 'ehp'."
        )
    return {f.name for f in fields}


def resolve_capture_profile(
    paradigm: Literal["act", "rl", "tem", "ehp"],
    profile: str = "metrics_only",
    *,
    profile_version: int = 1,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    task_bindings: dict[str, CaptureParadigmBinding] | None = None,
) -> TraceSpec:
    """Resolve a named capture profile to a concrete :class:`TraceSpec`.

    Parameters
    ----------
    paradigm:
        Trace paradigm for the experiment being evaluated.
    profile:
        Named capture profile registered in :data:`TRACE_PROFILES`.
    profile_version:
        Version of the profile contract.  Currently only version 1 exists; a
        mismatch raises ``ValueError``.
    include:
        Extra trace-field names to include beyond the profile.  Each name must
        correspond to a known trace field for *paradigm*.
    exclude:
        Profile field names to omit.  May only exclude *optional* profile
        fields — required profile fields raise ``ValueError``.
    task_bindings:
        Per-task field bindings appended to the profile's paradigm fields.
        Keys are task names (e.g. ``"goaltrace"``).  Each binding's required
        and optional fields are merged into the resolution.  Unknown field
        names in task bindings raise ``ValueError``.

    Returns
    -------
    TraceSpec
        A concrete trace specification ready for use by the evaluation runner.

    Raises
    ------
    ValueError
        If *profile* is unknown, if *profile_version* does not match, if
        *paradigm* is unsupported for the profile, if *exclude* intersects
        required profile fields, if *include* references unknown field names,
        or if a task binding references unknown field names.
    """
    profile_spec = TRACE_PROFILES.get(profile)
    if profile_spec is None:
        raise ValueError(
            f"Unknown capture profile {profile!r}. "
            f"Available: {sorted(TRACE_PROFILES)}."
        )
    if profile_spec.version != profile_version:
        raise ValueError(
            f"Capture profile {profile!r} version mismatch: "
            f"requested v{profile_version}, available v{profile_spec.version}."
        )

    binding = profile_spec.paradigm_fields.get(paradigm)
    if binding is None:
        raise ValueError(
            f"Capture profile {profile!r} does not support paradigm "
            f"{paradigm!r}. Supported: {sorted(profile_spec.paradigm_fields)}."
        )

    # Collect required and optional from profile + task bindings.
    all_known = _paradigm_all_field_names(paradigm)
    required_set: set[str] = set(binding.required)
    optional_set: set[str] = set(binding.optional)

    if task_bindings:
        for task_name, task_binding in task_bindings.items():
            for f in task_binding.required:
                # Task bindings may reference task-specific meta-keys not in
                # the core paradigm field set (e.g. goaltrace/firing_field).
                # Include validation is relaxed; build_trace_spec will filter
                # to known fields.
                required_set.add(f)
            for f in task_binding.optional:
                optional_set.add(f)

    include_set = set(include)
    exclude_set = set(exclude)

    # Validate include keys are known.
    unknown_include = include_set - all_known
    if unknown_include:
        raise ValueError(
            f"Capture include keys unknown for paradigm {paradigm!r}: "
            f"{sorted(unknown_include)}. Known: {sorted(all_known)}."
        )

    # Validate exclude does not intersect required.
    forbidden_exclude = exclude_set & required_set
    if forbidden_exclude:
        raise ValueError(
            f"Cannot exclude required capture fields for profile "
            f"{profile!r}: {sorted(forbidden_exclude)}."
        )

    # Deterministic ordering: required (profile order), then optional (profile
    # order), then include (user order).  Uses dict.fromkeys for order-preserving
    # dedup.
    final_names: list[str] = []
    seen: set[str] = set()

    # 1. Required profile fields in declaration order.
    for f in binding.required:
        if f not in seen:
            final_names.append(f)
            seen.add(f)
    # 1b. Required task-binding fields (appended after profile required).
    if task_bindings:
        for tb in task_bindings.values():
            for f in tb.required:
                if f not in seen:
                    final_names.append(f)
                    seen.add(f)

    # 2. Optional profile fields in declaration order, minus exclude.
    for f in binding.optional:
        if f not in seen and f not in exclude_set:
            final_names.append(f)
            seen.add(f)
    # 2b. Optional task-binding fields.
    if task_bindings:
        for tb in task_bindings.values():
            for f in tb.optional:
                if f not in seen and f not in exclude_set:
                    final_names.append(f)
                    seen.add(f)

    # 3. Explicit includes in user order.
    for f in include:
        if f not in seen:
            final_names.append(f)
            seen.add(f)

    return build_trace_spec(paradigm, include_keys=final_names)


# =============================================================================
__all__ = [
    "ReplayableEnvironments",
    "COMMON_TRACE_FIELDS",
    "ACT_TRACE_FIELDS",
    "RL_TRACE_FIELDS",
    "HRM_HIDDEN_STATE_FIELDS",
    "HRM_REASONING_TRACE_FIELDS",
    "TEM_TRACE_FIELDS",
    "GOALTRACE_TRACE_FIELDS",
    "TRACE_PROFILES",
    "CaptureProfileSpec",
    "CaptureParadigmBinding",
    "build_trace_spec",
    "resolve_capture_profile",
]
