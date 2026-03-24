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

from typing import Any, Iterable, Literal

import torch
from torch import Tensor

from ehc_sn.data.schema import O_ID
from ehc_sn.rollouts.collect import TraceField, TraceSpec, TraceValue
from ehc_sn.training.step_loop import StepContext


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


def _get_inputs_meta(ctx: StepContext) -> TraceValue:
    """Static input batch captured as metadata when figures request it."""
    value = ctx.carry.data.get("inputs")
    return None if value is None else value.detach()


def _get_labels_meta(ctx: StepContext) -> TraceValue:
    """Static label batch captured as metadata when figures request it."""
    value = ctx.carry.data.get("labels")
    return None if value is None else value.detach()


TRACE_LOSS = TraceField(
    name="loss/total",
    get=_get_loss,
)
TRACE_HALTED = TraceField(
    name="act/halted",
    get=_get_halted,
)
TRACE_STEPS = TraceField(
    name="act/steps",
    get=_get_steps,
)
TRACE_SOLUTION_OVERLAY = TraceField(
    name="pred/solution_overlay",
    get=_get_solution_overlay,
)
TRACE_INPUTS_META = TraceField(
    name="inputs",
    get=_get_inputs_meta,
    storage="meta",
)
TRACE_LABELS_META = TraceField(
    name="labels",
    get=_get_labels_meta,
    storage="meta",
)

COMMON_TRACE_FIELDS: tuple[TraceField, ...] = (
    TRACE_LOSS,
    TRACE_HALTED,
    TRACE_STEPS,
    TRACE_SOLUTION_OVERLAY,
    TRACE_INPUTS_META,
    TRACE_LABELS_META,
)


# =================================================================================================
# ACT-specific — produced by ACTLossHead / ACTController
# =================================================================================================


def _get_q_logits_act(ctx: StepContext) -> TraceValue:
    """Q-logits over halt/continue actions from the ACT controller."""
    logits_q: Tensor = ctx.outputs.outputs.logits[1]  # (B, n_actions)
    return logits_q.detach()


TRACE_Q_LOGITS_ACT = TraceField(
    name="value/q_logits",
    get=_get_q_logits_act,
)

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


def _get_world_observation_tem(ctx: StepContext) -> TraceValue:
    """Post-step observation encoding aligned with the current rollout state."""
    return ctx.carry.data["inputs"].detach()


def _get_world_location_ids_tem(ctx: StepContext) -> TraceValue:
    """Post-step location ids aligned with the current rollout state."""
    return ctx.carry.data["location_id"].squeeze(-1).detach()


def _get_environments_tem(ctx: StepContext) -> TraceValue:
    """Replayable world metadata derived from the static maze batch."""
    topology = ctx.carry.static_data["topology"]
    mask_valid = ctx.carry.static_data.get("mask_valid")
    return ReplayableEnvironments(
        [
            _build_environment_metadata(topology[idx], None if mask_valid is None else mask_valid[idx])
            for idx in range(int(topology.shape[0]))
        ]
    )


def _get_diagnostic_lec_cells_tem(ctx: StepContext) -> TraceValue:
    """Replayable LEC activations by frequency for diagnostic figures."""
    return [cell.detach() for cell in ctx.carry.model_state.lec.cells]


def _get_diagnostic_lec_filtered_tem(ctx: StepContext) -> TraceValue:
    """Replayable LEC filtered by frequency for diagnostic figures."""
    return [cell.detach() for cell in ctx.carry.model_state.lec.filtered]


def _get_diagnostic_mec_location_mean_tem(ctx: StepContext) -> TraceValue:
    """Replayable MEC location codes by frequency for diagnostic figures."""
    return [cell.detach() for cell in ctx.carry.model_state.mec.cells]


def _get_diagnostic_hpc_location_mean_tem(ctx: StepContext) -> TraceValue:
    """Replayable HPC grounded-location codes by frequency for diagnostic figures."""
    return [cell.detach() for cell in ctx.carry.model_state.hpc.cells]


def _get_diagnostic_hpc_memory_tem(ctx: StepContext) -> TraceValue:
    """Replayable final-step-compatible HPC memory state for diagnostic figures."""
    memory = ctx.carry.model_state.hpc.memory
    return {
        "g_cued": _memory_entry_for_trace(memory.g_cued),
        "x_cued": _memory_entry_for_trace(memory.x_cued),
    }


def _memory_entry_for_trace(memory: TraceValue) -> TraceValue:
    """Convert backend-specific memory entries into replayable trace tensors."""
    if isinstance(memory, torch.Tensor):
        return memory.detach()
    values = getattr(memory, "values", None)
    if isinstance(values, torch.Tensor):
        return values.detach()
    return memory


def _get_lec_alpha_sigmoid_tem(ctx: StepContext) -> TraceValue:
    """Static sigmoid-transformed LEC filter alpha values captured as metadata."""
    backbone = ctx.step_module.controller.backbone
    return torch.stack([torch.sigmoid(alpha).detach() for alpha in backbone.lec.filter.alpha])


def _get_lec_w_f_sigmoid_tem(ctx: StepContext) -> TraceValue:
    """Static sigmoid-transformed LEC frequency weights captured as metadata."""
    backbone = ctx.step_module.controller.backbone
    return torch.stack([torch.sigmoid(weight).detach() for weight in backbone.lec.w_f])


TRACE_Q_LOGITS_RL = TraceField(
    name="value/q_logits",
    get=_get_q_logits_rl,
)
TRACE_R_LOGITS_RL = TraceField(
    name="value/r_logits",
    get=_get_r_logits_rl,
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
TRACE_ENVIRONMENTS_TEM = TraceField(
    name="environments",
    get=_get_environments_tem,
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
    TRACE_WORLD_OBSERVATION_TEM,
    TRACE_WORLD_LOCATION_IDS_TEM,
    TRACE_ENVIRONMENTS_TEM,
    TRACE_DIAGNOSTIC_LEC_CELLS_TEM,
    TRACE_DIAGNOSTIC_LEC_FILTERED_TEM,
    TRACE_DIAGNOSTIC_MEC_LOCATION_MEAN_TEM,
    TRACE_DIAGNOSTIC_HPC_LOCATION_MEAN_TEM,
    TRACE_DIAGNOSTIC_HPC_MEMORY_TEM,
    TRACE_LEC_FILTER_ALPHA_SIGMOID_TEM,
    TRACE_LEC_W_F_SIGMOID_TEM,
)


# =================================================================================================
def _build_environment_metadata(  # ---------------------------------------------------------------
    topology: Tensor, mask_valid: Tensor | None,
) -> dict[str, Any]:  # fmt: skip
    """Build a lightweight world-like mapping for replayable spatial figures."""
    topology_np = topology.detach().cpu().to(torch.bool).numpy()
    mask_valid_np = None if mask_valid is None else mask_valid.detach().cpu().to(torch.bool).numpy()
    height, width = topology_np.shape[-2], topology_np.shape[-1]
    locations: list[dict[str, float | bool]] = []
    for row in range(height):
        for col in range(width):
            valid = bool(topology_np[row, col])
            if mask_valid_np is not None:
                valid = valid and bool(mask_valid_np[row, col])
            locations.append({"o": float(col), "y": float(row), "valid": valid})
    return {"locations": locations, "n_locations": len(locations)}


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
    paradigm: Literal["act", "rl", "tem"], *, include_keys: Iterable[str] | None = None,
) -> TraceSpec:  # fmt: skip
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
        fields = _select_trace_fields(COMMON_TRACE_FIELDS + ACT_TRACE_FIELDS, include_keys)
    elif paradigm == "rl":
        fields = _select_trace_fields(COMMON_TRACE_FIELDS + RL_TRACE_FIELDS, include_keys)
    elif paradigm == "tem":
        fields = _select_trace_fields(TEM_TRACE_FIELDS, include_keys)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm!r}. Expected 'act', 'rl', or 'tem'.")
    return TraceSpec(fields=list(fields))


# =================================================================================================
__all__ = [
    "ReplayableEnvironments",
    "COMMON_TRACE_FIELDS", "ACT_TRACE_FIELDS", "RL_TRACE_FIELDS", TEM_TRACE_FIELDS,
    "build_trace_spec",
]  # fmt: skip
