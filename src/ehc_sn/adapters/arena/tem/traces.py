"""Arena-TEM trace fields.

These fields bind Arena task semantics to TEM-specific output surfaces.
They live here — in the shared Arena+TEM bridge namespace — because they
read from the normalised TEM controller output (``ctx.outputs.obs_logits``) and
from the arena step payload (``ctx.carry.data``), both of which carry
task-model coupling that belongs at the adapter boundary.

Usage
-----
    from ehc_sn.adapters.arena.tem.traces import (
        ARENA_TEM_TRACE_FIELDS,
        select_arena_tem_trace_fields,
    )
    self.trace_specs = build_trace_spec("tem", extra_fields=ARENA_TEM_TRACE_FIELDS)
"""

from __future__ import annotations

from typing import Iterable, Protocol

from torch import Tensor

from ehc_sn.tasks.arena.evaluation import coerce_observation_ids
from ehc_sn.traces import TraceField, TraceValue


# =================================================================================================
class _ArenaTEMCarryData(Protocol):
    def __getitem__(self, key: str) -> Tensor: ...
    def get(self, key: str, default: Tensor | None = None) -> Tensor | None: ...


class _ArenaTEMCarry(Protocol):
    data: _ArenaTEMCarryData


class _ArenaTEMOutputs(Protocol):
    obs_logits: tuple[Tensor, Tensor, Tensor]


class _ArenaTEMTraceContext(Protocol):
    carry: _ArenaTEMCarry
    outputs: _ArenaTEMOutputs


# =================================================================================================
def _get_world_observation_id(ctx: _ArenaTEMTraceContext) -> TraceValue:
    obs_id: Tensor = ctx.carry.data["observation_id"]
    return coerce_observation_ids(obs_id).detach()


def _get_is_revisit(ctx: _ArenaTEMTraceContext) -> TraceValue:
    is_revisit: Tensor | None = ctx.carry.data.get("is_revisit")
    if is_revisit is None:
        return None
    return is_revisit.view(-1).bool().detach()


def _get_pred_obs_id_inference(ctx: _ArenaTEMTraceContext) -> TraceValue:
    return ctx.outputs.obs_logits[0].detach().argmax(dim=-1)


def _get_pred_obs_id_retrieved(ctx: _ArenaTEMTraceContext) -> TraceValue:
    return ctx.outputs.obs_logits[1].detach().argmax(dim=-1)


def _get_pred_obs_id_ancestral(ctx: _ArenaTEMTraceContext) -> TraceValue:
    return ctx.outputs.obs_logits[2].detach().argmax(dim=-1)


# =================================================================================================
ARENA_TEM_TRACE_WORLD_OBS_ID = TraceField(
    name="world_step/observation_id",
    get=_get_world_observation_id,
)

ARENA_TEM_TRACE_IS_REVISIT = TraceField(
    name="protocol/is_revisit",
    get=_get_is_revisit,
)

ARENA_TEM_TRACE_PRED_INFERENCE = TraceField(
    name="pred/observation_id/inference",
    get=_get_pred_obs_id_inference,
)

ARENA_TEM_TRACE_PRED_RETRIEVED = TraceField(
    name="pred/observation_id/retrieved",
    get=_get_pred_obs_id_retrieved,
)

ARENA_TEM_TRACE_PRED_ANCESTRAL = TraceField(
    name="pred/observation_id/ancestral",
    get=_get_pred_obs_id_ancestral,
)

ARENA_TEM_TRACE_FIELDS: tuple[TraceField, ...] = (
    ARENA_TEM_TRACE_WORLD_OBS_ID,
    ARENA_TEM_TRACE_IS_REVISIT,
    ARENA_TEM_TRACE_PRED_INFERENCE,
    ARENA_TEM_TRACE_PRED_RETRIEVED,
    ARENA_TEM_TRACE_PRED_ANCESTRAL,
)
"""All Arena TEM trace fields in canonical order."""


# =================================================================================================
def select_arena_tem_trace_fields(
    include_keys: Iterable[str] | None,
) -> tuple[TraceField, ...]:
    """Return the Arena TEM trace fields matching a requested key set."""
    if include_keys is None:
        return ARENA_TEM_TRACE_FIELDS
    requested = set(include_keys)
    return tuple(f for f in ARENA_TEM_TRACE_FIELDS if f.name in requested)


# =================================================================================================
__all__ = [
    "ARENA_TEM_TRACE_FIELDS",
    "ARENA_TEM_TRACE_WORLD_OBS_ID",
    "ARENA_TEM_TRACE_IS_REVISIT",
    "ARENA_TEM_TRACE_PRED_INFERENCE",
    "ARENA_TEM_TRACE_PRED_RETRIEVED",
    "ARENA_TEM_TRACE_PRED_ANCESTRAL",
    "select_arena_tem_trace_fields",
]
