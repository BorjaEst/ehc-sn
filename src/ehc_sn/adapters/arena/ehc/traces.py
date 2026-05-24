"""Arena-EHC trace fields.

These fields bind Arena task semantics to EHC-specific output surfaces.
They live here — in the shared Arena+EHC bridge namespace — because they
read from the replay controller wrapper (``ctx.outputs.backbone_output``) and
from the executed step payload (``ctx.batch``), both of which carry
task-model coupling that belongs at the adapter boundary.

World-step fields (observation_id, is_revisit) read ``ctx.batch``, which is
the executed_frame alias populated by the runner from carry-owned step data.
Prediction fields read ``ctx.outputs`` directly.

Usage
-----
    from ehc_sn.adapters.arena.ehc.traces import (
        ARENA_EHC_TRACE_FIELDS,
        select_arena_ehc_trace_fields,
    )
    self.trace_spec = build_trace_spec("ehc", extra_fields=ARENA_EHC_TRACE_FIELDS)
"""

from __future__ import annotations

from typing import Iterable, Mapping, Protocol

from torch import Tensor

from ehc_sn.tasks.arena.evaluation import coerce_observation_ids
from ehc_sn.traces import TraceField, TraceValue


# =============================================================================
class _ArenaEHCCarryData(Protocol):
    def __getitem__(self, key: str) -> Tensor: ...
    def get(self, key: str, default: Tensor | None = None) -> Tensor | None: ...


class _ArenaEHCCarry(Protocol):
    data: _ArenaEHCCarryData


class _ArenaEHCBackboneOutputs(Protocol):
    obs_logits: tuple[Tensor, Tensor, Tensor]


class _ArenaEHCOutputs(Protocol):
    backbone_output: _ArenaEHCBackboneOutputs


class _ArenaEHCTraceContext(Protocol):
    carry: _ArenaEHCCarry
    outputs: _ArenaEHCOutputs
    batch: Mapping[str, Tensor]


# =============================================================================
def _get_world_observation_id(ctx: _ArenaEHCTraceContext) -> TraceValue:
    obs_id: Tensor = ctx.batch["observation_id"]
    return coerce_observation_ids(obs_id).detach()


def _get_is_revisit(ctx: _ArenaEHCTraceContext) -> TraceValue:
    is_revisit: Tensor | None = ctx.batch.get("is_revisit")
    if is_revisit is None:
        return None
    return is_revisit.view(-1).bool().detach()


def _get_pred_obs_id_inference(ctx: _ArenaEHCTraceContext) -> TraceValue:
    return ctx.outputs.backbone_output.obs_logits[0].detach().argmax(dim=-1)


def _get_pred_obs_id_retrieved(ctx: _ArenaEHCTraceContext) -> TraceValue:
    return ctx.outputs.backbone_output.obs_logits[1].detach().argmax(dim=-1)


def _get_pred_obs_id_ancestral(ctx: _ArenaEHCTraceContext) -> TraceValue:
    return ctx.outputs.backbone_output.obs_logits[2].detach().argmax(dim=-1)


# =============================================================================
ARENA_EHC_TRACE_WORLD_OBS_ID = TraceField(
    name="world_step/observation_id",
    get=_get_world_observation_id,
)

ARENA_EHC_TRACE_IS_REVISIT = TraceField(
    name="protocol/is_revisit",
    get=_get_is_revisit,
)

ARENA_EHC_TRACE_PRED_INFERENCE = TraceField(
    name="pred/observation_id/inference",
    get=_get_pred_obs_id_inference,
)

ARENA_EHC_TRACE_PRED_RETRIEVED = TraceField(
    name="pred/observation_id/retrieved",
    get=_get_pred_obs_id_retrieved,
)

ARENA_EHC_TRACE_PRED_ANCESTRAL = TraceField(
    name="pred/observation_id/ancestral",
    get=_get_pred_obs_id_ancestral,
)

ARENA_EHC_TRACE_FIELDS: tuple[TraceField, ...] = (
    ARENA_EHC_TRACE_WORLD_OBS_ID,
    ARENA_EHC_TRACE_IS_REVISIT,
    ARENA_EHC_TRACE_PRED_INFERENCE,
    ARENA_EHC_TRACE_PRED_RETRIEVED,
    ARENA_EHC_TRACE_PRED_ANCESTRAL,
)
"""All Arena EHC trace fields in canonical order."""


# =============================================================================
def select_arena_ehc_trace_fields(
    include_keys: Iterable[str] | None,
) -> tuple[TraceField, ...]:
    """Return the Arena EHC trace fields matching a requested key set."""
    if include_keys is None:
        return ARENA_EHC_TRACE_FIELDS
    requested = set(include_keys)
    return tuple(f for f in ARENA_EHC_TRACE_FIELDS if f.name in requested)


# =============================================================================
__all__ = [
    "ARENA_EHC_TRACE_FIELDS",
    "ARENA_EHC_TRACE_WORLD_OBS_ID",
    "ARENA_EHC_TRACE_IS_REVISIT",
    "ARENA_EHC_TRACE_PRED_INFERENCE",
    "ARENA_EHC_TRACE_PRED_RETRIEVED",
    "ARENA_EHC_TRACE_PRED_ANCESTRAL",
    "select_arena_ehc_trace_fields",
]
