"""Arena-TEM trace fields.

These fields bind Arena task semantics to TEM-specific output surfaces.
They live here — in the shared Arena+TEM bridge namespace — because they
read from the replay controller wrapper (``ctx.outputs.backbone_output``) and
from the executed step payload (``ctx.batch``), both of which carry
task-model coupling that belongs at the adapter boundary.

World-step fields (observation_id, is_revisit) read ``ctx.batch``, which is
the executed_frame alias populated by the runner from carry-owned step data.
Prediction fields read ``ctx.outputs`` directly.

Usage
-----
    from ehc_sn.adapters.arena.tem.traces import (
        ARENA_TEM_TRACE_FIELDS,
        select_arena_tem_trace_fields,
    )
    self.trace_spec = build_trace_spec("tem", extra_fields=ARENA_TEM_TRACE_FIELDS)
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol

from torch import Tensor

from ehc_sn.tasks.arena.evaluation import coerce_observation_ids
from ehc_sn.traces import TraceField, TraceValue
from ehc_sn.traces.specs import (
    TRACE_DIAGNOSTIC_HPC_LOCATION_MEAN_TEM,
    TRACE_DIAGNOSTIC_MEC_LOCATION_MEAN_TEM,
)
from ehc_sn.types import Batch


# =============================================================================
class _ArenaTEMCarryData(Protocol):
    def __getitem__(  # -------------------------------------------------------
        self,
        key: str,
    ) -> Tensor: ...

    def get(  # ---------------------------------------------------------------
        self,
        key: str,
        default: Tensor | None = None,
    ) -> Tensor | None: ...


class _ArenaTEMCarry(Protocol):
    data: _ArenaTEMCarryData
    model_state: Any  # Needed when diagnostic trace fields are exported


class _ArenaTEMBackboneOutputs(Protocol):
    obs_logits: tuple[Tensor, Tensor, Tensor]


class _ArenaTEMOutputs(Protocol):
    backbone_output: _ArenaTEMBackboneOutputs


class _ArenaTEMTraceContext(Protocol):
    carry: _ArenaTEMCarry
    outputs: _ArenaTEMOutputs
    batch: Mapping[str, Tensor]


# =============================================================================
def _get_world_observation_id(  # ---------------------------------------------
    ctx: _ArenaTEMTraceContext,
) -> TraceValue:
    obs_id: Tensor = ctx.batch["observation_id"]
    return coerce_observation_ids(obs_id).detach()


# =============================================================================
def _get_is_revisit(  # -------------------------------------------------------
    ctx: _ArenaTEMTraceContext,
) -> TraceValue:
    is_revisit: Tensor | None = ctx.batch.get("is_revisit")
    if is_revisit is None:
        return None
    return is_revisit.view(-1).bool().detach()


# =============================================================================
def _get_pred_obs_id_post(  # -------------------------------------------------
    ctx: _ArenaTEMTraceContext,
) -> TraceValue:
    return ctx.outputs.backbone_output.obs_logits[0].detach().argmax(dim=-1)


# =============================================================================
def _get_pred_obs_id_recall(  # -----------------------------------------------
    ctx: _ArenaTEMTraceContext,
) -> TraceValue:
    return ctx.outputs.backbone_output.obs_logits[1].detach().argmax(dim=-1)


# =============================================================================
def _get_pred_obs_id_path(  # -------------------------------------------------
    ctx: _ArenaTEMTraceContext,
) -> TraceValue:
    return ctx.outputs.backbone_output.obs_logits[2].detach().argmax(dim=-1)


# =============================================================================
ARENA_TEM_TRACE_WORLD_OBS_ID = TraceField(
    name="world_step/observation_id",
    get=_get_world_observation_id,
)

ARENA_TEM_TRACE_IS_REVISIT = TraceField(
    name="protocol/is_revisit",
    get=_get_is_revisit,
)

ARENA_TEM_TRACE_PRED_POST = TraceField(
    name="pred/observation_id/post",
    get=_get_pred_obs_id_post,
)

ARENA_TEM_TRACE_PRED_RECALL = TraceField(
    name="pred/observation_id/recall",
    get=_get_pred_obs_id_recall,
)

ARENA_TEM_TRACE_PRED_PATH = TraceField(
    name="pred/observation_id/path",
    get=_get_pred_obs_id_path,
)

ARENA_TEM_TRACE_FIELDS: tuple[TraceField, ...] = (
    ARENA_TEM_TRACE_WORLD_OBS_ID,
    ARENA_TEM_TRACE_IS_REVISIT,
    ARENA_TEM_TRACE_PRED_POST,
    ARENA_TEM_TRACE_PRED_RECALL,
    ARENA_TEM_TRACE_PRED_PATH,
    TRACE_DIAGNOSTIC_MEC_LOCATION_MEAN_TEM,
    TRACE_DIAGNOSTIC_HPC_LOCATION_MEAN_TEM,
)
"""All Arena TEM trace fields in canonical order."""


# =============================================================================
def select_arena_tem_trace_fields(  # -----------------------------------------
    include_keys: Iterable[str] | None,
) -> tuple[TraceField, ...]:
    """Return the Arena TEM trace fields matching a requested key set."""
    if include_keys is None:
        return ARENA_TEM_TRACE_FIELDS
    requested = set(include_keys)
    return tuple(f for f in ARENA_TEM_TRACE_FIELDS if f.name in requested)


# =============================================================================
TARGET_OBSERVATION_ID_META_KEY: str = "target/observation_id"
"""Trace metadata key for the ground-truth observation-id trajectory.

Populated by :func:`build_arena_tem_trace_meta` and read by TEM figure
selectors via ``trace.get_meta_path(TARGET_OBSERVATION_ID_META_KEY)``.
"""


# =============================================================================
def build_arena_tem_trace_meta(  # --------------------------------------------
    batch: Batch,
) -> dict[str, object]:
    """Return ground-truth observation-id trajectory from an arena replay batch.

    Args:
        batch: Arena replay batch containing ``"trajectory_observation_id"``
            with shape ``(B, T_max)``.

    Returns:
        Nested dict keyed by ``"target/observation_id"`` with the full
        trajectory moved to CPU.
    """
    root_key, leaf_key = TARGET_OBSERVATION_ID_META_KEY.split("/", maxsplit=1)
    return {
        root_key: {
            leaf_key: batch["trajectory_observation_id"].detach().cpu(),
        },
    }


# =============================================================================
__all__ = [
    "ARENA_TEM_TRACE_FIELDS",
    "ARENA_TEM_TRACE_WORLD_OBS_ID",
    "ARENA_TEM_TRACE_IS_REVISIT",
    "ARENA_TEM_TRACE_PRED_POST",
    "ARENA_TEM_TRACE_PRED_RECALL",
    "ARENA_TEM_TRACE_PRED_PATH",
    "select_arena_tem_trace_fields",
    "TARGET_OBSERVATION_ID_META_KEY",
    "build_arena_tem_trace_meta",
]
