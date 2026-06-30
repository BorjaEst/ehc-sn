"""TODO"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from ehc_sn.tasks.arena.evaluation import coerce_observation_ids
from ehc_sn.tasks.mazehard.runtime import PATH_ID as _O_ID
from ehc_sn.traces import TraceField, TraceValue
from ehc_sn.traces.keys import MAZEHARD_META_KEY_GT_OVERLAY
from ehc_sn.traces.observer import StepContext
from ehc_sn.types import Batch

# =============================================================================
# Getter functions
# =============================================================================


def _prediction_overlay_from_task_logits(task_logits: Tensor) -> TraceValue:
    pred = torch.argmax(task_logits, dim=-1)  # (B, S)
    return (pred == _O_ID).to(torch.uint8)


def _get_maze_hard_prediction_overlay_q_halting(
    ctx: StepContext,
) -> TraceValue:
    return _prediction_overlay_from_task_logits(
        ctx.record.outputs.task_output.task_logits
    )


def build_mazehard_ehc_trace_meta(batch: Batch) -> dict[str, object]:
    """Return out-of-band trace metadata required by MazeHard+EHP figures."""
    root_key, leaf_key = MAZEHARD_META_KEY_GT_OVERLAY.split("/", maxsplit=1)
    return {
        root_key: {
            leaf_key: (batch["labels"] == _O_ID).to(torch.uint8),
        },
    }


# =============================================================================
# Named field objects
# =============================================================================

_MAZE_HARD_EHP_TRACE_PREDICTION_OVERLAY_Q_HALTING = TraceField(
    name="pred/prediction_overlay",
    get=_get_maze_hard_prediction_overlay_q_halting,
)

MAZE_HARD_EHP_Q_HALTING_TRACE_FIELDS: tuple[TraceField, ...] = (
    _MAZE_HARD_EHP_TRACE_PREDICTION_OVERLAY_Q_HALTING,
)


# =============================================================================

def _get_world_observation_id(ctx: StepContext) -> TraceValue:
    obs_id: Tensor = ctx.record.batch["observation_id"]
    return coerce_observation_ids(obs_id)


def _get_is_revisit(ctx: StepContext) -> TraceValue:
    is_revisit: Tensor | None = ctx.record.batch.get("is_revisit")
    if is_revisit is None:
        return None
    return is_revisit.view(-1).bool()


def _get_pred_obs_id_post(ctx: StepContext) -> TraceValue:
    return ctx.record.outputs.backbone_output.obs_logits[0].argmax(dim=-1)


def _get_pred_obs_id_recall(ctx: StepContext) -> TraceValue:
    return ctx.record.outputs.backbone_output.obs_logits[1].argmax(dim=-1)


def _get_pred_obs_id_path(ctx: StepContext) -> TraceValue:
    return ctx.record.outputs.backbone_output.obs_logits[2].argmax(dim=-1)


# =============================================================================
ARENA_EHP_TRACE_WORLD_OBS_ID = TraceField(
    name="world_step/observation_id",
    get=_get_world_observation_id,
)

ARENA_EHP_TRACE_IS_REVISIT = TraceField(
    name="protocol/is_revisit",
    get=_get_is_revisit,
)

ARENA_EHP_TRACE_PRED_POST = TraceField(
    name="pred/observation_id/post",
    get=_get_pred_obs_id_post,
)

ARENA_EHP_TRACE_PRED_RECALL = TraceField(
    name="pred/observation_id/recall",
    get=_get_pred_obs_id_recall,
)

ARENA_EHP_TRACE_PRED_PATH = TraceField(
    name="pred/observation_id/path",
    get=_get_pred_obs_id_path,
)

ARENA_EHP_TRACE_FIELDS: tuple[TraceField, ...] = (
    ARENA_EHP_TRACE_WORLD_OBS_ID,
    ARENA_EHP_TRACE_IS_REVISIT,
    ARENA_EHP_TRACE_PRED_POST,
    ARENA_EHP_TRACE_PRED_RECALL,
    ARENA_EHP_TRACE_PRED_PATH,
)
"""All Arena EHP trace fields in canonical order."""


# =============================================================================
def select_arena_ehc_trace_fields(
    include_keys: Iterable[str] | None,
) -> tuple[TraceField, ...]:
    """Return the Arena EHP trace fields matching a requested key set."""
    if include_keys is None:
        return ARENA_EHP_TRACE_FIELDS
    requested = set(include_keys)
    return tuple(f for f in ARENA_EHP_TRACE_FIELDS if f.name in requested)


# =============================================================================
__all__ = [
    "build_mazehard_ehc_trace_meta",
    "MAZE_HARD_EHP_Q_HALTING_TRACE_FIELDS",
    "ARENA_EHP_TRACE_FIELDS",
    "ARENA_EHP_TRACE_WORLD_OBS_ID",
    "ARENA_EHP_TRACE_IS_REVISIT",
    "ARENA_EHP_TRACE_PRED_POST",
    "ARENA_EHP_TRACE_PRED_RECALL",
    "ARENA_EHP_TRACE_PRED_PATH",
    "select_arena_ehc_trace_fields",
]
